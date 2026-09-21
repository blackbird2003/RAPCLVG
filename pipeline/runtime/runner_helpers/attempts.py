from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import requests

from ...agentic.visual_element_memory import (
    DEFAULT_VISUAL_ELEMENT_MODEL,
    VISUAL_ELEMENT_MAX_OUTPUT_TOKENS,
    _call_ark_chat,
    _extract_json_object,
)
from ...agentic.visual_plan_reflection import (
    VisualPlanReflectionRequest,
    apply_visual_plan_reflection,
    reflect_visual_plan_with_llm,
)
from ...generators.seedance_client import (
    SeedanceError,
    extract_last_frame_url,
    extract_task_id,
    extract_video_url,
)
from ...keyframes.settings import DEFAULT_PROFILE as DEFAULT_KEYFRAME_PROFILE
from ...models import RunConfig, ShotSpec
from ...prompting import (
    SMOOTH_CONTINUATION_INSTRUCTION,
    audio_item_with_metadata,
    compose_prompt,
    escape_prompt_line,
    image_item_with_metadata,
    story_prompt_outline,
    video_item_with_metadata,
)
from ..constants import (
    DEFAULT_ALGORITHM_STEP_MAX_ATTEMPTS,
    DEFAULT_STEP_RETRY_DELAY_SECONDS,
    FORCE_ANIMATION_PROMPT_PREFIX,
    MEDIA_PREFLIGHT_TIMEOUT_SECONDS,
    NAIVE_TOP_K_SELECTION_MODE,
    SINK_RECENT_KEYFRAME_PROFILE,
    SINK_RECENT_MEMORY_FIX,
    SINK_RECENT_MEMORY_MAX_SIZE,
    SINK_RECENT_MEMORY_SELECTION_MODE,
    STYLE_REFERENCE_GUIDANCE,
)
from ..contracts import (
    ReferenceSelectionContext,
    ReferenceSelectionResult,
    RunnerCancelledError,
    SeedanceGenerationContext,
    SeedanceGenerationResult,
    SeedancePromptCompositionResult,
    ShotExecutionContext,
    ShotExecutionResult,
    VideoPostprocessor,
    VideoPostprocessContext,
    VideoPostprocessResult,
    VisualMemoryPlanContext,
    VisualMemoryPlanResult,
    VisualMemoryPlanner,
)
from ..domain import (
    MAX_REFERENCE_IMAGES,
    MIN_SMOOTH_REFERENCE_SECONDS,
    SCHEMA_VERSION,
    STEP_SEQUENCE,
)
from ..gpu_lock import GpuLockCancelled, KeyframeGpuLock
from ..json_store import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic
from ..locks import GlobalRunLock
from ..project_store import InvalidProjectError, ProjectStore, _completed_prefix, _now

def _visual_plan_reflection_request(
    bundle: Dict[str, Any],
    shot: Dict[str, Any],
    source_rows: List[Dict[str, Any]],
) -> VisualPlanReflectionRequest:
    current_order = int(shot.get("order_index", 0))
    story = bundle.get("story") or {}
    project = bundle.get("project") or {}
    return VisualPlanReflectionRequest(
        story_title=str(project.get("name") or story.get("title") or ""),
        project_overview=str(story.get("overview") or ""),
        current_shot_id=str(shot["shot_id"]),
        shots_until_current=[
            {
                "shot_id": item.get("shot_id"),
                "scene_num": item.get("scene_num"),
                "shot_num": item.get("shot_num"),
                "video_prompt": (item.get("inputs") or {}).get("video_prompt", ""),
                "is_cut": (item.get("inputs") or {}).get("is_cut"),
                "generation_mode": (item.get("inputs") or {}).get("generation_mode"),
            }
            for item in bundle.get("shots", [])[: current_order + 1]
        ],
        current_visual_plan=source_rows,
    )


def _all_step_logs(attempt_dir: Path) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    for step in STEP_SEQUENCE:
        logs.extend(read_jsonl_atomic_safe(attempt_dir / step / "logs.jsonl"))
    return logs


def read_jsonl_atomic_safe(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    from ..json_store import read_jsonl

    return read_jsonl(path)


def _copy_attempt_path(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _plan_result_from_details(details: Dict[str, Any]):
    from pipeline.agentic.visual_element_memory import (
        ExistingElementState,
        NewElement,
        ShotDecision,
        VisualElement,
        VisualElementPlanResult,
        _normalize_element_type,
    )

    record = details.get("visual_element_record") or {}
    decision_data = record.get("decision") or {}
    inserted_data = record.get("inserted") or []
    decision = ShotDecision(
        existing_element_states=[
            ExistingElementState(
                id=str(item.get("id")),
                state=item.get("state"),
                reason=str(item.get("reason") or ""),
            )
            for item in decision_data.get("existing_element_states", [])
        ],
        new_elements=[
            NewElement(
                name=str(item.get("name") or ""),
                type=_normalize_element_type(item.get("type")),
                notes=str(item.get("notes") or ""),
            )
            for item in decision_data.get("new_elements", [])
        ],
        shot_notes=[str(item) for item in decision_data.get("shot_notes", [])],
    )
    inserted = [
        VisualElement(
            id=str(item.get("id")),
            name=str(item.get("name") or ""),
            type=_normalize_element_type(item.get("type")),
            introduced_at=str(item.get("introduced_at") or ""),
            notes=str(item.get("notes") or ""),
        )
        for item in inserted_data
    ]
    report_record = {
        "visual_element_decision": decision_data,
        "visual_element_inserted": inserted_data,
        "visual_element_plan": _visual_plan_from_record(record),
    }
    return VisualElementPlanResult(
        decision=decision,
        inserted_elements=inserted,
        report_record=report_record,
        record=record,
    )


def _visual_plan_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    registry = {str(item.get("id")): item for item in record.get("registry_after", []) if item.get("id")}
    groups = {
        "should_reference": [],
        "should_exclude": [],
        "optional_or_uncertain": [],
        "new_elements": [],
    }
    decision = record.get("decision") or {}
    for state in decision.get("existing_element_states", []):
        element = registry.get(str(state.get("id"))) or {}
        groups.setdefault(str(state.get("state")), []).append(
            {
                "id": state.get("id"),
                "name": element.get("name") or state.get("id"),
                "type": element.get("type") or "object",
                "reason": state.get("reason") or "",
            }
        )
    for element in record.get("inserted", []) or []:
        groups["new_elements"].append(
            {
                "id": element.get("id"),
                "name": element.get("name"),
                "type": element.get("type"),
                "reason": element.get("notes") or "",
            }
        )
    return groups
