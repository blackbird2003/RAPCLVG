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

def _fake_visual_status(shot: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "element_id": f"element-{shot['shot_id']}-001",
            "name": _short_name(shot["inputs"]["video_prompt"]),
            "type": "scene",
            "introduced_at": shot["shot_id"],
            "notes": "Fake runner placeholder element.",
            "status": "new",
            "reason": "Created by the fake runner to validate UI and persistence.",
        }
    ]


def _initial_step_state() -> Dict[str, Dict[str, Any]]:
    return {step: {"status": "draft", "started_at": None, "updated_at": None} for step in STEP_SEQUENCE}


def _is_running_status(status: Any) -> bool:
    value = str(status or "")
    return value == "running" or value.endswith("_running") or value.endswith("_queued")


def _shot_by_id(shots: List[Dict[str, Any]], shot_id: str) -> Dict[str, Any]:
    for shot in shots:
        if shot.get("shot_id") == shot_id:
            return shot
    raise InvalidProjectError(f"Shot not found: {shot_id}")


def _next_step_for_shot(shot: Dict[str, Any]) -> str:
    status = str(shot.get("state", {}).get("status") or "draft")
    if status == "draft":
        return STEP_SEQUENCE[0]
    if status == "completed":
        raise InvalidProjectError(f"Shot {shot.get('shot_id')} is already completed")
    if status == "interrupted":
        steps = shot.get("steps") or {}
        for step in STEP_SEQUENCE:
            if str(steps.get(step, {}).get("status")) != f"{step}_completed":
                return step
        return STEP_SEQUENCE[-1]
    for index, step in enumerate(STEP_SEQUENCE):
        if status in {f"{step}_failed", f"{step}_running"}:
            return step
        if status == f"{step}_completed":
            return STEP_SEQUENCE[index + 1] if index + 1 < len(STEP_SEQUENCE) else step
    return STEP_SEQUENCE[0]


def _require_prior_steps(shot: Dict[str, Any], step: str) -> None:
    steps = shot.get("steps") or {}
    for prior in STEP_SEQUENCE[: STEP_SEQUENCE.index(step)]:
        if str(steps.get(prior, {}).get("status")) != f"{prior}_completed":
            raise InvalidProjectError(f"Cannot run {step} before {prior} is completed")


def _reset_downstream_steps(shot: Dict[str, Any], step: str) -> None:
    steps = shot.setdefault("steps", _initial_step_state())
    for downstream in STEP_SEQUENCE[STEP_SEQUENCE.index(step) + 1 :]:
        steps[downstream] = {"status": "draft", "started_at": None, "updated_at": None}


def _requires_seedance_confirmation(bundle: Dict[str, Any]) -> bool:
    return bool(
        bundle.get("settings", {})
        .get("generation", {})
        .get("require_human_confirmation_before_seedance", False)
    )


def _auto_reflect_visual_plan_enabled(bundle: Dict[str, Any]) -> bool:
    return bool(
        bundle.get("settings", {})
        .get("generation", {})
        .get("auto_reflect_visual_plan", False)
    )


def _project_status_from_prefix(bundle: Dict[str, Any]) -> str:
    completed_prefix = _completed_prefix(bundle.get("shots", []))
    shot_count = int((bundle.get("project") or {}).get("shot_count") or len(bundle.get("shots", [])))
    if completed_prefix >= shot_count and shot_count > 0:
        return "completed"
    if completed_prefix > 0:
        return "partial"
    return "draft"


from .logging import _short_name
