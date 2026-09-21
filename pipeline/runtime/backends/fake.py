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
    SeedanceClient,
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
    BackendExecutionError,
    ReferenceSelectionContext,
    ReferenceSelectionResult,
    RunnerBackend,
    RunnerCancelledError,
    SeedanceGenerationContext,
    SeedanceGenerationResult,
    SeedanceLikeClient,
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
from ..runner_helpers import (
    _combined_static_references,
    _effective_reference_budget,
    _fake_produced_memory,
    _fake_prompt,
    _fake_references,
    _fake_visual_status,
    _image_asset_record,
    _log,
    _maybe_force_animation_prompt,
    _relative,
)


class FakeExecutionBackend:
    name = "fake"

    def __init__(self, store: ProjectStore) -> None:
        self.store = store

    def execute(self, context: ShotExecutionContext) -> ShotExecutionResult:
        visual_status = _fake_visual_status(context.shot)
        references = _fake_references(context.previous_shots)
        combined_references = _combined_static_references(context, references)
        prompt = _fake_prompt(context.input_snapshot, visual_status, references)
        produced_memory = _fake_produced_memory(context.shot, context.attempt_id)
        video_asset_id = f"vid_{context.shot['shot_id']}_{context.attempt_id}"
        keyframe_asset_id = f"img_{context.shot['shot_id']}_{context.attempt_id}_001"
        assets = self._write_fake_media(context, video_asset_id, keyframe_asset_id)
        task_id = f"fake-{context.project_id}-{context.shot['shot_id']}-{context.attempt_id}"
        logs = [
            _log("planning_visual_elements", "fake visual element status generated"),
            _log("selecting_visual_references", f"{len(references)} fake references selected"),
            _log("submitting", "fake Seedance request prepared"),
            _log("extracting_keyframes", "fake keyframe asset registered"),
        ]
        return ShotExecutionResult(
            visual_element_status=visual_status,
            selected_references=references,
            produced_visual_memory=produced_memory,
            submitted_prompt=prompt,
            request={
                "runner": self.name,
                "content_order": [
                    "text",
                    *[reference.get("reference_id") or reference.get("id") for reference in combined_references],
                ],
                "submitted_prompt": prompt,
            },
            response={"runner": self.name, "task_id": task_id, "status": "completed"},
            raw_video_asset_id=video_asset_id,
            assets=assets,
            task_id=task_id,
            usage={"estimated_cny": 0.0},
            postprocess={"fake": True},
            logs=logs,
        )

    def plan_visual_elements(self, context: VisualMemoryPlanContext) -> VisualMemoryPlanResult:
        visual_status = _fake_visual_status(context.shot)
        return VisualMemoryPlanResult(
            visual_element_status=visual_status,
            details={"visual_element_record": {"registry_after": visual_status, "fake": True}},
            logs=[_log("planning_visual_elements", "fake visual element status generated")],
        )

    def select_historical_references(self, context: ReferenceSelectionContext) -> ReferenceSelectionResult:
        budget = _effective_reference_budget(context)
        references = _fake_references(context.previous_shots)[: budget["effective_max_retrieved_frames"]]
        return ReferenceSelectionResult(
            selected_references=references,
            prompt_context=_fake_prompt(context.input_snapshot, context.visual_element_status, references),
            details={"fake": True, "reference_budget": budget},
            logs=[
                _log("selecting_visual_references", f"{len(references)} fake references selected"),
                _log(
                    "selecting_visual_references",
                    (
                        "static image budget: "
                        f"predefined={budget['predefined_reference_count']}, "
                        f"sink={budget['effective_sink_frame_count']}, "
                        f"retrieved_limit={budget['effective_max_retrieved_frames']}"
                    ),
                ),
            ],
        )

    def compose_seedance_prompt(self, context: SeedanceGenerationContext) -> SeedancePromptCompositionResult:
        prompt = _maybe_force_animation_prompt(
            context.prompt_context or _fake_prompt(context.input_snapshot, context.visual_element_status, context.selected_references),
            context.input_snapshot,
            context.bundle,
        )
        return SeedancePromptCompositionResult(
            submitted_prompt=prompt,
            request_content_summary=[
                {"type": "text", "text": prompt},
                *[
                    {"type": "image_asset", "asset_id": reference.get("asset_id"), "metadata": reference}
                    for reference in _combined_static_references(context, context.selected_references)
                ],
            ],
            details={"fake": True},
            logs=[_log("composing_prompt", "fake Seedance prompt composed locally")],
        )

    def generate_seedance_video(self, context: SeedanceGenerationContext) -> SeedanceGenerationResult:
        video_asset_id = f"vid_{context.shot['shot_id']}_{context.attempt_id}"
        keyframe_asset_id = f"img_{context.shot['shot_id']}_{context.attempt_id}_001"
        assets = self._write_fake_media(
            ShotExecutionContext(
                project_id=context.project_id,
                bundle=context.bundle,
                shot=context.shot,
                previous_shots=context.previous_shots,
                attempt_id=context.attempt_id,
                attempt_dir=context.attempt_dir,
                input_snapshot=context.input_snapshot,
                started_at=_now(),
                should_cancel=context.should_cancel,
            ),
            video_asset_id,
            keyframe_asset_id,
        )
        task_id = f"fake-{context.project_id}-{context.shot['shot_id']}-{context.attempt_id}"
        combined_references = _combined_static_references(context, context.selected_references)
        prompt = context.prompt_context or _fake_prompt(context.input_snapshot, context.visual_element_status, context.selected_references)
        return SeedanceGenerationResult(
            submitted_prompt=prompt,
            request={
                "runner": self.name,
                "content_order": ["text"] + [reference.get("reference_id") or reference.get("id") for reference in combined_references],
                "submitted_prompt": prompt,
            },
            response={"runner": self.name, "task_id": task_id, "status": "completed"},
            raw_video_asset_id=video_asset_id,
            assets={video_asset_id: assets[video_asset_id]},
            task_id=task_id,
            usage={"estimated_cny": 0.0},
            logs=[_log("submitting", "fake Seedance request prepared")],
        )

    def maintain_keyframes(self, context: VideoPostprocessContext) -> VideoPostprocessResult:
        keyframe_asset_id = f"img_{context.shot['shot_id']}_{context.attempt_id}_001"
        project_dir = self.store.project_dir(context.project_id)
        image_path = project_dir / "assets" / "images" / f"{keyframe_asset_id}.jpg"
        if not image_path.exists():
            image_path.write_bytes(b"fake image placeholder\n")
        produced_memory = _fake_produced_memory(context.shot, context.attempt_id)
        return VideoPostprocessResult(
            assets={
                keyframe_asset_id: _image_asset_record(
                    project_dir,
                    image_path,
                    keyframe_asset_id,
                    source_type="fake_produced_keyframe",
                    shot_id=context.shot["shot_id"],
                    attempt_id=context.attempt_id,
                    metadata={"fake": True},
                )
            },
            produced_visual_memory=produced_memory,
            postprocess={"fake": True},
            logs=[_log("extracting_keyframes", "fake keyframe asset registered")],
        )

    def _write_fake_media(
        self,
        context: ShotExecutionContext,
        video_asset_id: str,
        keyframe_asset_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        project_dir = self.store.project_dir(context.project_id)
        video_path = project_dir / "assets" / "videos" / f"{video_asset_id}.mp4"
        image_path = project_dir / "assets" / "images" / f"{keyframe_asset_id}.jpg"
        video_path.write_bytes(b"fake video placeholder\n")
        image_path.write_bytes(b"fake image placeholder\n")
        return {
            video_asset_id: {
                "kind": "video",
                "path": _relative(project_dir, video_path),
                "created_at": _now(),
                "source": {
                    "type": "fake_seedance_output",
                    "shot_id": context.shot["shot_id"],
                    "attempt_id": context.attempt_id,
                },
                "metadata": {"duration_seconds": 0, "has_audio": False, "fake": True},
            },
            keyframe_asset_id: {
                "kind": "image",
                "path": _relative(project_dir, image_path),
                "created_at": _now(),
                "source": {
                    "type": "fake_produced_keyframe",
                    "shot_id": context.shot["shot_id"],
                    "attempt_id": context.attempt_id,
                },
                "metadata": {"fake": True},
            },
        }
