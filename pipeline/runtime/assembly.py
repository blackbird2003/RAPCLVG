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

from ..agentic.visual_element_memory import (
    DEFAULT_VISUAL_ELEMENT_MODEL,
    VISUAL_ELEMENT_MAX_OUTPUT_TOKENS,
    _call_ark_chat,
    _extract_json_object,
)
from ..agentic.visual_plan_reflection import (
    VisualPlanReflectionRequest,
    apply_visual_plan_reflection,
    reflect_visual_plan_with_llm,
)
from ..generators.seedance_client import (
    SeedanceError,
    extract_last_frame_url,
    extract_task_id,
    extract_video_url,
)
from ..keyframes.settings import DEFAULT_PROFILE as DEFAULT_KEYFRAME_PROFILE
from ..models import RunConfig, ShotSpec
from ..prompting import (
    SMOOTH_CONTINUATION_INSTRUCTION,
    audio_item_with_metadata,
    compose_prompt,
    escape_prompt_line,
    image_item_with_metadata,
    story_prompt_outline,
    video_item_with_metadata,
)
from .constants import (
    DEFAULT_ALGORITHM_STEP_MAX_ATTEMPTS,
    DEFAULT_STEP_RETRY_DELAY_SECONDS,
    FORCE_ANIMATION_PROMPT_PREFIX,
    MEDIA_PREFLIGHT_TIMEOUT_SECONDS,
    NAIVE_TOP_K_SELECTION_MODE,
    RETRYABLE_ALGORITHM_STEPS,
    SINK_RECENT_KEYFRAME_PROFILE,
    SINK_RECENT_MEMORY_FIX,
    SINK_RECENT_MEMORY_MAX_SIZE,
    SINK_RECENT_MEMORY_SELECTION_MODE,
    STYLE_REFERENCE_GUIDANCE,
)
from .contracts import (
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
from .domain import (
    MAX_REFERENCE_IMAGES,
    MIN_SMOOTH_REFERENCE_SECONDS,
    SCHEMA_VERSION,
    STEP_SEQUENCE,
)
from .gpu_lock import GpuLockCancelled, KeyframeGpuLock
from .json_store import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic
from .locks import GlobalRunLock
from .project_store import InvalidProjectError, ProjectStore, _completed_prefix, _now
from .runner_helpers import *

class AssemblyMixin:

    def _rerun_assembly_without_lock(self, project_id: str) -> None:
        bundle = self.store.get_project(project_id)
        completed_prefix = _completed_prefix(bundle["shots"])
        if completed_prefix <= 0:
            raise InvalidProjectError("No completed shots are available for assembly")
        final_asset_id = None
        assembly: Dict[str, Any] = {}
        for prefix in range(1, completed_prefix + 1):
            if self._cancel_event(project_id).is_set():
                raise RunnerCancelledError("Project run was interrupted")
            bundle = self.store.get_project(project_id)
            final_asset_id, assembly = self._assemble_completed_prefix(
                project_id,
                bundle,
                prefix,
                force_full_rebuild=True,
            )
            write_json_atomic(
                self.store.project_dir(project_id) / "final" / "assembly.json",
                assembly,
            )
            completed_shot = bundle["shots"][prefix - 1]
            if final_asset_id:
                self._set_shot_current_final(project_id, completed_shot, final_asset_id)
            self.store.update_project_json(
                project_id,
                {
                    "completed_prefix": prefix,
                    "current_final_video_asset_id": final_asset_id,
                    "run_state": {
                        "status": "assembling",
                        "scope": "rerun_assembly",
                        "completed_prefix": prefix,
                        "total_prefix": completed_prefix,
                        "updated_at": _now(),
                    },
                },
            )
        status = "completed" if completed_prefix == len(bundle["shots"]) else "partial"
        self.store.update_project_json(
            project_id,
            {
                "status": status,
                "active_shot_id": None,
                "completed_prefix": completed_prefix,
                "current_final_video_asset_id": final_asset_id,
            },
        )

    def _set_shot_current_final(self, project_id: str, shot: Dict[str, Any], final_asset_id: str) -> None:
        state = shot.setdefault("state", {})
        state["current_final_video_asset_id"] = final_asset_id
        state["updated_at"] = _now()
        self.store.write_shot(project_id, shot)
        attempt_id = state.get("current_attempt_id")
        if not attempt_id:
            return
        attempt_dir = self.store.attempt_dir(project_id, shot["shot_id"], str(attempt_id))
        attempt = read_json(attempt_dir / "attempt.json", {})
        if isinstance(attempt, dict):
            outputs = attempt.get("outputs")
            if not isinstance(outputs, dict):
                outputs = {}
            outputs["current_final_video_asset_id"] = final_asset_id
            attempt["outputs"] = outputs
            attempt["updated_at"] = _now()
            write_json_atomic(attempt_dir / "attempt.json", attempt)

    def _assemble_completed_prefix(
        self,
        project_id: str,
        bundle: Dict[str, Any],
        completed_prefix: int,
        *,
        force_full_rebuild: bool = False,
    ) -> tuple[str | None, Dict[str, Any]]:
        raw_prefix = _prefix_video_assets(bundle["shots"])
        project_dir = self.store.project_dir(project_id)
        existing = _read_json(project_dir / "final" / "assembly.json")
        final_by_prefix = dict(existing.get("prefix_final_video_asset_ids") or {})
        assembly = {
            "schema_version": SCHEMA_VERSION,
            "runner": self.backend.name,
            "completed_prefix": completed_prefix,
            "prefix_video_asset_ids": raw_prefix,
            "prefix_final_video_asset_ids": final_by_prefix,
            "logs": list(existing.get("logs") or []),
        }
        if completed_prefix <= 0:
            return None, assembly
        completed_shots = bundle["shots"][:completed_prefix]
        last_shot_id = completed_shots[-1]["shot_id"]
        last_attempt_id = str((completed_shots[-1].get("attempt") or {}).get("attempt_id") or "current")
        final_asset_id = f"vid_final_{last_shot_id}_{last_attempt_id}"
        final_path = project_dir / "final" / f"prefix_{last_shot_id}_{last_attempt_id}.mp4"
        clips = _assembly_clips(self.store, project_id, completed_shots, bundle.get("assets", {}))
        log: Dict[str, Any] = {
            "time": _now(),
            "prefix_shot_id": last_shot_id,
            "final_asset_id": final_asset_id,
            "clip_count": len(clips),
            "source_strategy": "raw_full_rebuild" if force_full_rebuild else "raw_prefix_rebuild",
        }
        if _contains_placeholder_clip(clips):
            _write_placeholder_final(final_path, clips)
            log["mode"] = "placeholder"
        elif any(item["generation_mode"] == "smooth" for item in clips[1:]):
            from ..media.smooth_transition import SmoothVideoAssembler

            SmoothVideoAssembler()(clips, str(final_path))
            log["mode"] = "smooth_rife"
        elif len(clips) == 1:
            _concat_videos_ffmpeg([Path(clips[0]["output_video"])], final_path)
            log["mode"] = "single_clip_copy"
        else:
            from ..media.smooth_transition import SmoothVideoAssembler

            SmoothVideoAssembler()(clips, str(final_path))
            log["mode"] = "frame_accurate_reencode"
        final_by_prefix[last_shot_id] = final_asset_id
        assembly["prefix_final_video_asset_ids"] = final_by_prefix
        assembly.setdefault("shot_current_final_video_asset_ids", {})[last_shot_id] = final_asset_id
        assembly["logs"].append(log)
        self.store.update_asset_manifest(
            project_id,
            {
                final_asset_id: {
                    "kind": "video",
                    "path": _relative(project_dir, final_path),
                    "created_at": _now(),
                    "source": {
                        "type": "final_prefix_assembly",
                        "completed_prefix": completed_prefix,
                        "last_shot_id": last_shot_id,
                    },
                    "metadata": {
                        "clip_count": len(clips),
                        "assembly_mode": log["mode"],
                        "raw_video_asset_ids": [item["raw_video_asset_id"] for item in clips],
                    },
                }
            },
        )
        return final_asset_id, assembly
