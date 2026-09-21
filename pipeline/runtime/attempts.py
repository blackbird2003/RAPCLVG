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

class AttemptLifecycleMixin:

    def _ensure_current_attempt(
        self,
        project_id: str,
        shot: Dict[str, Any],
        bundle: Dict[str, Any],
        step: str,
    ) -> str:
        state = shot.setdefault("state", {})
        attempt_id = state.get("current_attempt_id")
        if attempt_id:
            return str(attempt_id)
        attempt_id = self.store.next_attempt_id(project_id, shot["shot_id"])
        attempt_dir = self.store.attempt_dir(project_id, shot["shot_id"], attempt_id)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        for step in STEP_SEQUENCE:
            (attempt_dir / step).mkdir(exist_ok=True)
        input_snapshot = deepcopy(shot["inputs"])
        input_snapshot["project_settings"] = bundle["settings"]
        attempt = {
            "schema_version": SCHEMA_VERSION,
            "attempt_id": attempt_id,
            "shot_id": shot["shot_id"],
            "revision": state.get("revision", 1),
            "status": "running",
            "runner": {"backend": self.backend.name},
            "created_at": _now(),
            "updated_at": _now(),
            "input_snapshot": input_snapshot,
            "prompt": {"submitted_prompt": ""},
            "seedance": {"usage": {}},
            "outputs": {},
            "postprocess": {},
            "error": None,
        }
        write_json_atomic(attempt_dir / "attempt.json", attempt)
        write_json_atomic(attempt_dir / "input_snapshot.json", input_snapshot)
        write_json_atomic(attempt_dir / "steps.json", shot.get("steps") or {})
        state["current_attempt_id"] = attempt_id
        state["updated_at"] = _now()
        self.store.write_shot(project_id, shot)
        self._inherit_completed_step_artifacts(project_id, shot, attempt_id, step)
        return attempt_id

    def _inherit_completed_step_artifacts(
        self,
        project_id: str,
        shot: Dict[str, Any],
        attempt_id: str,
        step: str,
    ) -> None:
        prior_steps = [
            prior
            for prior in STEP_SEQUENCE[: STEP_SEQUENCE.index(step)]
            if str((shot.get("steps") or {}).get(prior, {}).get("status")) == f"{prior}_completed"
        ]
        if not prior_steps:
            return
        source_id = self._latest_archived_attempt_id(shot)
        if not source_id:
            return
        source_dir = self.store.attempt_dir(project_id, shot["shot_id"], source_id)
        target_dir = self.store.attempt_dir(project_id, shot["shot_id"], attempt_id)
        if not source_dir.exists():
            return

        for prior in prior_steps:
            _copy_attempt_path(source_dir / prior, target_dir / prior)

        root_files = []
        if "visual_plan" in prior_steps:
            root_files.extend(("visual_element_status.json", "visual_element_details.json"))
        if "reference_selection" in prior_steps:
            root_files.append("selected_references.json")
        if "seedance_generation" in prior_steps:
            root_files.extend(("request.json", "response.json"))
        for filename in root_files:
            _copy_attempt_path(source_dir / filename, target_dir / filename)

        source_attempt = read_json(source_dir / "attempt.json", {})
        target_attempt = read_json(target_dir / "attempt.json", {})
        if not isinstance(source_attempt, dict) or not isinstance(target_attempt, dict):
            return
        if "visual_plan" in prior_steps and "visual_element_status" in source_attempt:
            target_attempt["visual_element_status"] = deepcopy(source_attempt.get("visual_element_status"))
        if "reference_selection" in prior_steps and "selected_references" in source_attempt:
            target_attempt["selected_references"] = deepcopy(source_attempt.get("selected_references"))
        if "seedance_prompt" in prior_steps and isinstance(source_attempt.get("prompt"), dict):
            target_attempt["prompt"] = deepcopy(source_attempt["prompt"])
        if "seedance_generation" in prior_steps:
            for key in ("prompt", "seedance", "outputs"):
                if isinstance(source_attempt.get(key), dict):
                    target_attempt[key] = deepcopy(source_attempt[key])
        target_attempt["updated_at"] = _now()
        write_json_atomic(target_dir / "attempt.json", target_attempt)

    def _latest_archived_attempt_id(self, shot: Dict[str, Any]) -> str | None:
        archived = shot.get("archived_attempt_ids") or []
        for attempt_id in reversed(archived):
            if attempt_id:
                return str(attempt_id)
        return None

    def _archive_current_attempt_if_needed(self, project_id: str, shot: Dict[str, Any]) -> None:
        current = shot.get("state", {}).get("current_attempt_id")
        if not current:
            return
        archived = shot.setdefault("archived_attempt_ids", [])
        if current not in archived:
            archived.append(current)
        shot["state"]["current_attempt_id"] = None
        shot["state"]["updated_at"] = _now()
        self.store.write_shot(project_id, shot)

    def _set_step_status(self, project_id: str, shot: Dict[str, Any], step: str, status: str) -> None:
        now = _now()
        steps = shot.setdefault("steps", _initial_step_state())
        steps.setdefault(step, {})["status"] = status
        if status.endswith("_running") and not steps[step].get("started_at"):
            steps[step]["started_at"] = now
        elif not steps[step].get("started_at"):
            steps[step]["started_at"] = now
        steps[step]["updated_at"] = now
        shot["state"]["status"] = "completed" if status == "keyframe_maintaining_completed" else status
        shot["state"]["updated_at"] = now
        self.store.write_shot(project_id, shot)
        attempt_id = shot.get("state", {}).get("current_attempt_id")
        if attempt_id:
            attempt_dir = self.store.attempt_dir(project_id, shot["shot_id"], str(attempt_id))
            write_json_atomic(attempt_dir / "steps.json", steps)
            attempt = read_json(attempt_dir / "attempt.json", {})
            if isinstance(attempt, dict):
                attempt["status"] = shot["state"]["status"]
                attempt["updated_at"] = now
                write_json_atomic(attempt_dir / "attempt.json", attempt)


    def _update_attempt(self, project_id: str, shot_id: str, attempt_id: str, **updates: Any) -> None:
        attempt_dir = self.store.attempt_dir(project_id, shot_id, attempt_id)
        attempt = read_json(attempt_dir / "attempt.json", {})
        if not isinstance(attempt, dict):
            attempt = {"schema_version": SCHEMA_VERSION, "attempt_id": attempt_id, "shot_id": shot_id}
        for key, value in updates.items():
            if key in {"prompt", "seedance", "outputs", "postprocess"} and isinstance(value, dict):
                current = attempt.get(key)
                if not isinstance(current, dict):
                    current = {}
                current.update(value)
                attempt[key] = current
            elif key in {"visual_element_status", "selected_references"}:
                attempt[key] = value
            else:
                attempt[key] = value
        attempt["updated_at"] = _now()
        write_json_atomic(attempt_dir / "attempt.json", attempt)

    def _record_step_terminal(
        self,
        project_id: str,
        shot_id: str,
        attempt_id: str,
        step: str,
        exc: Exception,
        *,
        status: str,
    ) -> None:
        attempt_dir = self.store.attempt_dir(project_id, shot_id, attempt_id)
        error = {"type": type(exc).__name__, "message": str(exc)}
        log_step = _terminal_log_step_for_step(step, status)
        logs = [_log(log_step, str(exc))]
        (attempt_dir / step).mkdir(parents=True, exist_ok=True)
        details = getattr(exc, "details", None)
        if isinstance(details, dict) and details:
            existing_details = read_json(attempt_dir / step / "details.json", {})
            if not isinstance(existing_details, dict):
                existing_details = {}
            merged_details = {**existing_details, "error_details": details}
            if step == "visual_plan":
                merged_details.setdefault("visual_element_record", details)
                write_json_atomic(attempt_dir / "visual_element_details.json", merged_details)
            elif step == "keyframe_maintaining":
                write_json_atomic(attempt_dir / "visual_element_details.json", merged_details)
            write_json_atomic(attempt_dir / step / "details.json", merged_details)
            logs.extend(_attempt_diagnostic_logs(details, log_step))
        logs.append(
            {
                "time": _now(),
                "step": log_step,
                "message": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            }
        )
        existing_logs = read_jsonl_atomic_safe(attempt_dir / step / "logs.jsonl")
        write_jsonl_atomic(attempt_dir / step / "logs.jsonl", [*existing_logs, *logs])
        write_jsonl_atomic(attempt_dir / "logs.jsonl", _all_step_logs(attempt_dir))
        attempt = read_json(attempt_dir / "attempt.json", {})
        if not isinstance(attempt, dict):
            attempt = {"schema_version": SCHEMA_VERSION, "attempt_id": attempt_id, "shot_id": shot_id}
        attempt["status"] = status
        attempt["updated_at"] = _now()
        attempt["error"] = error
        write_json_atomic(attempt_dir / "attempt.json", attempt)
        shot = _shot_by_id(self.store.get_project(project_id)["shots"], shot_id)
        shot.setdefault("steps", _initial_step_state()).setdefault(step, {})["status"] = status
        now = _now()
        if not shot["steps"][step].get("started_at"):
            shot["steps"][step]["started_at"] = now
        shot["steps"][step]["updated_at"] = now
        shot["state"]["status"] = status
        shot["state"]["updated_at"] = now
        self.store.write_shot(project_id, shot)
        self.store.update_project_json(project_id, {"status": status, "active_shot_id": None})

    def _record_failed_attempt(self, context: ShotExecutionContext, exc: Exception) -> None:
        self._record_terminal_attempt(context, exc, status="failed")

    def _record_interrupted_attempt(self, context: ShotExecutionContext, exc: Exception) -> None:
        self._record_terminal_attempt(context, exc, status="interrupted")

    def _record_terminal_attempt(self, context: ShotExecutionContext, exc: Exception, *, status: str) -> None:
        attempt_dir = context.attempt_dir
        error = {"type": type(exc).__name__, "message": str(exc)}
        logs = [
            _log(status, str(exc)),
            {
                "time": _now(),
                "step": "traceback",
                "message": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            },
        ]
        visual_status = read_json(attempt_dir / "visual_element_status.json", [])
        selected_references = read_json(attempt_dir / "selected_references.json", [])
        produced_memory = read_json(attempt_dir / "produced_visual_memory.json", [])
        request = read_json(attempt_dir / "request.json", {"runner": self.backend.name})
        if not isinstance(request, dict):
            request = {"runner": self.backend.name}
        request["error"] = error
        write_json_atomic(attempt_dir / "visual_element_status.json", visual_status if isinstance(visual_status, list) else [])
        write_json_atomic(attempt_dir / "selected_references.json", selected_references if isinstance(selected_references, list) else [])
        write_json_atomic(attempt_dir / "produced_visual_memory.json", produced_memory if isinstance(produced_memory, list) else [])
        write_json_atomic(attempt_dir / "request.json", request)
        write_json_atomic(attempt_dir / "response.json", {"runner": self.backend.name, "status": status, "error": error})
        write_jsonl_atomic(attempt_dir / "logs.jsonl", logs)
        write_json_atomic(
            attempt_dir / "attempt.json",
            {
                "schema_version": SCHEMA_VERSION,
                "attempt_id": context.attempt_id,
                "shot_id": context.shot["shot_id"],
                "revision": context.shot["state"].get("revision", 1),
                "status": status,
                "runner": {"backend": self.backend.name},
                "created_at": context.started_at,
                "updated_at": _now(),
                "input_snapshot": context.input_snapshot,
                "visual_element": {
                    "state_before_path": "visual_element_status.json",
                    "details_path": "visual_element_details.json",
                },
                "references": {"selected_references_path": "selected_references.json"},
                "prompt": {"submitted_prompt": ""},
                "seedance": {"request_path": "request.json", "response_path": "response.json", "usage": {}},
                "outputs": {},
                "postprocess": {},
                "logs_path": "logs.jsonl",
                "error": error,
            },
        )
        shot = context.shot
        shot["state"]["status"] = status
        shot["state"]["updated_at"] = _now()
        self.store.write_shot(context.project_id, shot)
        self.store.update_project_json(
            context.project_id,
            {"status": status, "active_shot_id": None},
        )


    def _update_project_after_completion(
        self,
        project_id: str,
        video_asset_id: str,
        *,
        thumbnail_asset_id: str | None = None,
    ) -> None:
        bundle = self.store.get_project(project_id)
        completed_prefix = _completed_prefix(bundle["shots"])
        status = "completed" if completed_prefix == len(bundle["shots"]) else "partial"
        final_asset_id, assembly = self._assemble_completed_prefix(project_id, bundle, completed_prefix)
        write_json_atomic(
            self.store.project_dir(project_id) / "final" / "assembly.json",
            assembly,
        )
        updates = {
            "status": status,
            "active_shot_id": None,
            "completed_prefix": completed_prefix,
            "current_final_video_asset_id": final_asset_id,
            "cost_summary": {
                "estimated_cny": 0.0,
                "seedance_tasks": completed_prefix,
            },
        }
        if thumbnail_asset_id:
            updates["thumbnail_asset_id"] = thumbnail_asset_id
        self.store.update_project_json(
            project_id,
            updates,
        )
        if completed_prefix > 0 and final_asset_id:
            completed_shot = bundle["shots"][completed_prefix - 1]
            self._set_shot_current_final(project_id, completed_shot, final_asset_id)
