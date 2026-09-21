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
    BackendExecutionError,
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

class StepFlowMixin:

    def _wait_before_next_step(
        self,
        project_id: str,
        scope: str,
        shot_id: str,
        from_step: str,
        to_step: str,
    ) -> None:
        bundle = self.store.get_project(project_id)
        delay = int(
            bundle.get("settings", {})
            .get("generation", {})
            .get("auto_run_step_review_delay_seconds", 30)
            or 0
        )
        if delay <= 0 or self.backend.name == "fake" or getattr(self.backend, "dry_run", False):
            return
        deadline = time.time() + delay
        self.store.update_project_json(
            project_id,
            {
                "status": "waiting_retry",
                "active_shot_id": shot_id,
                "run_state": {
                    "status": "waiting_next_step",
                    "scope": scope,
                    "shot_id": shot_id,
                    "from_step": from_step,
                    "to_step": to_step,
                    "delay_seconds": delay,
                    "deadline_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(deadline)),
                    "updated_at": _now(),
                },
            },
        )
        while time.time() < deadline:
            if self._cancel_event(project_id).is_set():
                raise RunnerCancelledError("Project run was interrupted")
            time.sleep(min(0.5, max(0.0, deadline - time.time())))
        self.store.update_project_json(
            project_id,
            {
                "status": "running",
                "active_shot_id": shot_id,
                "run_state": {
                    "status": "running",
                    "scope": scope,
                    "shot_id": shot_id,
                    "step": to_step,
                    "updated_at": _now(),
                },
            },
        )

    def _wait_before_retry(
        self,
        project_id: str,
        shot_id: str,
        step: str,
        *,
        attempt_index: int,
        total_attempts: int,
    ) -> None:
        delay = DEFAULT_STEP_RETRY_DELAY_SECONDS
        if delay <= 0 or self.backend.name == "fake" or getattr(self.backend, "dry_run", False):
            return
        deadline = time.time() + delay
        self.store.update_project_json(
            project_id,
            {
                "status": "waiting_next_step",
                "active_shot_id": shot_id,
                "run_state": {
                    "status": "waiting_retry",
                    "scope": "step_retry",
                    "shot_id": shot_id,
                    "step": step,
                    "attempt": attempt_index,
                    "total_attempts": total_attempts,
                    "delay_seconds": delay,
                    "deadline_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(deadline)),
                    "updated_at": _now(),
                },
            },
        )
        while time.time() < deadline:
            if self._cancel_event(project_id).is_set():
                raise RunnerCancelledError("Project run was interrupted")
            time.sleep(min(0.5, max(0.0, deadline - time.time())))
        self.store.update_project_json(
            project_id,
            {
                "status": "running",
                "active_shot_id": shot_id,
                "run_state": {
                    "status": "running",
                    "scope": "step_retry",
                    "shot_id": shot_id,
                    "step": step,
                    "updated_at": _now(),
                },
            },
        )

    def _run_shot_steps_without_lock(
        self,
        project_id: str,
        shot_id: str,
        *,
        apply_review_delay: bool = False,
        scope: str = "run_shot",
    ) -> bool:
        while True:
            bundle = self.store.get_project(project_id)
            shot = _shot_by_id(bundle["shots"], shot_id)
            if shot.get("state", {}).get("status") == "completed":
                status = "running" if scope == "run_all" else "completed"
                self._set_run_state(project_id, {"status": status, "scope": scope, "shot_id": shot_id})
                return True
            step = _next_step_for_shot(shot)
            if (
                apply_review_delay
                and step == "seedance_generation"
                and _requires_seedance_confirmation(bundle)
            ):
                self._set_run_state(
                    project_id,
                    {
                        "status": "paused_for_confirmation",
                        "scope": scope,
                        "shot_id": shot_id,
                        "step": step,
                        "message": "Seedance generation requires manual Run Step confirmation.",
                    },
                )
                self.store.update_project_json(project_id, {"status": "partial", "active_shot_id": shot_id})
                return False
            self._run_step_without_lock(project_id, shot_id, step, keep_project_running=True, scope=scope)
            if apply_review_delay and step == "visual_plan":
                updated = self.store.get_project(project_id)
                if _auto_reflect_visual_plan_enabled(updated) and not _reference_selection_naive(updated):
                    try:
                        self._reflect_visual_plan_without_lock(project_id, shot_id, allow_running_project=True)
                    except Exception as exc:
                        self._record_visual_plan_reflection_failure(project_id, shot_id, exc)
            if apply_review_delay:
                updated = self.store.get_project(project_id)
                updated_shot = _shot_by_id(updated["shots"], shot_id)
                if updated_shot.get("state", {}).get("status") != "completed":
                    next_step = _next_step_for_shot(updated_shot)
                    self._wait_before_next_step(project_id, scope, shot_id, step, next_step)

    def _run_step_without_lock(
        self,
        project_id: str,
        shot_id: str,
        step: str,
        *,
        keep_project_running: bool = False,
        scope: str = "run_step",
    ) -> None:
        if step not in STEP_SEQUENCE:
            raise InvalidProjectError(f"Unknown step: {step}")
        bundle = self.store.get_project(project_id)
        shots = bundle["shots"]
        shot = _shot_by_id(shots, shot_id)
        max_attempts = _algorithm_step_max_attempts(bundle)
        index = int(shot["order_index"])
        for predecessor in shots[:index]:
            if predecessor.get("state", {}).get("status") != "completed":
                raise InvalidProjectError(f"Cannot run {shot_id} before completing {predecessor['shot_id']}")
        _require_prior_steps(shot, step)

        if _is_running_status(shot.get("state", {}).get("status")):
            raise InvalidProjectError(f"Shot {shot_id} is already running")
        if step == "visual_plan":
            self._archive_current_attempt_if_needed(project_id, shot)
        if index + 1 < len(shots):
            self.store._reset_from_index(project_id, index + 1)
            bundle = self.store.get_project(project_id)
            shots = bundle["shots"]
            shot = _shot_by_id(shots, shot_id)
        attempt_id = self._ensure_current_attempt(project_id, shot, bundle, step)
        attempt_dir = self.store.attempt_dir(project_id, shot_id, attempt_id)
        _reset_downstream_steps(shot, step)
        initial_status = "keyframe_maintaining_queued" if step == "keyframe_maintaining" else f"{step}_running"
        self._set_run_state(
            project_id,
            {
                "status": initial_status,
                "scope": scope,
                "shot_id": shot_id,
                "step": step,
                "attempt_id": attempt_id,
            },
        )
        self._set_step_status(project_id, shot, step, initial_status)
        self.store.update_project_json(project_id, {"status": "running", "active_shot_id": shot_id})

        try:
            if step in RETRYABLE_ALGORITHM_STEPS:
                self._run_retryable_algorithm_step(project_id, shot_id, attempt_id, step, max_attempts)
            elif step == "seedance_prompt":
                self._run_seedance_prompt_step(project_id, shot_id, attempt_id)
            elif step == "keyframe_maintaining":
                self._run_keyframe_maintaining_with_gpu_lock(
                    project_id,
                    shot_id,
                    attempt_id,
                    max_attempts=max_attempts,
                    scope=scope,
                )
            if self._cancel_event(project_id).is_set():
                raise RunnerCancelledError("Project run was interrupted")
            updated = _shot_by_id(self.store.get_project(project_id)["shots"], shot_id)
            if updated.get("state", {}).get("status") != "completed":
                self.store.update_project_json(
                    project_id,
                    {"status": "running" if keep_project_running else "partial", "active_shot_id": shot_id if keep_project_running else None},
                )
        except Exception as exc:
            if isinstance(exc, RunnerCancelledError) or self._cancel_event(project_id).is_set():
                self._record_step_terminal(project_id, shot_id, attempt_id, step, exc, status="interrupted")
                self._set_run_state(project_id, {"status": "stopped", "scope": "run_step", "shot_id": shot_id, "step": step})
            else:
                self._record_step_terminal(project_id, shot_id, attempt_id, step, exc, status=f"{step}_failed")
                self._set_run_state(project_id, {"status": "failed", "scope": "run_step", "shot_id": shot_id, "step": step})
            raise InvalidProjectError(str(exc)) from exc
        finally:
            attempt_dir.mkdir(parents=True, exist_ok=True)

    def _run_retryable_algorithm_step(
        self,
        project_id: str,
        shot_id: str,
        attempt_id: str,
        step: str,
        max_attempts: int,
    ) -> None:
        total_attempts = max(1, int(max_attempts or DEFAULT_ALGORITHM_STEP_MAX_ATTEMPTS))
        retry_logs: List[Dict[str, Any]] = []
        retry_attempts: List[Dict[str, Any]] = []
        log_step = _terminal_log_step_for_step(step, f"{step}_failed")
        for attempt_index in range(1, total_attempts + 1):
            try:
                if step == "visual_plan":
                    self._run_visual_plan_step(project_id, shot_id, attempt_id)
                elif step == "reference_selection":
                    self._run_reference_selection_step(project_id, shot_id, attempt_id)
                elif step == "seedance_generation":
                    self._run_seedance_generation_step(project_id, shot_id, attempt_id)
                else:
                    raise InvalidProjectError(f"Step {step} is not retryable")
                if retry_logs:
                    retry_logs.append(
                        _log(log_step, f"{_step_retry_label(step)} attempt {attempt_index}/{total_attempts} succeeded.")
                    )
                    self._prepend_step_logs(project_id, shot_id, attempt_id, step, retry_logs)
                return
            except Exception as exc:
                if isinstance(exc, RunnerCancelledError) or self._cancel_event(project_id).is_set():
                    raise
                retry_attempts.append(_retry_attempt_record(attempt_index, exc))
                if attempt_index >= total_attempts:
                    _attach_step_retry_details(exc, retry_attempts)
                    raise
                retry_logs.append(
                    _log(
                        log_step,
                        (
                            f"{_step_retry_label(step)} attempt {attempt_index}/{total_attempts} failed; "
                            f"waiting {DEFAULT_STEP_RETRY_DELAY_SECONDS}s before retrying: {type(exc).__name__}: {exc}"
                        ),
                    )
                )
                self._append_step_logs(
                    project_id,
                    shot_id,
                    attempt_id,
                    step,
                    [retry_logs[-1]],
                )
                self._wait_before_retry(project_id, shot_id, step, attempt_index=attempt_index, total_attempts=total_attempts)

    def _prepend_step_logs(
        self,
        project_id: str,
        shot_id: str,
        attempt_id: str,
        step: str,
        logs: List[Dict[str, Any]],
    ) -> None:
        attempt_dir = self.store.attempt_dir(project_id, shot_id, attempt_id)
        step_log_path = attempt_dir / step / "logs.jsonl"
        existing = read_jsonl_atomic_safe(step_log_path)
        write_jsonl_atomic(step_log_path, [*logs, *existing])
        write_jsonl_atomic(attempt_dir / "logs.jsonl", _all_step_logs(attempt_dir))

    def _append_step_logs(
        self,
        project_id: str,
        shot_id: str,
        attempt_id: str,
        step: str,
        logs: List[Dict[str, Any]],
    ) -> None:
        attempt_dir = self.store.attempt_dir(project_id, shot_id, attempt_id)
        step_log_path = attempt_dir / step / "logs.jsonl"
        existing = read_jsonl_atomic_safe(step_log_path)
        write_jsonl_atomic(step_log_path, [*existing, *logs])
        write_jsonl_atomic(attempt_dir / "logs.jsonl", _all_step_logs(attempt_dir))

    def _reflect_visual_plan_without_lock(
        self,
        project_id: str,
        shot_id: str,
        *,
        allow_running_project: bool = False,
    ) -> None:
        bundle = self.store.get_project(project_id)
        shot = _shot_by_id(bundle["shots"], shot_id)
        steps = shot.get("steps") or {}
        if steps.get("visual_plan", {}).get("status") != "visual_plan_completed":
            raise InvalidProjectError(f"Shot {shot_id} has no completed visual plan to reflect")
        attempt = shot.get("attempt") or {}
        source_rows = attempt.get("visual_element_status") or []
        if not source_rows:
            raise InvalidProjectError(f"Shot {shot_id} has no visual element rows to reflect")
        request = _visual_plan_reflection_request(bundle, shot, source_rows)
        result = reflect_visual_plan_with_llm(request)
        revised_rows = apply_visual_plan_reflection(source_rows, result.rows)
        reflection_payload = {
            "updated_at": _now(),
            "summary": result.summary,
            "rows": result.rows,
            "warnings": result.warnings,
            "metadata": result.metadata,
            "prompt": result.prompt,
            "raw_response": result.raw_response,
        }
        self.store.apply_visual_plan_reflection(
            project_id,
            shot_id,
            revised_rows,
            reflection_payload,
            allow_running_project=allow_running_project,
        )

    def _record_visual_plan_reflection_failure(self, project_id: str, shot_id: str, exc: Exception) -> None:
        bundle = self.store.get_project(project_id)
        shot = _shot_by_id(bundle["shots"], shot_id)
        attempt_id = shot.get("state", {}).get("current_attempt_id")
        if not attempt_id:
            return
        attempt_dir = self.store.attempt_dir(project_id, shot_id, str(attempt_id))
        logs = read_jsonl_atomic_safe(attempt_dir / "visual_plan" / "logs.jsonl")
        logs.append(
            {
                "time": _now(),
                "step": "visual_plan_reflection",
                "message": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            }
        )
        write_jsonl_atomic(attempt_dir / "visual_plan" / "logs.jsonl", logs)
        write_jsonl_atomic(attempt_dir / "logs.jsonl", _all_step_logs(attempt_dir))


    def _run_visual_plan_step(self, project_id: str, shot_id: str, attempt_id: str) -> None:
        context = self._visual_context(project_id, shot_id, attempt_id)
        if _reference_selection_skips_visual_plan(context.bundle):
            result = _empty_visual_plan_result(_reference_selection_mode(context.bundle))
        else:
            result = self.backend.plan_visual_elements(context)
        attempt_dir = context.attempt_dir
        write_json_atomic(attempt_dir / "visual_plan" / "visual_element_status.json", result.visual_element_status)
        write_json_atomic(attempt_dir / "visual_plan" / "details.json", result.details)
        write_jsonl_atomic(attempt_dir / "visual_plan" / "logs.jsonl", result.logs)
        write_json_atomic(attempt_dir / "visual_element_status.json", result.visual_element_status)
        write_json_atomic(attempt_dir / "visual_element_details.json", result.details)
        write_jsonl_atomic(attempt_dir / "logs.jsonl", _all_step_logs(attempt_dir))
        self._update_attempt(project_id, shot_id, attempt_id, visual_element_status=result.visual_element_status)
        shot = _shot_by_id(self.store.get_project(project_id)["shots"], shot_id)
        self._set_step_status(project_id, shot, "visual_plan", "visual_plan_completed")

    def _run_reference_selection_step(self, project_id: str, shot_id: str, attempt_id: str) -> None:
        context = self._reference_context(project_id, shot_id, attempt_id)
        if _reference_selection_naive(context.bundle):
            result = _select_naive_top_k_references(context, self.store)
        elif _reference_selection_sink_recent(context.bundle):
            result = _select_sink_recent_memory_references(context, self.store)
        elif _reference_selection_disabled(context.bundle):
            prompt_context = _prompt_for_generation_mode(
                _visual_reference_prompt_context(
                    context.bundle,
                    context.shot,
                    context.input_snapshot,
                    context.visual_element_status,
                    [],
                ),
                str(context.input_snapshot.get("generation_mode") or "default"),
                default_last_frame_continuity=_uses_default_last_frame_continuity(context.shot, context.input_snapshot),
            )
            result = ReferenceSelectionResult(
                selected_references=[],
                prompt_context=prompt_context,
                details={
                    "skipped": True,
                    "selection_mode": "none",
                    "reason": "Historical reference selection disabled by project setting.",
                },
                logs=[
                    _log(
                        "selecting_visual_references",
                        "Historical Reference Selection skipped because Reference selection is set to none.",
                    )
                ],
            )
        else:
            result = self.backend.select_historical_references(context)
        attempt_dir = context.attempt_dir
        write_json_atomic(attempt_dir / "reference_selection" / "selected_references.json", result.selected_references)
        write_json_atomic(attempt_dir / "reference_selection" / "details.json", result.details)
        write_json_atomic(attempt_dir / "reference_selection" / "prompt_context.json", {"prompt_context": result.prompt_context})
        write_jsonl_atomic(attempt_dir / "reference_selection" / "logs.jsonl", result.logs)
        write_json_atomic(attempt_dir / "selected_references.json", result.selected_references)
        write_jsonl_atomic(attempt_dir / "logs.jsonl", _all_step_logs(attempt_dir))
        self._update_attempt(project_id, shot_id, attempt_id, selected_references=result.selected_references)
        shot = _shot_by_id(self.store.get_project(project_id)["shots"], shot_id)
        self._set_step_status(project_id, shot, "reference_selection", "reference_selection_completed")

    def _run_seedance_prompt_step(self, project_id: str, shot_id: str, attempt_id: str) -> None:
        context = self._seedance_context(project_id, shot_id, attempt_id, use_composed_prompt=False)
        result = self.backend.compose_seedance_prompt(context)
        attempt_dir = context.attempt_dir
        write_json_atomic(attempt_dir / "seedance_prompt" / "prompt.json", {"submitted_prompt": result.submitted_prompt})
        (attempt_dir / "seedance_prompt" / "prompt.txt").write_text(result.submitted_prompt, encoding="utf-8")
        write_json_atomic(
            attempt_dir / "seedance_prompt" / "request_content_summary.json",
            {"content": result.request_content_summary},
        )
        write_json_atomic(attempt_dir / "seedance_prompt" / "details.json", result.details)
        write_jsonl_atomic(attempt_dir / "seedance_prompt" / "logs.jsonl", result.logs)
        write_jsonl_atomic(attempt_dir / "logs.jsonl", _all_step_logs(attempt_dir))
        self._update_attempt(project_id, shot_id, attempt_id, prompt={"submitted_prompt": result.submitted_prompt})
        shot = _shot_by_id(self.store.get_project(project_id)["shots"], shot_id)
        self._set_step_status(project_id, shot, "seedance_prompt", "seedance_prompt_completed")

    def _run_seedance_generation_step(self, project_id: str, shot_id: str, attempt_id: str) -> None:
        context = self._seedance_context(project_id, shot_id, attempt_id)
        result = self.backend.generate_seedance_video(context)
        attempt_dir = context.attempt_dir
        self.store.update_asset_manifest(project_id, result.assets)
        write_json_atomic(attempt_dir / "seedance_generation" / "prompt.json", {"submitted_prompt": result.submitted_prompt})
        (attempt_dir / "seedance_generation" / "prompt.txt").write_text(result.submitted_prompt, encoding="utf-8")
        write_json_atomic(attempt_dir / "seedance_generation" / "request.json", result.request)
        write_json_atomic(attempt_dir / "seedance_generation" / "response.json", result.response)
        write_jsonl_atomic(attempt_dir / "seedance_generation" / "logs.jsonl", result.logs)
        write_json_atomic(attempt_dir / "request.json", result.request)
        write_json_atomic(attempt_dir / "response.json", result.response)
        write_jsonl_atomic(attempt_dir / "logs.jsonl", _all_step_logs(attempt_dir))
        self._update_attempt(
            project_id,
            shot_id,
            attempt_id,
            prompt={"submitted_prompt": result.submitted_prompt},
            seedance={"task_id": result.task_id, "request_path": "seedance_generation/request.json", "response_path": "seedance_generation/response.json", "usage": result.usage},
            outputs={"raw_video_asset_id": result.raw_video_asset_id},
        )
        shot = _shot_by_id(self.store.get_project(project_id)["shots"], shot_id)
        self._set_step_status(project_id, shot, "seedance_generation", "seedance_generation_completed")

    def _run_keyframe_maintaining_with_gpu_lock(
        self,
        project_id: str,
        shot_id: str,
        attempt_id: str,
        *,
        max_attempts: int,
        scope: str,
    ) -> None:
        queue_logs = [
            _log("waiting_for_gpu_keyframe_slot", "Keyframe Maintaining queued for the local GPU lock.")
        ]
        self._append_step_logs(project_id, shot_id, attempt_id, "keyframe_maintaining", queue_logs)
        try:
            lease = self.gpu_lock.acquire(
                project_id,
                shot_id,
                should_cancel=lambda: self._cancel_event(project_id).is_set(),
            )
        except GpuLockCancelled as exc:
            raise RunnerCancelledError("Project run was interrupted while waiting for the keyframe GPU lock") from exc

        try:
            acquired_log = _log("waiting_for_gpu_keyframe_slot", "Keyframe Maintaining acquired the local GPU lock.")
            queue_logs.append(acquired_log)
            self._append_step_logs(project_id, shot_id, attempt_id, "keyframe_maintaining", [acquired_log])
            shot = _shot_by_id(self.store.get_project(project_id)["shots"], shot_id)
            self._set_run_state(
                project_id,
                {
                    "status": "keyframe_maintaining_running",
                    "scope": scope,
                    "shot_id": shot_id,
                    "step": "keyframe_maintaining",
                    "attempt_id": attempt_id,
                },
            )
            self._set_step_status(project_id, shot, "keyframe_maintaining", "keyframe_maintaining_running")
            self._run_keyframe_maintaining_step(
                project_id,
                shot_id,
                attempt_id,
                max_attempts=max_attempts,
                pre_logs=queue_logs,
            )
        finally:
            lease.release()
            self._append_step_logs(
                project_id,
                shot_id,
                attempt_id,
                "keyframe_maintaining",
                [_log("waiting_for_gpu_keyframe_slot", "Keyframe Maintaining released the local GPU lock.")],
            )

    def _run_keyframe_maintaining_step(
        self,
        project_id: str,
        shot_id: str,
        attempt_id: str,
        *,
        max_attempts: int,
        pre_logs: List[Dict[str, Any]] | None = None,
    ) -> None:
        retry_logs: List[Dict[str, Any]] = []
        retry_attempts: List[Dict[str, Any]] = []
        total_attempts = max(1, int(max_attempts or DEFAULT_ALGORITHM_STEP_MAX_ATTEMPTS))
        for attempt_index in range(1, total_attempts + 1):
            context = self._postprocess_context(project_id, shot_id, attempt_id)
            try:
                result = self.backend.maintain_keyframes(context)
                if attempt_index > 1:
                    retry_logs.append(
                        _log(
                            "annotating_visual_memory",
                            f"Keyframe Maintaining full attempt {attempt_index}/{total_attempts} succeeded.",
                        )
                )
                break
            except Exception as exc:
                if isinstance(exc, RunnerCancelledError) or self._cancel_event(project_id).is_set():
                    raise
                retry_attempts.append(
                    {
                        "attempt": attempt_index,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "details": getattr(exc, "details", None),
                    }
                )
                if attempt_index >= total_attempts:
                    _attach_keyframe_retry_details(exc, retry_attempts)
                    raise
                retry_logs.append(
                    _log(
                        "annotating_visual_memory",
                        (
                            f"Keyframe Maintaining full attempt {attempt_index}/{total_attempts} failed; "
                            f"waiting {DEFAULT_STEP_RETRY_DELAY_SECONDS}s before retrying: {type(exc).__name__}: {exc}"
                        ),
                    )
                )
                self._append_step_logs(
                    project_id,
                    shot_id,
                    attempt_id,
                    "keyframe_maintaining",
                    [retry_logs[-1]],
                )
                self._wait_before_retry(
                    project_id,
                    shot_id,
                    "keyframe_maintaining",
                    attempt_index=attempt_index,
                    total_attempts=total_attempts,
                )
        attempt_dir = context.attempt_dir
        self.store.update_asset_manifest(project_id, result.assets)
        keyframe_details = read_json(attempt_dir / "visual_element_details.json", {})
        if not isinstance(keyframe_details, dict):
            keyframe_details = {}
        write_json_atomic(attempt_dir / "keyframe_maintaining" / "produced_visual_memory.json", result.produced_visual_memory)
        write_json_atomic(attempt_dir / "keyframe_maintaining" / "postprocess.json", result.postprocess)
        write_json_atomic(attempt_dir / "keyframe_maintaining" / "details.json", keyframe_details)
        write_jsonl_atomic(attempt_dir / "keyframe_maintaining" / "logs.jsonl", [*(pre_logs or []), *retry_logs, *result.logs])
        write_json_atomic(attempt_dir / "produced_visual_memory.json", result.produced_visual_memory)
        write_jsonl_atomic(attempt_dir / "logs.jsonl", _all_step_logs(attempt_dir))
        self._update_attempt(
            project_id,
            shot_id,
            attempt_id,
            postprocess=result.postprocess,
            produced_visual_memory=result.produced_visual_memory,
        )
        shot = _shot_by_id(self.store.get_project(project_id)["shots"], shot_id)
        self._set_step_status(project_id, shot, "keyframe_maintaining", "keyframe_maintaining_completed")
        self._update_project_after_completion(
            project_id,
            context.video_asset_id,
            thumbnail_asset_id=_thumbnail_from_memory(result.produced_visual_memory),
        )
        self.store.refresh_memory_state(project_id)

    def _visual_context(self, project_id: str, shot_id: str, attempt_id: str) -> VisualMemoryPlanContext:
        bundle = self.store.get_project(project_id)
        shot = _shot_by_id(bundle["shots"], shot_id)
        attempt_dir = self.store.attempt_dir(project_id, shot_id, attempt_id)
        return VisualMemoryPlanContext(
            project_id=project_id,
            bundle=bundle,
            shot=shot,
            previous_shots=bundle["shots"][: int(shot["order_index"])],
            attempt_id=attempt_id,
            attempt_dir=attempt_dir,
            input_snapshot=read_json(attempt_dir / "input_snapshot.json", deepcopy(shot["inputs"])),
        )

    def _reference_context(self, project_id: str, shot_id: str, attempt_id: str) -> ReferenceSelectionContext:
        base = self._visual_context(project_id, shot_id, attempt_id)
        visual_status = read_json(base.attempt_dir / "visual_plan" / "visual_element_status.json", [])
        details = read_json(base.attempt_dir / "visual_plan" / "details.json", {})
        return ReferenceSelectionContext(
            **base.__dict__,
            visual_element_status=visual_status if isinstance(visual_status, list) else [],
            visual_element_details=details if isinstance(details, dict) else {},
        )

    def _seedance_context(
        self,
        project_id: str,
        shot_id: str,
        attempt_id: str,
        *,
        use_composed_prompt: bool = True,
    ) -> SeedanceGenerationContext:
        base = self._visual_context(project_id, shot_id, attempt_id)
        visual_status = read_json(base.attempt_dir / "visual_plan" / "visual_element_status.json", [])
        references = read_json(base.attempt_dir / "reference_selection" / "selected_references.json", [])
        prompt_context = read_json(base.attempt_dir / "reference_selection" / "prompt_context.json", {})
        reference_details = read_json(base.attempt_dir / "reference_selection" / "details.json", {})
        composed_prompt = read_json(base.attempt_dir / "seedance_prompt" / "prompt.json", {})
        prompt_value = None
        if use_composed_prompt and isinstance(composed_prompt, dict):
            prompt_value = composed_prompt.get("submitted_prompt")
        if prompt_value is None and isinstance(reference_details, dict) and reference_details.get("manual_edit"):
            prompt_value = _prompt_for_generation_mode(
                _visual_reference_prompt_context(
                    base.bundle,
                    base.shot,
                    base.input_snapshot,
                    visual_status if isinstance(visual_status, list) else [],
                    references if isinstance(references, list) else [],
                ),
                str(base.input_snapshot.get("generation_mode") or "default"),
                default_last_frame_continuity=_uses_default_last_frame_continuity(base.shot, base.input_snapshot),
            )
        if prompt_value is None and isinstance(prompt_context, dict):
            prompt_value = prompt_context.get("prompt_context")
        return SeedanceGenerationContext(
            project_id=base.project_id,
            bundle=base.bundle,
            shot=base.shot,
            previous_shots=base.previous_shots,
            attempt_id=base.attempt_id,
            attempt_dir=base.attempt_dir,
            input_snapshot=base.input_snapshot,
            visual_element_status=visual_status if isinstance(visual_status, list) else [],
            selected_references=references if isinstance(references, list) else [],
            prompt_context=str(prompt_value) if prompt_value is not None else None,
            should_cancel=self._cancel_event(project_id).is_set,
        )

    def _postprocess_context(self, project_id: str, shot_id: str, attempt_id: str) -> VideoPostprocessContext:
        base = self._visual_context(project_id, shot_id, attempt_id)
        attempt = read_json(base.attempt_dir / "attempt.json", {})
        raw_video_asset_id = str((attempt.get("outputs") or {}).get("raw_video_asset_id") or "")
        if not raw_video_asset_id:
            raise BackendExecutionError("Seedance video generation has no raw output asset")
        visual_status = read_json(base.attempt_dir / "visual_plan" / "visual_element_status.json", [])
        return VideoPostprocessContext(
            project_id=project_id,
            bundle=base.bundle,
            shot=base.shot,
            attempt_id=attempt_id,
            attempt_dir=base.attempt_dir,
            video_asset_id=raw_video_asset_id,
            video_path=self.store.asset_path(project_id, raw_video_asset_id),
            visual_element_status=visual_status if isinstance(visual_status, list) else [],
            dry_run=getattr(self.backend, "dry_run", False),
        )


    def _run_shot_without_lock(self, project_id: str, shot_id: str) -> None:
        bundle = self.store.get_project(project_id)
        shots = bundle["shots"]
        shot_by_id = {shot["shot_id"]: shot for shot in shots}
        if shot_id not in shot_by_id:
            raise InvalidProjectError(f"Shot not found: {shot_id}")
        shot = shot_by_id[shot_id]
        index = int(shot["order_index"])
        for predecessor in shots[:index]:
            if predecessor.get("state", {}).get("status") != "completed":
                raise InvalidProjectError(f"Cannot run {shot_id} before completing {predecessor['shot_id']}")
        if shot.get("state", {}).get("status") not in {"draft", "failed", "interrupted"}:
            raise InvalidProjectError(f"Shot {shot_id} is locked in state {shot.get('state', {}).get('status')}")

        now = _now()
        attempt_id = self.store.next_attempt_id(project_id, shot_id)
        attempt_dir = self.store.attempt_dir(project_id, shot_id, attempt_id)
        attempt_dir.mkdir(parents=True, exist_ok=True)

        shot["state"]["status"] = "running"
        shot["state"]["current_attempt_id"] = attempt_id
        shot["state"]["updated_at"] = now
        self.store.write_shot(project_id, shot)
        self.store.update_project_json(project_id, {"status": "running", "active_shot_id": shot_id})

        input_snapshot = deepcopy(shot["inputs"])
        input_snapshot["project_settings"] = bundle["settings"]
        context = ShotExecutionContext(
            project_id=project_id,
            bundle=bundle,
            shot=shot,
            previous_shots=shots[:index],
            attempt_id=attempt_id,
            attempt_dir=attempt_dir,
            input_snapshot=input_snapshot,
            started_at=now,
            should_cancel=self._cancel_event(project_id).is_set,
        )
        try:
            result = self.backend.execute(context)
            if context.should_cancel():
                raise RunnerCancelledError("Project run was interrupted")
        except Exception as exc:
            if isinstance(exc, RunnerCancelledError) or context.should_cancel():
                self._record_interrupted_attempt(context, exc)
            else:
                self._record_failed_attempt(context, exc)
            raise InvalidProjectError(str(exc)) from exc

        self.store.update_asset_manifest(project_id, result.assets)
        write_json_atomic(attempt_dir / "visual_element_status.json", result.visual_element_status)
        write_json_atomic(attempt_dir / "selected_references.json", result.selected_references)
        write_json_atomic(attempt_dir / "produced_visual_memory.json", result.produced_visual_memory)
        write_json_atomic(attempt_dir / "request.json", result.request)
        write_json_atomic(attempt_dir / "response.json", result.response)
        write_jsonl_atomic(attempt_dir / "logs.jsonl", result.logs)
        attempt = {
            "schema_version": SCHEMA_VERSION,
            "attempt_id": attempt_id,
            "shot_id": shot_id,
            "revision": shot["state"].get("revision", 1),
            "status": "completed",
            "runner": {"backend": self.backend.name},
            "created_at": now,
            "updated_at": _now(),
            "input_snapshot": input_snapshot,
            "visual_element": {
                "state_before_path": "visual_element_status.json",
                "state_after_path": "visual_element_status.json",
                "details_path": "visual_element_details.json",
            },
            "references": {"selected_references_path": "selected_references.json"},
            "prompt": {"submitted_prompt": result.submitted_prompt},
            "seedance": {
                "task_id": result.task_id,
                "request_path": "request.json",
                "response_path": "response.json",
                "usage": result.usage,
            },
            "outputs": {"raw_video_asset_id": result.raw_video_asset_id},
            "postprocess": {
                "produced_visual_memory_path": "produced_visual_memory.json",
                "keyframe_profile": bundle["settings"].get("keyframes", {}).get("profile", DEFAULT_KEYFRAME_PROFILE),
                **result.postprocess,
            },
            "logs_path": "logs.jsonl",
            "error": None,
        }
        write_json_atomic(attempt_dir / "attempt.json", attempt)

        shot["state"]["status"] = "completed"
        shot["state"]["updated_at"] = _now()
        self.store.write_shot(project_id, shot)
        self._update_project_after_completion(
            project_id,
            result.raw_video_asset_id,
            thumbnail_asset_id=_thumbnail_from_memory(result.produced_visual_memory),
        )
        self.store.refresh_memory_state(project_id)
