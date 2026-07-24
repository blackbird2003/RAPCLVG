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
from typing import Any, Callable, Dict, List, Protocol

import requests

from keyframe_settings import DEFAULT_PROFILE as DEFAULT_KEYFRAME_PROFILE
from seedance_client import DEFAULT_MODEL as DEFAULT_SEEDANCE_MODEL
from seedance_client import SeedanceClient, SeedanceError, extract_last_frame_url, extract_task_id, extract_video_url
from storymem_seedance.models import RunConfig, ShotSpec
from storymem_seedance.prompting import (
    SMOOTH_CONTINUATION_INSTRUCTION,
    compose_prompt,
    escape_prompt_line,
    image_item_with_metadata,
    story_prompt_outline,
    video_item_with_metadata,
)
from storymem_seedance.visual_element_memory import (
    DEFAULT_VISUAL_ELEMENT_MODEL,
    VISUAL_ELEMENT_MAX_OUTPUT_TOKENS,
    _call_ark_chat,
    _extract_json_object,
)
from storymem_seedance.visual_plan_reflection import (
    VisualPlanReflectionRequest,
    apply_visual_plan_reflection,
    reflect_visual_plan_with_llm,
)

from .domain import MAX_REFERENCE_IMAGES, SCHEMA_VERSION
from .domain import STEP_SEQUENCE
from .gpu_lock import GpuLockCancelled, KeyframeGpuLock
from .json_store import read_json, write_json_atomic, write_jsonl_atomic
from .locks import GlobalRunLock
from .project_store import InvalidProjectError, ProjectStore, _completed_prefix, _now


DEFAULT_ALGORITHM_STEP_MAX_ATTEMPTS = 5
DEFAULT_STEP_RETRY_DELAY_SECONDS = 60
RETRYABLE_ALGORITHM_STEPS = {"visual_plan", "reference_selection", "seedance_generation"}
MEDIA_PREFLIGHT_TIMEOUT_SECONDS = 15
FORCE_ANIMATION_PROMPT_PREFIX = "Top Priority Constraint:​ Regardless of the script description, all human faces—whether characters or images appearing on screens/photos—must be rendered in a non-photorealistic 2D animated style. Photorealistic faces are strictly prohibited; and all characters must be originally designed, do not directly use any copyrighted or trademarked characters."
STYLE_REFERENCE_GUIDANCE = "本张参考图中可能不含任何新视频所需的视觉元素，请将其作为视觉风格参考。"
EVAL_SUBMIT_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "submit_videogen_eval_to_a6000.py"
NAIVE_TOP_K_SELECTION_MODE = "naive_top_k"
STORYMEM_MEMORY_SELECTION_MODE = "storymem_memory"
STORYMEM_MEMORY_MAX_SIZE = 10
STORYMEM_MEMORY_FIX = 3
STORYMEM_KEYFRAME_PROFILE = "storymem_original"


@dataclass(frozen=True)
class ShotExecutionContext:
    project_id: str
    bundle: Dict[str, Any]
    shot: Dict[str, Any]
    previous_shots: List[Dict[str, Any]]
    attempt_id: str
    attempt_dir: Path
    input_snapshot: Dict[str, Any]
    started_at: str
    should_cancel: Callable[[], bool] = lambda: False


@dataclass(frozen=True)
class ShotExecutionResult:
    visual_element_status: List[Dict[str, Any]]
    selected_references: List[Dict[str, Any]]
    produced_visual_memory: List[Dict[str, Any]]
    submitted_prompt: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    raw_video_asset_id: str
    assets: Dict[str, Dict[str, Any]]
    task_id: str
    usage: Dict[str, Any] = field(default_factory=dict)
    postprocess: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class VideoPostprocessContext:
    project_id: str
    bundle: Dict[str, Any]
    shot: Dict[str, Any]
    attempt_id: str
    attempt_dir: Path
    video_asset_id: str
    video_path: Path
    visual_element_status: List[Dict[str, Any]]
    dry_run: bool


@dataclass(frozen=True)
class VideoPostprocessResult:
    assets: Dict[str, Dict[str, Any]]
    produced_visual_memory: List[Dict[str, Any]]
    postprocess: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class VisualMemoryPlanContext:
    project_id: str
    bundle: Dict[str, Any]
    shot: Dict[str, Any]
    previous_shots: List[Dict[str, Any]]
    attempt_id: str
    attempt_dir: Path
    input_snapshot: Dict[str, Any]


@dataclass(frozen=True)
class VisualMemoryPlanResult:
    visual_element_status: List[Dict[str, Any]]
    selected_references: List[Dict[str, Any]] = field(default_factory=list)
    prompt_context: str | None = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceSelectionContext:
    project_id: str
    bundle: Dict[str, Any]
    shot: Dict[str, Any]
    previous_shots: List[Dict[str, Any]]
    attempt_id: str
    attempt_dir: Path
    input_snapshot: Dict[str, Any]
    visual_element_status: List[Dict[str, Any]]
    visual_element_details: Dict[str, Any]


@dataclass(frozen=True)
class ReferenceSelectionResult:
    selected_references: List[Dict[str, Any]]
    prompt_context: str | None = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SeedancePromptCompositionResult:
    submitted_prompt: str
    request_content_summary: List[Dict[str, Any]]
    details: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SeedanceGenerationContext:
    project_id: str
    bundle: Dict[str, Any]
    shot: Dict[str, Any]
    previous_shots: List[Dict[str, Any]]
    attempt_id: str
    attempt_dir: Path
    input_snapshot: Dict[str, Any]
    visual_element_status: List[Dict[str, Any]]
    selected_references: List[Dict[str, Any]]
    prompt_context: str | None
    should_cancel: Callable[[], bool] = lambda: False


@dataclass(frozen=True)
class SeedanceGenerationResult:
    submitted_prompt: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    raw_video_asset_id: str
    assets: Dict[str, Dict[str, Any]]
    task_id: str
    usage: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)


class RunnerBackend(Protocol):
    name: str

    def execute(self, context: ShotExecutionContext) -> ShotExecutionResult:
        """Run one shot and return persisted-attempt payloads."""


class BackendExecutionError(RuntimeError):
    pass


class RunnerCancelledError(RuntimeError):
    pass


class SeedanceLikeClient(Protocol):
    def create_task(self, content, **kwargs):
        ...

    def wait_task(self, task_id, **kwargs):
        ...

    def download_video(self, url, output_path):
        ...


class VideoPostprocessor(Protocol):
    name: str

    def process(self, context: VideoPostprocessContext) -> VideoPostprocessResult:
        ...


class VisualMemoryPlanner(Protocol):
    name: str

    def plan(self, context: VisualMemoryPlanContext) -> VisualMemoryPlanResult:
        ...

    def select_references(self, context: ReferenceSelectionContext) -> ReferenceSelectionResult:
        ...


def create_runner(
    store: ProjectStore,
    backend_name: str = "fake",
    *,
    real_client: SeedanceLikeClient | None = None,
    real_visual_planner: VisualMemoryPlanner | None = None,
    real_postprocessor: VideoPostprocessor | None = None,
    real_submit: bool | None = None,
) -> "NotebookRunner":
    normalized = (backend_name or "fake").strip().lower()
    if normalized == "fake":
        return NotebookRunner(store, FakeExecutionBackend(store))
    if normalized == "real":
        submit_enabled = (
            _env_flag("VIDEOGEN_NOTEBOOK_REAL_SUBMIT", default=True)
            if real_submit is None
            else bool(real_submit)
        )
        coordinator = (
            NotebookVisualElementMemoryAdapter(store)
            if submit_enabled and (real_visual_planner is None or real_postprocessor is None)
            else None
        )
        return NotebookRunner(
            store,
            RealExecutionBackend(
                store,
                client=real_client,
                visual_planner=real_visual_planner or coordinator,
                postprocessor=real_postprocessor or coordinator,
                dry_run=not submit_enabled,
            ),
        )
    raise ValueError(f"Unknown videogen_notebook runner backend: {backend_name}")


class NotebookRunner:
    def __init__(self, store: ProjectStore, backend: RunnerBackend, *, gpu_lock: KeyframeGpuLock | None = None) -> None:
        self.store = store
        self.backend = backend
        self.lock = GlobalRunLock(store.locks_dir)
        self.gpu_lock = gpu_lock or KeyframeGpuLock()
        self._cancel_events: Dict[str, threading.Event] = {}
        self._cancel_lock = threading.Lock()

    def run_all(self, project_id: str, *, apply_review_delay: bool = False) -> Dict[str, Any]:
        cancel_event = self._reset_cancel_event(project_id)
        self.lock.acquire(project_id)
        try:
            self._set_run_state(project_id, {"status": "running", "scope": "run_all"})
            bundle = self.store.get_project(project_id)
            for shot in bundle["shots"]:
                if cancel_event.is_set():
                    raise RunnerCancelledError("Project run was interrupted")
                if shot.get("state", {}).get("status") != "completed":
                    completed = self._run_shot_steps_without_lock(
                        project_id,
                        shot["shot_id"],
                        apply_review_delay=apply_review_delay,
                        scope="run_all",
                    )
                    if not completed:
                        return self.store.get_project(project_id)
                    bundle = self.store.get_project(project_id)
            self._set_run_state(project_id, {"status": "completed", "scope": "run_all"})
            return self.store.get_project(project_id)
        finally:
            self.lock.release(project_id)
            self._clear_cancel_event(project_id)

    def run_shot(self, project_id: str, shot_id: str, *, apply_review_delay: bool = False) -> Dict[str, Any]:
        self._reset_cancel_event(project_id)
        self.lock.acquire(project_id)
        try:
            self._set_run_state(project_id, {"status": "running", "scope": "run_shot", "shot_id": shot_id})
            self._run_shot_steps_without_lock(
                project_id,
                shot_id,
                apply_review_delay=apply_review_delay,
                scope="run_shot",
            )
            return self.store.get_project(project_id)
        finally:
            self.lock.release(project_id)
            self._clear_cancel_event(project_id)

    def run_step(self, project_id: str, shot_id: str, step: str) -> Dict[str, Any]:
        self._reset_cancel_event(project_id)
        self.lock.acquire(project_id)
        try:
            self._set_run_state(project_id, {"status": "running", "scope": "run_step", "shot_id": shot_id, "step": step})
            self._run_step_without_lock(project_id, shot_id, step)
            self._set_run_state(project_id, {"status": "idle"})
            return self.store.get_project(project_id)
        finally:
            self.lock.release(project_id)
            self._clear_cancel_event(project_id)

    def reflect_visual_plan(self, project_id: str, shot_id: str) -> Dict[str, Any]:
        self._reset_cancel_event(project_id)
        self.lock.acquire(project_id)
        try:
            self._set_run_state(project_id, {"status": "reflecting_visual_plan", "scope": "reflect", "shot_id": shot_id})
            self.store.update_project_json(project_id, {"status": "running", "active_shot_id": shot_id})
            try:
                self._reflect_visual_plan_without_lock(project_id, shot_id, allow_running_project=True)
            except Exception as exc:
                self._record_visual_plan_reflection_failure(project_id, shot_id, exc)
                self._set_run_state(project_id, {"status": "failed", "scope": "reflect", "shot_id": shot_id})
                bundle = self.store.get_project(project_id)
                self.store.update_project_json(project_id, {"active_shot_id": None, "status": _project_status_from_prefix(bundle)})
                raise InvalidProjectError(str(exc)) from exc
            self._set_run_state(project_id, {"status": "idle"})
            bundle = self.store.get_project(project_id)
            self.store.update_project_json(project_id, {"active_shot_id": None, "status": _project_status_from_prefix(bundle)})
            return self.store.get_project(project_id)
        finally:
            self.lock.release(project_id)
            self._clear_cancel_event(project_id)

    def rerun_assembly(self, project_id: str) -> Dict[str, Any]:
        self._reset_cancel_event(project_id)
        self.lock.acquire(project_id)
        try:
            self._set_run_state(project_id, {"status": "assembling", "scope": "rerun_assembly"})
            self.store.update_project_json(project_id, {"status": "running", "active_shot_id": None})
            try:
                self._rerun_assembly_without_lock(project_id)
            except Exception as exc:
                self._set_run_state(project_id, {"status": "failed", "scope": "rerun_assembly"})
                bundle = self.store.get_project(project_id)
                self.store.update_project_json(project_id, {"active_shot_id": None, "status": _project_status_from_prefix(bundle)})
                raise InvalidProjectError(str(exc)) from exc
            self._set_run_state(project_id, {"status": "idle"})
            bundle = self.store.get_project(project_id)
            self.store.update_project_json(project_id, {"active_shot_id": None, "status": _project_status_from_prefix(bundle)})
            return self.store.get_project(project_id)
        finally:
            self.lock.release(project_id)
            self._clear_cancel_event(project_id)

    def interrupt_project(self, project_id: str) -> Dict[str, Any]:
        self._cancel_event(project_id).set()
        bundle = self.store.get_project(project_id)
        for shot in bundle["shots"]:
            if _is_running_status(shot.get("state", {}).get("status")):
                shot["state"]["status"] = "interrupted"
                shot["state"]["updated_at"] = _now()
                self.store.write_shot(project_id, shot)
        self.store.update_project_json(
            project_id,
            {"status": "interrupted", "active_shot_id": None, "run_state": {"status": "stopped"}},
        )
        return self.store.get_project(project_id)

    def _reset_cancel_event(self, project_id: str) -> threading.Event:
        event = self._cancel_event(project_id)
        event.clear()
        return event

    def _cancel_event(self, project_id: str) -> threading.Event:
        with self._cancel_lock:
            return self._cancel_events.setdefault(project_id, threading.Event())

    def _clear_cancel_event(self, project_id: str) -> None:
        with self._cancel_lock:
            self._cancel_events.pop(project_id, None)

    def _set_run_state(self, project_id: str, state: Dict[str, Any]) -> None:
        payload = {"updated_at": _now(), **state}
        try:
            self.store.update_project_json(project_id, {"run_state": payload})
        except Exception:
            pass

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
        elif _reference_selection_storymem(context.bundle):
            result = _select_storymem_memory_references(context, self.store)
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
        if status == "completed":
            self._maybe_submit_eval(project_id)

    def _maybe_submit_eval(self, project_id: str) -> None:
        bundle = self.store.get_project(project_id)
        if not _auto_submit_eval_enabled(bundle):
            return
        project_dir = self.store.project_dir(project_id)
        log_path = project_dir / "eval_submit.log"
        try:
            log_file = log_path.open("ab")
            subprocess.Popen(
                [
                    sys.executable,
                    str(EVAL_SUBMIT_SCRIPT),
                    "--project-dir",
                    str(project_dir),
                ],
                cwd=str(EVAL_SUBMIT_SCRIPT.parents[1]),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
        except Exception as exc:
            write_json_atomic(
                project_dir / "eval_submitted.json",
                {
                    "submitted": False,
                    "state": "failed",
                    "failed_at": _now(),
                    "project_dir": str(project_dir),
                    "error": repr(exc),
                    "note": "Failed to start the local EntityBench eval submitter subprocess.",
                },
            )
        finally:
            try:
                log_file.close()
            except Exception:
                pass

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
            from storymem_web.smooth_transition import SmoothVideoAssembler

            SmoothVideoAssembler()(clips, str(final_path))
            log["mode"] = "smooth_rife"
        elif len(clips) == 1:
            _concat_videos_ffmpeg([Path(clips[0]["output_video"])], final_path)
            log["mode"] = "single_clip_copy"
        else:
            from storymem_web.smooth_transition import SmoothVideoAssembler

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


class RealExecutionBackend:
    name = "real"

    def __init__(
        self,
        store: ProjectStore,
        *,
        client: SeedanceLikeClient | None = None,
        visual_planner: VisualMemoryPlanner | None = None,
        postprocessor: VideoPostprocessor | None = None,
        dry_run: bool = True,
    ) -> None:
        self.store = store
        self.client = client
        self.visual_planner = visual_planner or PlaceholderVisualMemoryPlanner()
        self.postprocessor = postprocessor or PlaceholderVideoPostprocessor(store)
        self.dry_run = dry_run
        self._legacy_plan_cache: Dict[tuple[str, str, str], VisualMemoryPlanResult] = {}

    def execute(self, context: ShotExecutionContext) -> ShotExecutionResult:
        prepared = self._prepare_request(context)
        if self.dry_run:
            return self._execute_dry_run(context, prepared)
        return self._execute_submit(context, prepared)

    def plan_visual_elements(self, context: VisualMemoryPlanContext) -> VisualMemoryPlanResult:
        planner = self.visual_planner
        if hasattr(planner, "plan_visual_elements"):
            return planner.plan_visual_elements(context)  # type: ignore[attr-defined]
        result = planner.plan(context)
        self._legacy_plan_cache[(context.project_id, context.shot["shot_id"], context.attempt_id)] = result
        return VisualMemoryPlanResult(
            visual_element_status=result.visual_element_status,
            details=result.details,
            logs=[log for log in result.logs if log.get("step") != "selecting_visual_references"],
        )

    def select_historical_references(self, context: ReferenceSelectionContext) -> ReferenceSelectionResult:
        planner = self.visual_planner
        if hasattr(planner, "select_references"):
            return planner.select_references(context)  # type: ignore[attr-defined]
        result = self._legacy_plan_cache.pop(
            (context.project_id, context.shot["shot_id"], context.attempt_id),
            None,
        )
        if result is None:
            result = planner.plan(
                VisualMemoryPlanContext(
                    project_id=context.project_id,
                    bundle=context.bundle,
                    shot=context.shot,
                    previous_shots=context.previous_shots,
                    attempt_id=context.attempt_id,
                    attempt_dir=context.attempt_dir,
                    input_snapshot=context.input_snapshot,
                )
            )
        return ReferenceSelectionResult(
            selected_references=result.selected_references,
            prompt_context=result.prompt_context,
            details=result.details,
            logs=[log for log in result.logs if log.get("step") == "selecting_visual_references"],
        )

    def compose_seedance_prompt(self, context: SeedanceGenerationContext) -> SeedancePromptCompositionResult:
        submitted_prompt = self._compose_submitted_prompt(context, force_context_prompt=True)
        shot_context = ShotExecutionContext(
            project_id=context.project_id,
            bundle=context.bundle,
            shot=context.shot,
            previous_shots=context.previous_shots,
            attempt_id=context.attempt_id,
            attempt_dir=context.attempt_dir,
            input_snapshot=context.input_snapshot,
            started_at=_now(),
            should_cancel=context.should_cancel,
        )
        return SeedancePromptCompositionResult(
            submitted_prompt=submitted_prompt,
            request_content_summary=self._request_content_summary(
                shot_context,
                submitted_prompt,
                _combined_static_references(context, context.selected_references),
            ),
            details={
                "generation_mode": context.input_snapshot.get("generation_mode", "default"),
                "reference_count": len(context.selected_references),
                "predefined_reference_count": len(_predefined_references_from_input(context.input_snapshot)),
            },
            logs=[_log("composing_prompt", "Seedance prompt composed locally")],
        )

    def generate_seedance_video(self, context: SeedanceGenerationContext) -> SeedanceGenerationResult:
        settings = context.bundle["settings"]
        submitted_prompt = self._compose_submitted_prompt(context)
        seedance_options = {
            "model": settings.get("seedance", {}).get("model"),
            "duration": context.input_snapshot.get("duration_seconds"),
            "ratio": settings.get("seedance", {}).get("ratio"),
            "resolution": settings.get("seedance", {}).get("resolution"),
            "generate_audio": settings.get("generation", {}).get("audio", True),
            "watermark": False,
            "return_last_frame": True,
            "execution_expires_after": 86400,
            "callback_url": None,
        }
        shot_context = ShotExecutionContext(
            project_id=context.project_id,
            bundle=context.bundle,
            shot=context.shot,
            previous_shots=context.previous_shots,
            attempt_id=context.attempt_id,
            attempt_dir=context.attempt_dir,
            input_snapshot=context.input_snapshot,
            started_at=_now(),
            should_cancel=context.should_cancel,
        )
        request = {
            "runner": self.name,
            "dry_run": self.dry_run,
            "seedance": seedance_options,
            "content": self._request_content_summary(
                shot_context,
                submitted_prompt,
                _combined_static_references(context, context.selected_references),
            ),
        }
        if self.dry_run:
            video_asset_id = f"vid_{context.shot['shot_id']}_{context.attempt_id}_dryrun"
            video_path = self._write_video_asset(
                shot_context,
                video_asset_id,
                video_bytes=b"dry-run video placeholder\n",
            )
            assets = self._video_asset_record(
                shot_context,
                video_asset_id,
                video_path,
                source_prefix="dry_run",
                dry_run=True,
                duration_seconds=context.input_snapshot.get("duration_seconds"),
            )
            task_id = f"dry-run-{context.project_id}-{context.shot['shot_id']}-{context.attempt_id}"
            return SeedanceGenerationResult(
                submitted_prompt=submitted_prompt,
                request=request,
                response={"runner": self.name, "dry_run": True, "task_id": task_id, "status": "completed"},
                raw_video_asset_id=video_asset_id,
                assets=assets,
                task_id=task_id,
                usage={"estimated_cny": 0.0, "dry_run": True},
                logs=[
                    _log("submitting", "Seedance generation used composed prompt from seedance_prompt step"),
                    _log("submitting", "dry-run mode skipped Seedance task creation"),
                ],
            )

        options = dict(seedance_options)
        client = self.client or SeedanceClient(model=str(options.get("model") or DEFAULT_SEEDANCE_MODEL))
        content = self._seedance_content_for_submit(
            shot_context,
            client,
            submitted_prompt,
            context.selected_references,
        )
        request["dry_run"] = False
        request["content"] = _content_summary_for_storage(content)
        request["media_debug"] = _media_debug_for_submission(content)
        self._persist_seedance_submission_request(context.attempt_dir, request)
        created, content, sensitive_filter_logs = self._create_seedance_task_with_image_safety_filter(
            client,
            content,
            options,
            context.attempt_dir,
            request,
        )
        task_id = extract_task_id(created)
        request["create_task_response"] = created
        request["seedance_task_id"] = task_id
        self._persist_seedance_submission_request(context.attempt_dir, request)
        completed = client.wait_task(
            task_id,
            poll_interval=10,
            max_wait_seconds=1800,
            should_cancel=context.should_cancel,
        )
        video_url = extract_video_url(completed)
        video_asset_id = f"vid_{context.shot['shot_id']}_{context.attempt_id}"
        project_dir = self.store.project_dir(context.project_id)
        video_path = project_dir / "assets" / "videos" / f"{video_asset_id}.mp4"
        client.download_video(video_url, str(video_path))
        assets = self._video_asset_record(
            shot_context,
            video_asset_id,
            video_path,
            source_prefix="seedance",
            dry_run=False,
            duration_seconds=int(options["duration"]),
        )
        usage = completed.get("usage") if isinstance(completed, dict) else {}
        request["completed_task_response"] = completed
        request["output_video_url"] = video_url
        return SeedanceGenerationResult(
            submitted_prompt=submitted_prompt,
            request=request,
            response={"create_task": created, "completed_task": completed, "video_url": video_url},
            raw_video_asset_id=video_asset_id,
            assets=assets,
            task_id=task_id,
            usage=usage if isinstance(usage, dict) else {},
            logs=[
                _log("submitting", "Seedance generation used composed prompt from seedance_prompt step"),
                *sensitive_filter_logs,
                _log("submitting", f"Seedance task created: {task_id}"),
                _log("waiting_for_seedance", f"Seedance task completed: {task_id}"),
                _log("downloading", "Seedance video downloaded into project bundle"),
            ],
        )

    def _compose_submitted_prompt(
        self,
        context: SeedanceGenerationContext,
        *,
        force_context_prompt: bool = False,
    ) -> str:
        if context.prompt_context is not None:
            if force_context_prompt:
                return _maybe_force_animation_prompt(context.prompt_context, context.input_snapshot, context.bundle)
            return context.prompt_context
        shot_spec = _shot_spec_from_bundle_shot(context.shot)
        generation_mode = context.input_snapshot.get("generation_mode", "default")
        reference_items = [
            {"type": "image_asset", "asset_id": item.get("asset_id"), "metadata": item}
            for item in _combined_static_references(context, context.selected_references)
            if item.get("asset_id") or item.get("source_path")
        ]
        if _uses_default_last_frame_continuity(context.shot, context.input_snapshot):
            previous = context.previous_shots[-1]
            reference_items.append(
                {
                    "type": "image_asset",
                    "metadata": {
                        "label": "previous_last_frame",
                        "roles": ["previous_last_frame"],
                        "source_type": "auto",
                        "source_shot_id": previous.get("shot_id"),
                        "source_scene_num": previous.get("scene_num"),
                        "source_shot_num": previous.get("shot_num"),
                        "source_prompt": previous.get("inputs", {}).get("video_prompt", ""),
                        "file": "seedance_last_frame_url",
                    },
                }
            )
        return _maybe_force_animation_prompt(
            compose_prompt(
                prompt=shot_spec.prompt,
                is_first=not context.previous_shots,
                is_cut=shot_spec.is_cut,
                refs=reference_items,
                story_script=_story_script(context.bundle),
                scene_num=shot_spec.scene_num,
                shot_num=shot_spec.shot_num,
                enhanced_text_prompt=True,
                generation_mode=str(generation_mode),
            ),
            context.input_snapshot,
            context.bundle,
        )

    def maintain_keyframes(self, context: VideoPostprocessContext) -> VideoPostprocessResult:
        return self.postprocessor.process(context)

    def _prepare_request(self, context: ShotExecutionContext) -> Dict[str, Any]:
        settings = context.bundle["settings"]
        plan = self.visual_planner.plan(
            VisualMemoryPlanContext(
                project_id=context.project_id,
                bundle=context.bundle,
                shot=context.shot,
                previous_shots=context.previous_shots,
                attempt_id=context.attempt_id,
                attempt_dir=context.attempt_dir,
                input_snapshot=context.input_snapshot,
            )
        )
        visual_status = plan.visual_element_status
        references = plan.selected_references
        reference_items = [
            {"type": "image_asset", "asset_id": item.get("asset_id"), "metadata": item}
            for item in _combined_static_references(context, references)
            if item.get("asset_id") or item.get("source_path")
        ]
        shot_spec = _shot_spec_from_bundle_shot(context.shot)
        story_script = _story_script(context.bundle)
        generation_mode = context.input_snapshot.get("generation_mode", "default")
        submitted_prompt = _maybe_force_animation_prompt(
            plan.prompt_context
            if plan.prompt_context is not None
            else compose_prompt(
                prompt=shot_spec.prompt,
                is_first=not context.previous_shots,
                is_cut=shot_spec.is_cut,
                refs=reference_items,
                story_script=story_script,
                scene_num=shot_spec.scene_num,
                shot_num=shot_spec.shot_num,
                enhanced_text_prompt=True,
                generation_mode=generation_mode,
            ),
            context.input_snapshot,
            context.bundle,
        )
        seedance_options = {
            "model": settings.get("seedance", {}).get("model"),
            "duration": context.input_snapshot.get("duration_seconds"),
            "ratio": settings.get("seedance", {}).get("ratio"),
            "resolution": settings.get("seedance", {}).get("resolution"),
            "generate_audio": settings.get("generation", {}).get("audio", True),
            "watermark": False,
            "return_last_frame": True,
            "execution_expires_after": 86400,
            "callback_url": None,
        }
        request = {
            "runner": self.name,
            "dry_run": self.dry_run,
            "seedance": seedance_options,
            "content": self._request_content_summary(
                context,
                submitted_prompt,
                _combined_static_references(context, references),
            ),
        }
        write_json_atomic(context.attempt_dir / "visual_element_status.json", visual_status)
        write_json_atomic(context.attempt_dir / "selected_references.json", references)
        write_json_atomic(context.attempt_dir / "request.json", request)
        if plan.details:
            write_json_atomic(context.attempt_dir / "visual_element_details.json", plan.details)
        return {
            "settings": settings,
            "visual_status": visual_status,
            "references": references,
            "reference_items": reference_items,
            "submitted_prompt": submitted_prompt,
            "request": request,
            "seedance_options": seedance_options,
            "plan_logs": plan.logs,
            "plan_details": plan.details,
        }

    def _request_content_summary(
        self,
        context: ShotExecutionContext,
        submitted_prompt: str,
        references: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": submitted_prompt}]
        mode = str(context.input_snapshot.get("generation_mode") or "default")
        if _uses_smooth_reference_video(context.shot, context.input_snapshot):
            content.append(
                {
                    "type": "video_asset_pending_publication",
                    "role": "reference_video",
                    "metadata": self._previous_tail_metadata(
                        context,
                        source_path=None,
                        duration_seconds=_smooth_reference_seconds(context),
                    ),
                }
            )
        elif mode == "last_frame_only":
            content.append(
                {
                    "type": "seedance_last_frame_url",
                    "role": "first_frame",
                    "metadata": self._previous_last_frame_metadata(context),
                }
            )
            return content
        content.extend(
            {"type": "image_asset", "asset_id": item.get("asset_id"), "metadata": item}
            for item in references
        )
        if _uses_default_last_frame_continuity(context.shot, context.input_snapshot):
            content.append(
                {
                    "type": "seedance_last_frame_url",
                    "role": "reference_image",
                    "metadata": self._previous_last_frame_metadata(context),
                }
            )
        return content

    def _execute_dry_run(
        self,
        context: ShotExecutionContext,
        prepared: Dict[str, Any],
    ) -> ShotExecutionResult:
        video_asset_id = f"vid_{context.shot['shot_id']}_{context.attempt_id}_dryrun"
        video_path = self._write_video_asset(
            context,
            video_asset_id,
            video_bytes=b"dry-run video placeholder\n",
        )
        video_assets = self._video_asset_record(
            context,
            video_asset_id,
            video_path,
            source_prefix="dry_run",
            dry_run=True,
            duration_seconds=context.input_snapshot.get("duration_seconds"),
        )
        postprocessed = self.postprocessor.process(
            VideoPostprocessContext(
                project_id=context.project_id,
                bundle=context.bundle,
                shot=context.shot,
                attempt_id=context.attempt_id,
                attempt_dir=context.attempt_dir,
                video_asset_id=video_asset_id,
                video_path=video_path,
                visual_element_status=prepared["visual_status"],
                dry_run=True,
            )
        )
        assets = {**video_assets, **postprocessed.assets}
        task_id = f"dry-run-{context.project_id}-{context.shot['shot_id']}-{context.attempt_id}"
        logs = [
            *prepared["plan_logs"],
            _log("composing_prompt", "Seedance prompt composed locally"),
            _log("submitting", "dry-run mode skipped Seedance task creation"),
            *postprocessed.logs,
        ]
        return ShotExecutionResult(
            visual_element_status=prepared["visual_status"],
            selected_references=prepared["references"],
            produced_visual_memory=postprocessed.produced_visual_memory,
            submitted_prompt=prepared["submitted_prompt"],
            request=prepared["request"],
            response={"runner": self.name, "dry_run": True, "task_id": task_id, "status": "completed"},
            raw_video_asset_id=video_asset_id,
            assets=assets,
            task_id=task_id,
            usage={"estimated_cny": 0.0, "dry_run": True},
            postprocess={"dry_run": True, **postprocessed.postprocess},
            logs=logs,
        )

    def _execute_submit(
        self,
        context: ShotExecutionContext,
        prepared: Dict[str, Any],
    ) -> ShotExecutionResult:
        options = dict(prepared["seedance_options"])
        client = self.client or SeedanceClient(model=str(options.get("model") or DEFAULT_SEEDANCE_MODEL))
        content = self._seedance_content_for_submit(
            context,
            client,
            prepared["submitted_prompt"],
            prepared["references"],
        )
        request = dict(prepared["request"])
        request["dry_run"] = False
        request["content"] = _content_summary_for_storage(content)
        request["media_debug"] = _media_debug_for_submission(content)
        self._persist_seedance_submission_request(context.attempt_dir, request)
        created, content, sensitive_filter_logs = self._create_seedance_task_with_image_safety_filter(
            client,
            content,
            options,
            context.attempt_dir,
            request,
        )
        task_id = extract_task_id(created)
        request["create_task_response"] = created
        request["seedance_task_id"] = task_id
        self._persist_seedance_submission_request(context.attempt_dir, request)
        completed = client.wait_task(
            task_id,
            poll_interval=10,
            max_wait_seconds=1800,
            should_cancel=context.should_cancel,
        )
        video_url = extract_video_url(completed)

        video_asset_id = f"vid_{context.shot['shot_id']}_{context.attempt_id}"
        project_dir = self.store.project_dir(context.project_id)
        video_path = project_dir / "assets" / "videos" / f"{video_asset_id}.mp4"
        client.download_video(video_url, str(video_path))
        video_assets = self._video_asset_record(
            context,
            video_asset_id,
            video_path,
            source_prefix="seedance",
            dry_run=False,
            duration_seconds=int(options["duration"]),
        )
        postprocessed = self.postprocessor.process(
            VideoPostprocessContext(
                project_id=context.project_id,
                bundle=context.bundle,
                shot=context.shot,
                attempt_id=context.attempt_id,
                attempt_dir=context.attempt_dir,
                video_asset_id=video_asset_id,
                video_path=video_path,
                visual_element_status=prepared["visual_status"],
                dry_run=False,
            )
        )
        assets = {**video_assets, **postprocessed.assets}
        request["completed_task_response"] = completed
        request["output_video_url"] = video_url
        logs = [
            *prepared["plan_logs"],
            _log("composing_prompt", "Seedance prompt composed locally"),
            _log("submitting", f"Seedance task created: {task_id}"),
            *sensitive_filter_logs,
            _log("waiting_for_seedance", f"Seedance task completed: {task_id}"),
            _log("downloading", "Seedance video downloaded into project bundle"),
            *postprocessed.logs,
        ]
        usage = completed.get("usage") if isinstance(completed, dict) else {}
        return ShotExecutionResult(
            visual_element_status=prepared["visual_status"],
            selected_references=prepared["references"],
            produced_visual_memory=postprocessed.produced_visual_memory,
            submitted_prompt=prepared["submitted_prompt"],
            request=request,
            response={"create_task": created, "completed_task": completed, "video_url": video_url},
            raw_video_asset_id=video_asset_id,
            assets=assets,
            task_id=task_id,
            usage=usage if isinstance(usage, dict) else {},
            postprocess=postprocessed.postprocess,
            logs=logs,
        )

    def _persist_seedance_submission_request(self, attempt_dir: Path, request: Dict[str, Any]) -> None:
        generation_dir = attempt_dir / "seedance_generation"
        generation_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(generation_dir / "request.json", request)
        write_json_atomic(attempt_dir / "request.json", request)
        media_debug = request.get("media_debug")
        if isinstance(media_debug, dict):
            write_json_atomic(generation_dir / "media_debug.json", media_debug)

    def _create_seedance_task_with_image_safety_filter(
        self,
        client: SeedanceLikeClient,
        content: List[Dict[str, Any]],
        options: Dict[str, Any],
        attempt_dir: Path,
        request: Dict[str, Any],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        filtered_content = list(content)
        dropped_records: List[Dict[str, Any]] = []
        logs: List[Dict[str, Any]] = []
        while True:
            try:
                created = client.create_task(
                    content=filtered_content,
                    duration=int(options["duration"]),
                    ratio=str(options["ratio"]),
                    resolution=str(options["resolution"]),
                    generate_audio=bool(options["generate_audio"]),
                    watermark=bool(options["watermark"]),
                    return_last_frame=bool(options["return_last_frame"]),
                    execution_expires_after=int(options["execution_expires_after"]),
                    callback_url=options.get("callback_url"),
                )
                request["content"] = _content_summary_for_storage(filtered_content)
                request["media_debug"] = _media_debug_for_submission(filtered_content)
                if dropped_records:
                    request["sensitive_content_filter"] = {
                        "policy": "drop_input_image_sensitive_content_for_this_submission_only",
                        "dropped": dropped_records,
                        "remaining_content_count": len(filtered_content),
                    }
                    self._persist_seedance_submission_request(attempt_dir, request)
                return created, filtered_content, logs
            except SeedanceError as exc:
                content_index = _seedance_rejected_content_index(exc)
                if content_index is None:
                    raise
                if content_index < 0 or content_index >= len(filtered_content):
                    raise
                rejected_item = filtered_content[content_index]
                if rejected_item.get("type") != "image_url":
                    raise
                dropped = _sensitive_image_drop_record(content_index, rejected_item, exc)
                dropped_records.append(dropped)
                logs.append(
                    _log(
                        "submitting",
                        (
                            "Seedance rejected image content"
                            f"[{content_index}] with InputImageSensitiveContentDetected; "
                            "dropped it for this submission only and retried."
                        ),
                    )
                )
                del filtered_content[content_index]
                request["content"] = _content_summary_for_storage(filtered_content)
                request["media_debug"] = _media_debug_for_submission(filtered_content)
                request["sensitive_content_filter"] = {
                    "policy": "drop_input_image_sensitive_content_for_this_submission_only",
                    "dropped": dropped_records,
                    "remaining_content_count": len(filtered_content),
                }
                self._persist_seedance_submission_request(attempt_dir, request)

    def _seedance_content_for_submit(
        self,
        context: ShotExecutionContext,
        client: SeedanceLikeClient,
        submitted_prompt: str,
        references: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        mode = str(context.input_snapshot.get("generation_mode") or "default")
        content: List[Dict[str, Any]] = [{"type": "text", "text": submitted_prompt}]
        static_references = _combined_static_references(context, references)
        if _uses_smooth_reference_video(context.shot, context.input_snapshot):
            content.append(self._smooth_reference_video_item(context))
        elif mode == "last_frame_only":
            content.append(self._previous_last_frame_item(context, client, first_frame=True))
            return content
        content.extend(
            _seedance_content(
                context.project_id,
                self.store,
                "",
                static_references,
                include_text=False,
            )
        )
        if _uses_default_last_frame_continuity(context.shot, context.input_snapshot):
            content.append(self._previous_last_frame_item(context, client, first_frame=False))
        return content

    def _previous_last_frame_item(
        self,
        context: ShotExecutionContext,
        client: SeedanceLikeClient,
        *,
        first_frame: bool,
    ) -> Dict[str, Any]:
        previous = _previous_completed_shot(context)
        previous_attempt = previous.get("attempt") or {}
        task_id = previous_attempt.get("seedance", {}).get("task_id")
        if not task_id:
            raise BackendExecutionError("Previous shot has no Seedance task id for last_frame_url")
        response = client.get_task(str(task_id))
        metadata = self._previous_last_frame_metadata(context)
        metadata["input_origin"] = "seedance_last_frame_url"
        item = {
            "type": "image_url",
            "image_url": {"url": extract_last_frame_url(response)},
            "role": "first_frame" if first_frame else "reference_image",
            "metadata": metadata,
        }
        return item

    def _smooth_reference_video_item(self, context: ShotExecutionContext) -> Dict[str, Any]:
        previous = _previous_completed_shot(context)
        previous_attempt = previous.get("attempt") or {}
        raw_video_asset_id = previous_attempt.get("outputs", {}).get("raw_video_asset_id")
        if not raw_video_asset_id:
            raise BackendExecutionError("Smooth mode requires the previous raw output video")
        raw_video_path = self.store.asset_path(context.project_id, str(raw_video_asset_id))
        tail_path = context.attempt_dir / "smooth_reference_tail.mp4"
        from storymem_web.reference_video import (
            configured_reference_video_publisher,
            extract_video_tail,
            require_public_https_url,
        )

        duration_seconds = _smooth_reference_seconds(context)
        extract_video_tail(raw_video_path, tail_path, seconds=duration_seconds)
        public_url = require_public_https_url(configured_reference_video_publisher().publish(tail_path))
        return video_item_with_metadata(
            public_url,
            self._previous_tail_metadata(
                context,
                source_path=tail_path,
                duration_seconds=duration_seconds,
            ),
        )

    def _previous_last_frame_metadata(self, context: ShotExecutionContext) -> Dict[str, Any]:
        previous = _previous_completed_shot(context)
        return {
            "label": "previous_last_frame",
            "roles": ["previous_last_frame"],
            "source_type": "auto",
            "source_shot_id": previous["shot_id"],
            "source_scene_num": previous.get("scene_num"),
            "source_shot_num": previous.get("shot_num"),
            "source_prompt": previous.get("inputs", {}).get("video_prompt", ""),
        }

    def _previous_tail_metadata(
        self,
        context: ShotExecutionContext,
        *,
        source_path: Path | None,
        duration_seconds: float,
    ) -> Dict[str, Any]:
        previous = _previous_completed_shot(context)
        metadata = {
            "label": "previous_tail_video",
            "roles": ["previous_tail_video"],
            "source_type": "auto",
            "source_shot_id": previous["shot_id"],
            "source_attempt_id": (previous.get("attempt") or {}).get("attempt_id"),
            "source_scene_num": previous.get("scene_num"),
            "source_shot_num": previous.get("shot_num"),
            "source_prompt": previous.get("inputs", {}).get("video_prompt", ""),
            "input_origin": "published_previous_raw_tail",
            "duration_seconds": duration_seconds,
        }
        if source_path is not None:
            metadata["source_path"] = str(source_path)
            metadata["file"] = source_path.name
        return metadata

    def _write_video_asset(
        self,
        context: ShotExecutionContext,
        video_asset_id: str,
        *,
        video_bytes: bytes,
    ) -> Path:
        project_dir = self.store.project_dir(context.project_id)
        video_path = project_dir / "assets" / "videos" / f"{video_asset_id}.mp4"
        video_path.write_bytes(video_bytes)
        return video_path

    def _video_asset_record(
        self,
        context: ShotExecutionContext,
        video_asset_id: str,
        video_path: Path,
        *,
        source_prefix: str,
        dry_run: bool,
        duration_seconds: Any,
    ) -> Dict[str, Dict[str, Any]]:
        project_dir = self.store.project_dir(context.project_id)
        return {
            video_asset_id: {
                "kind": "video",
                "path": _relative(project_dir, video_path),
                "created_at": _now(),
                "source": {
                    "type": f"{source_prefix}_seedance_output",
                    "shot_id": context.shot["shot_id"],
                    "attempt_id": context.attempt_id,
                },
                "metadata": {
                    "duration_seconds": duration_seconds,
                    "has_audio": False,
                    "dry_run": dry_run,
                },
            }
        }


class PlaceholderVisualMemoryPlanner:
    name = "placeholder_visual_memory"

    def plan(self, context: VisualMemoryPlanContext) -> VisualMemoryPlanResult:
        plan = self.plan_visual_elements(context)
        selection = self.select_references(
            ReferenceSelectionContext(
                project_id=context.project_id,
                bundle=context.bundle,
                shot=context.shot,
                previous_shots=context.previous_shots,
                attempt_id=context.attempt_id,
                attempt_dir=context.attempt_dir,
                input_snapshot=context.input_snapshot,
                visual_element_status=plan.visual_element_status,
                visual_element_details=plan.details,
            )
        )
        return VisualMemoryPlanResult(
            visual_element_status=plan.visual_element_status,
            selected_references=selection.selected_references,
            prompt_context=selection.prompt_context,
            logs=[*plan.logs, *selection.logs],
            details={**plan.details, **selection.details},
        )

    def plan_visual_elements(self, context: VisualMemoryPlanContext) -> VisualMemoryPlanResult:
        visual_status = _dry_run_visual_status(context.shot)
        return VisualMemoryPlanResult(
            visual_element_status=visual_status,
            logs=[
                _log("planning_visual_elements", "dry-run visual element plan generated without LLM call"),
            ],
            details={"dry_run": True},
        )

    def select_references(self, context: ReferenceSelectionContext) -> ReferenceSelectionResult:
        references = _dry_run_references(context.previous_shots, context.bundle["settings"])
        return ReferenceSelectionResult(
            selected_references=references,
            prompt_context=None,
            logs=[_log("selecting_visual_references", f"{len(references)} dry-run references selected from current lineage")],
            details={"dry_run": True},
        )


class NotebookVisualElementMemoryAdapter:
    name = "visual_element_memory"

    def __init__(self, store: ProjectStore) -> None:
        self.store = store
        self._planned: Dict[tuple[str, str, str], Any] = {}

    def plan(self, context: VisualMemoryPlanContext) -> VisualMemoryPlanResult:
        plan = self.plan_visual_elements(context)
        selection = self.select_references(
            ReferenceSelectionContext(
                project_id=context.project_id,
                bundle=context.bundle,
                shot=context.shot,
                previous_shots=context.previous_shots,
                attempt_id=context.attempt_id,
                attempt_dir=context.attempt_dir,
                input_snapshot=context.input_snapshot,
                visual_element_status=plan.visual_element_status,
                visual_element_details=plan.details,
            )
        )
        return VisualMemoryPlanResult(
            visual_element_status=plan.visual_element_status,
            selected_references=selection.selected_references,
            prompt_context=selection.prompt_context,
            details={**plan.details, **selection.details},
            logs=[*plan.logs, *selection.logs],
        )

    def plan_visual_elements(self, context: VisualMemoryPlanContext) -> VisualMemoryPlanResult:
        memory = self._memory_from_bundle(context.project_id, context.bundle)
        shot_spec = _shot_spec_from_bundle_shot(context.shot)
        plan = memory.plan_visual_elements_for_shot(int(context.shot["order_index"]), shot_spec)
        report = plan.report_record
        record = deepcopy(memory.records[-1]) if memory.records else {}
        visual_status = _visual_status_rows(report)
        self._planned[(context.project_id, context.shot["shot_id"], context.attempt_id)] = memory
        return VisualMemoryPlanResult(
            visual_element_status=visual_status,
            details={"visual_element_record": record},
            logs=[
                _log(
                    "planning_visual_elements",
                    f"visual element planner produced {len(visual_status)} element rows",
                ),
            ],
        )

    def select_references(self, context: ReferenceSelectionContext) -> ReferenceSelectionResult:
        key = (context.project_id, context.shot["shot_id"], context.attempt_id)
        memory = self._planned.get(key) or self._memory_for_reference_selection(context)
        budget = _effective_reference_budget(context)
        memory.config.visual_element_sink_frame_count = budget["effective_sink_frame_count"]
        memory.config.visual_element_max_retrieved_frames = budget["effective_max_retrieved_frames"]
        shot_spec = _shot_spec_from_bundle_shot(context.shot)
        plan_result = _plan_result_from_details(context.visual_element_details)
        selection = memory.select_historical_references_for_shot(
            int(context.shot["order_index"]),
            shot_spec,
            plan_result,
        )
        report = selection.report_record
        record = deepcopy(memory.records[-1]) if memory.records else {}
        self._planned[key] = memory
        selected_references = _visual_reference_rows(
            store=self.store,
            project_id=context.project_id,
            shots=context.bundle["shots"],
            references=report.get("visual_element_selected_references") or [],
        )
        selected_references, guidance_details, guidance_logs = _add_reference_guidance(context, selected_references)
        prompt_context = _visual_reference_prompt_context(
            context.bundle,
            context.shot,
            context.input_snapshot,
            context.visual_element_status,
            selected_references,
        )
        return ReferenceSelectionResult(
            selected_references=selected_references,
            prompt_context=_prompt_for_generation_mode(
                prompt_context,
                str(context.input_snapshot.get("generation_mode") or "default"),
                default_last_frame_continuity=_uses_default_last_frame_continuity(context.shot, context.input_snapshot),
            ),
            details={
                "visual_element_record": record,
                "reference_guidance": guidance_details,
                "reference_budget": budget,
            },
            logs=[
                _log(
                    "selecting_visual_references",
                    f"visual element planner selected {len(selected_references)} references",
                ),
                _log(
                    "selecting_visual_references",
                    (
                        "static image budget: "
                        f"predefined={budget['predefined_reference_count']}, "
                        f"sink={budget['effective_sink_frame_count']}, "
                        f"retrieved_limit={budget['effective_max_retrieved_frames']}"
                    ),
                ),
                *guidance_logs,
            ],
        )

    def process(self, context: VideoPostprocessContext) -> VideoPostprocessResult:
        from storymem_seedance.executor import extract_shot_memory

        bundle = context.bundle
        settings = bundle["settings"]
        project_dir = self.store.project_dir(context.project_id)
        key = (context.project_id, context.shot["shot_id"], context.attempt_id)
        memory = self._planned.pop(key, None) or self._memory_for_current_postprocess(context)
        storymem_mode = _reference_selection_storymem(bundle)
        existing_memory_paths = (
            _storymem_existing_memory_paths(self.store, context)
            if storymem_mode
            else _existing_memory_paths(
                self.store,
                context.project_id,
                bundle["shots"],
                context.shot["shot_id"],
            )
        )
        profile = (
            STORYMEM_KEYFRAME_PROFILE
            if storymem_mode
            else str(settings.get("keyframes", {}).get("profile") or DEFAULT_KEYFRAME_PROFILE)
        )
        result = extract_shot_memory(
            str(context.video_path),
            existing_memory_paths=existing_memory_paths,
            keyframe_profile=profile,
        )
        assets: Dict[str, Dict[str, Any]] = {}
        keyframe_paths = []
        for index, source in enumerate(result.get("keyframe_paths") or [], start=1):
            source_path = Path(source)
            if not source_path.is_file():
                continue
            asset_id = f"img_{context.shot['shot_id']}_{context.attempt_id}_{index:03d}"
            target_path = project_dir / "assets" / "images" / f"{asset_id}{source_path.suffix or '.jpg'}"
            shutil.copyfile(source_path, target_path)
            keyframe_paths.append(str(target_path))
            assets[asset_id] = _image_asset_record(
                project_dir,
                target_path,
                asset_id,
                source_type="seedance_produced_keyframe",
                shot_id=context.shot["shot_id"],
                attempt_id=context.attempt_id,
                metadata={"postprocessor": self.name, "rank": index},
            )
        last_frame_path = result.get("last_frame_path")
        if last_frame_path and Path(str(last_frame_path)).is_file():
            asset_id = f"img_{context.shot['shot_id']}_{context.attempt_id}_last"
            source_path = Path(str(last_frame_path))
            target_path = project_dir / "assets" / "images" / f"{asset_id}{source_path.suffix or '.jpg'}"
            shutil.copyfile(source_path, target_path)
            assets[asset_id] = _image_asset_record(
                project_dir,
                target_path,
                asset_id,
                source_type="seedance_last_frame",
                shot_id=context.shot["shot_id"],
                attempt_id=context.attempt_id,
                metadata={"postprocessor": self.name, "active_in_memory_pool": False},
            )
        if _reference_selection_skips_visual_plan(bundle):
            produced = _naive_produced_memory_from_keyframes(
                keyframe_paths,
                context.shot["shot_id"],
                context.attempt_id,
            )
            details = read_json(context.attempt_dir / "visual_element_details.json", {})
            if not isinstance(details, dict):
                details = {}
            selection_mode = _reference_selection_mode(bundle)
            details[selection_mode] = {
                "vlm_annotation_skipped": True,
                "keyframe_count": len(produced),
                "keyframe_profile": profile,
            }
            write_json_atomic(context.attempt_dir / "visual_element_details.json", details)
            postprocess = {
                "postprocessor": self.name,
                "raw_keyframe_result": result,
                "visual_element_snapshot_path": "memory/visual_element_memory.json",
                "vlm_annotation_skipped": True,
                "keyframe_profile": profile,
                "history_compare_scope": (
                    "selected_storymem_memory_bank" if storymem_mode else "all_previous_memory"
                ),
                "history_compare_path_count": len(existing_memory_paths),
            }
            return VideoPostprocessResult(
                assets=assets,
                produced_visual_memory=produced,
                postprocess=postprocess,
                logs=[
                    _log("extracting_keyframes", f"extracted {len(keyframe_paths)} keyframes"),
                    _log(
                        "annotating_visual_memory",
                        f"VLM keyframe annotation skipped because Reference selection is {selection_mode}.",
                    ),
                ],
            )
        shot_spec = _shot_spec_from_bundle_shot(context.shot)
        annotations = memory.annotate_completed_shot(
            shot_index=int(context.shot["order_index"]) + 1,
            shot=shot_spec,
            keyframe_paths=keyframe_paths,
        )
        produced = _produced_memory_from_annotations(annotations, context.shot["shot_id"], context.attempt_id)
        details = read_json(context.attempt_dir / "visual_element_details.json", {})
        if not isinstance(details, dict):
            details = {}
        if memory.records:
            details["visual_element_record"] = deepcopy(memory.records[-1])
        write_json_atomic(context.attempt_dir / "visual_element_details.json", details)
        postprocess = {
            "postprocessor": self.name,
            "raw_keyframe_result": result,
            "visual_element_snapshot_path": "memory/visual_element_memory.json",
            "keyframe_profile": profile,
            "history_compare_scope": "all_previous_memory",
            "history_compare_path_count": len(existing_memory_paths),
        }
        return VideoPostprocessResult(
            assets=assets,
            produced_visual_memory=produced,
            postprocess=postprocess,
            logs=[
                _log("extracting_keyframes", f"extracted {len(keyframe_paths)} keyframes"),
                _log("annotating_visual_memory", f"annotated {len(annotations)} keyframes with visual elements"),
            ],
        )

    def _memory_from_bundle(self, project_id: str, bundle: Dict[str, Any]):
        from storymem_seedance.visual_element_memory import VisualElementMemory

        project_dir = self.store.project_dir(project_id)
        config = _run_config_for_bundle(bundle, project_dir / "memory")
        shots = [_shot_spec_from_bundle_shot(shot) for shot in bundle["shots"]]
        memory = VisualElementMemory(config, _story_script(bundle), shots)
        memory.load_snapshot(_visual_memory_snapshot_from_shots(bundle["shots"]))
        return memory

    def _memory_for_reference_selection(self, context: ReferenceSelectionContext):
        memory = self._memory_from_bundle(context.project_id, context.bundle)
        record = context.visual_element_details.get("visual_element_record") or {}
        registry_after = record.get("registry_after")
        if isinstance(registry_after, list):
            snapshot = _visual_memory_snapshot_from_shots(context.bundle["shots"])
            snapshot["registry"] = registry_after
            snapshot["records"] = [record]
            memory.load_snapshot(snapshot)
        return memory

    def _memory_for_current_postprocess(self, context: VideoPostprocessContext):
        memory = self._memory_from_bundle(context.project_id, context.bundle)
        details = read_json(context.attempt_dir / "visual_plan" / "details.json", {})
        record = details.get("visual_element_record") if isinstance(details, dict) else {}
        if isinstance(record, dict) and isinstance(record.get("registry_after"), list):
            snapshot = _visual_memory_snapshot_from_shots(context.bundle["shots"])
            snapshot["registry"] = record["registry_after"]
            snapshot["records"] = [record]
            memory.load_snapshot(snapshot)
        return memory


class PlaceholderVideoPostprocessor:
    name = "placeholder"

    def __init__(self, store: ProjectStore) -> None:
        self.store = store

    def process(self, context: VideoPostprocessContext) -> VideoPostprocessResult:
        suffix = "dryrun_001" if context.dry_run else "001"
        keyframe_asset_id = f"img_{context.shot['shot_id']}_{context.attempt_id}_{suffix}"
        project_dir = self.store.project_dir(context.project_id)
        image_path = project_dir / "assets" / "images" / f"{keyframe_asset_id}.jpg"
        image_path.write_bytes(
            b"dry-run image placeholder\n"
            if context.dry_run
            else b"pending keyframe extraction placeholder\n"
        )
        source_prefix = "dry_run" if context.dry_run else "seedance"
        produced = _placeholder_produced_memory(
            keyframe_asset_id,
            context.visual_element_status,
            "Dry-run placeholder for future VLM annotation."
            if context.dry_run
            else "Placeholder until real keyframe extraction and VLM annotation are connected.",
        )
        return VideoPostprocessResult(
            assets={
                keyframe_asset_id: {
                    "kind": "image",
                    "path": _relative(project_dir, image_path),
                    "created_at": _now(),
                    "source": {
                        "type": f"{source_prefix}_produced_keyframe",
                        "shot_id": context.shot["shot_id"],
                        "attempt_id": context.attempt_id,
                    },
                    "metadata": {"dry_run": context.dry_run, "postprocessor": self.name},
                }
            },
            produced_visual_memory=produced,
            postprocess={
                "postprocessor": self.name,
                "keyframe_extraction_pending": not context.dry_run,
            },
            logs=[
                _log(
                    "extracting_keyframes",
                    "placeholder keyframe asset registered"
                    if not context.dry_run
                    else "dry-run keyframe placeholder registered",
                )
            ],
        )


class RealKeyframePostprocessor:
    name = "keyframe_extractor"

    def __init__(self, store: ProjectStore) -> None:
        self.store = store

    def process(self, context: VideoPostprocessContext) -> VideoPostprocessResult:
        from storymem_seedance.executor import extract_shot_memory

        profile = str(context.shot.get("inputs", {}).get("keyframe_profile") or DEFAULT_KEYFRAME_PROFILE)
        result = extract_shot_memory(str(context.video_path), keyframe_profile=profile)
        keyframes = result.get("keyframe_paths") or []
        assets: Dict[str, Dict[str, Any]] = {}
        produced: List[Dict[str, Any]] = []
        project_dir = self.store.project_dir(context.project_id)
        for index, keyframe in enumerate(keyframes, start=1):
            source_path = Path(str(keyframe))
            if not source_path.exists():
                continue
            asset_id = f"img_{context.shot['shot_id']}_{context.attempt_id}_{index:03d}"
            target_path = project_dir / "assets" / "images" / f"{asset_id}{source_path.suffix or '.jpg'}"
            target_path.write_bytes(source_path.read_bytes())
            assets[asset_id] = {
                "kind": "image",
                "path": _relative(project_dir, target_path),
                "created_at": _now(),
                "source": {
                    "type": "seedance_produced_keyframe",
                    "shot_id": context.shot["shot_id"],
                    "attempt_id": context.attempt_id,
                },
                "metadata": {"dry_run": False, "postprocessor": self.name},
            }
            produced.append(
                {
                    "asset_id": asset_id,
                    "rank": index,
                    "visible_elements": [item["element_id"] for item in context.visual_element_status],
                    "annotation": "Keyframe extracted; VLM annotation pending.",
                    "active_in_memory_pool": True,
                }
            )
        return VideoPostprocessResult(
            assets=assets,
            produced_visual_memory=produced,
            postprocess={"postprocessor": self.name, "raw_result": result},
            logs=[_log("extracting_keyframes", f"extracted {len(produced)} keyframes")],
        )


class FakeNotebookRunner(NotebookRunner):
    """Compatibility wrapper that keeps the low-cost fake runner as default."""

    def __init__(self, store: ProjectStore) -> None:
        super().__init__(store, FakeExecutionBackend(store))


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


def _auto_submit_eval_enabled(bundle: Dict[str, Any]) -> bool:
    return bool(
        bundle.get("settings", {})
        .get("evaluation", {})
        .get("auto_submit_eval", False)
    )


def _project_status_from_prefix(bundle: Dict[str, Any]) -> str:
    completed_prefix = _completed_prefix(bundle.get("shots", []))
    shot_count = int((bundle.get("project") or {}).get("shot_count") or len(bundle.get("shots", [])))
    if completed_prefix >= shot_count and shot_count > 0:
        return "completed"
    if completed_prefix > 0:
        return "partial"
    return "draft"


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
    from .json_store import read_jsonl

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
    from storymem_seedance.visual_element_memory import (
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


def _dry_run_visual_status(shot: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "element_id": f"element-{shot['shot_id']}-001",
            "name": _short_name(shot["inputs"]["video_prompt"]),
            "type": "scene",
            "introduced_at": shot["shot_id"],
            "notes": "Dry-run placeholder; real LLM/VLM element maintenance is not invoked.",
            "status": "new",
            "reason": "Dry-run creates a deterministic element row to validate request construction.",
        }
    ]


def _dry_run_references(previous_shots: List[Dict[str, Any]], settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    limit = int(settings.get("visual_element_memory", {}).get("max_retrieved_frames", 4) or 0)
    if limit <= 0:
        return []
    references: List[Dict[str, Any]] = []
    for shot in reversed(previous_shots):
        if len(references) >= limit:
            break
        attempt = shot.get("attempt") or {}
        produced = attempt.get("produced_visual_memory") or []
        if not produced:
            continue
        memory = produced[0]
        references.append(
            {
                "reference_id": f"ref-{shot['shot_id']}-{attempt.get('attempt_id', 'unknown')}",
                "asset_id": memory.get("asset_id"),
                "source_shot_id": shot["shot_id"],
                "source_scene_num": shot.get("scene_num"),
                "source_shot_num": shot.get("shot_num"),
                "source_prompt": shot.get("inputs", {}).get("video_prompt", ""),
                "file": str(memory.get("asset_id") or ""),
                "roles": ["visual_element_memory"],
                "covered_elements": list(memory.get("visible_elements") or []),
                "conflict_elements": [],
                "score": 1.0,
                "reason": "Dry-run selected the previous produced memory placeholder.",
            }
        )
    references.reverse()
    return references


def _empty_visual_plan_result(selection_mode: str) -> VisualMemoryPlanResult:
    if selection_mode == NAIVE_TOP_K_SELECTION_MODE:
        reason = "Naive top-k ablation keeps the visual element collection empty."
    elif selection_mode == STORYMEM_MEMORY_SELECTION_MODE:
        reason = "StoryMem memory mode uses Sink+Recent memory without visual element planning."
    else:
        reason = f"Reference selection mode {selection_mode} does not require visual element planning."
    return VisualMemoryPlanResult(
        visual_element_status=[],
        details={
            "skipped": True,
            "selection_mode": selection_mode,
            "reason": reason,
        },
        logs=[
            _log(
                "planning_visual_elements",
                f"Visual Elements Plan skipped because Reference selection is {selection_mode}.",
            )
        ],
    )


def _select_naive_top_k_references(context: ReferenceSelectionContext, store: ProjectStore) -> ReferenceSelectionResult:
    budget = _naive_reference_budget(context)
    limit = int(budget["effective_max_retrieved_frames"])
    prompt = str(context.input_snapshot.get("video_prompt") or context.shot.get("inputs", {}).get("video_prompt") or "")
    score_details: Dict[str, Any] = {
        "method": "clip_text_image",
        "model": "ViT-B/32",
        "image_embedding_cache": "assets/embeddings",
        "fallback": None,
        "cache_hits": 0,
        "cache_writes": 0,
        "score_failures": [],
    }
    try:
        text_embedding = _clip_text_embedding(prompt)
    except Exception as exc:
        text_embedding = None
        score_details["fallback"] = f"text embedding failed; all scores set to 0: {type(exc).__name__}: {exc}"
    candidates = _naive_reference_candidates(context, store, text_embedding, score_details)
    candidates.sort(
        key=lambda item: (
            -float(item.get("score") or 0.0),
            -int(item.get("_source_order_index") or 0),
            int(item.get("rank") or 0),
            str(item.get("asset_id") or ""),
        )
    )
    selected_raw = candidates[:limit] if limit > 0 else []
    for item in selected_raw:
        item.pop("_source_order_index", None)
    selected = _visual_reference_rows(
        store=store,
        project_id=context.project_id,
        shots=context.bundle["shots"],
        references=selected_raw,
    )
    for index, item in enumerate(selected, start=1):
        item["reference_index"] = index
    prompt_context = _visual_reference_prompt_context(
        context.bundle,
        context.shot,
        context.input_snapshot,
        [],
        selected,
    )
    return ReferenceSelectionResult(
        selected_references=selected,
        prompt_context=_prompt_for_generation_mode(
            prompt_context,
            str(context.input_snapshot.get("generation_mode") or "default"),
            default_last_frame_continuity=_uses_default_last_frame_continuity(context.shot, context.input_snapshot),
        ),
        details={
            "selection_mode": NAIVE_TOP_K_SELECTION_MODE,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "score": score_details,
            "reference_budget": budget,
            "skipped_visual_element_planning": True,
        },
        logs=[
            _log(
                "selecting_visual_references",
                f"naive top-k CLIP-scored {len(candidates)} historical keyframes and selected {len(selected)} references",
            ),
            _log(
                "selecting_visual_references",
                (
                    "static image budget: "
                    f"predefined={budget['predefined_reference_count']}, "
                    "sink=0, "
                    f"retrieved_limit={budget['effective_max_retrieved_frames']}"
                ),
            ),
        ],
    )


def _select_storymem_memory_references(context: ReferenceSelectionContext, store: ProjectStore) -> ReferenceSelectionResult:
    budget = _storymem_reference_budget(context)
    limit = int(budget["effective_max_memory_size"])
    candidates = _storymem_memory_candidates(context, store)
    paths = [str(item["source_path"]) for item in candidates]
    memory_bank = _storymem_default_memory_bank(paths, limit, STORYMEM_MEMORY_FIX) if limit > 0 else []
    role_by_path = _storymem_default_memory_roles(memory_bank, limit, STORYMEM_MEMORY_FIX)
    candidate_by_path = {str(item["source_path"]): item for item in candidates}
    selected_raw: List[Dict[str, Any]] = []
    for path in memory_bank:
        item = dict(candidate_by_path.get(str(path)) or {})
        if not item:
            continue
        roles = role_by_path.get(str(path), ["default_memory"])
        item["roles"] = roles
        if "early_sink_memory" in roles:
            item["reference_intent"] = "StoryMem early sink memory selected as a stable long-range anchor."
        elif "recent_window_memory" in roles:
            item["reference_intent"] = "StoryMem recent window memory selected for local visual continuity."
        else:
            item["reference_intent"] = "StoryMem default memory reference."
        selected_raw.append(item)
    selected = _visual_reference_rows(
        store=store,
        project_id=context.project_id,
        shots=context.bundle["shots"],
        references=selected_raw,
    )
    for index, item in enumerate(selected, start=1):
        item["reference_index"] = index
    prompt_context = _visual_reference_prompt_context(
        context.bundle,
        context.shot,
        context.input_snapshot,
        [],
        selected,
    )
    return ReferenceSelectionResult(
        selected_references=selected,
        prompt_context=_prompt_for_generation_mode(
            prompt_context,
            str(context.input_snapshot.get("generation_mode") or "default"),
            default_last_frame_continuity=_uses_default_last_frame_continuity(context.shot, context.input_snapshot),
        ),
        details={
            "selection_mode": STORYMEM_MEMORY_SELECTION_MODE,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "storymem_memory": {
                "max_memory_size": STORYMEM_MEMORY_MAX_SIZE,
                "fix": STORYMEM_MEMORY_FIX,
                "keyframe_profile": STORYMEM_KEYFRAME_PROFILE,
            },
            "reference_budget": budget,
            "skipped_visual_element_planning": True,
        },
        logs=[
            _log(
                "selecting_visual_references",
                (
                    "StoryMem memory selected "
                    f"{len(selected)} of {len(candidates)} historical keyframes "
                    f"with max_memory_size={limit}, fix={min(STORYMEM_MEMORY_FIX, limit)}."
                ),
            ),
            _log(
                "selecting_visual_references",
                (
                    "static image budget: "
                    f"predefined={budget['predefined_reference_count']}, "
                    f"default_last_frame_reserved={budget['default_last_frame_reserved']}, "
                    f"storymem_memory_limit={budget['effective_max_memory_size']}"
                ),
            ),
        ],
    )


def _storymem_memory_candidates(context: ReferenceSelectionContext, store: ProjectStore) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for shot in context.previous_shots:
        if shot.get("state", {}).get("status") != "completed":
            continue
        attempt = shot.get("attempt") or {}
        attempt_id = str(attempt.get("attempt_id") or shot.get("state", {}).get("current_attempt_id") or "unknown")
        source_prompt = str(shot.get("inputs", {}).get("video_prompt") or "")
        memories = [
            memory for memory in (attempt.get("produced_visual_memory") or [])
            if isinstance(memory, dict) and memory.get("active_in_memory_pool") is not False and memory.get("asset_id")
        ]
        memories.sort(key=lambda item: int(item.get("rank") or 0))
        for memory in memories:
            asset_id = str(memory.get("asset_id") or "")
            try:
                source_path = store.asset_path(context.project_id, asset_id)
            except Exception:
                continue
            candidates.append(
                {
                    "reference_id": f"storymem-{shot['shot_id']}-{attempt_id}-{asset_id}",
                    "asset_id": asset_id,
                    "source_path": str(source_path),
                    "source_shot_id": shot["shot_id"],
                    "source_scene_num": shot.get("scene_num"),
                    "source_shot_num": shot.get("shot_num"),
                    "source_prompt": source_prompt,
                    "file": source_path.name,
                    "roles": ["default_memory"],
                    "rank": memory.get("rank"),
                    "score": None,
                    "visual_element_selection": {
                        "score": None,
                        "newly_covered_element_ids": [],
                        "already_covered_element_ids": [],
                        "should_reference": [],
                        "should_exclude": [],
                        "optional_or_uncertain": [],
                    },
                    "holistic_description": "",
                    "reference_guidance": "",
                    "_source_order_index": shot.get("order_index", 0),
                }
            )
    return candidates


def _storymem_default_memory_bank(all_memory: List[str], max_memory_size: int, fix: int) -> List[str]:
    memory_bank = list(all_memory)
    if len(memory_bank) <= max_memory_size:
        return memory_bank
    fixed = max(0, min(fix, max_memory_size))
    recent = max_memory_size - fixed
    return memory_bank[:fixed] + (memory_bank[-recent:] if recent > 0 else [])


def _storymem_default_memory_roles(memory_bank: List[str], max_memory_size: int, fix: int) -> Dict[str, List[str]]:
    if not memory_bank:
        return {}
    fixed = max(0, min(fix, max_memory_size, len(memory_bank)))
    roles: Dict[str, List[str]] = {}
    for path in memory_bank[:fixed]:
        roles.setdefault(path, []).append("early_sink_memory")
    for path in memory_bank[fixed:]:
        roles.setdefault(path, []).append("recent_window_memory")
    return roles


def _naive_reference_candidates(
    context: ReferenceSelectionContext,
    store: ProjectStore,
    text_embedding: List[float] | None,
    score_details: Dict[str, Any],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for shot in context.previous_shots:
        if shot.get("state", {}).get("status") != "completed":
            continue
        attempt = shot.get("attempt") or {}
        attempt_id = str(attempt.get("attempt_id") or shot.get("state", {}).get("current_attempt_id") or "unknown")
        source_prompt = str(shot.get("inputs", {}).get("video_prompt") or "")
        for memory in attempt.get("produced_visual_memory") or []:
            if not isinstance(memory, dict) or memory.get("active_in_memory_pool") is False:
                continue
            asset_id = str(memory.get("asset_id") or "")
            if not asset_id:
                continue
            try:
                source_path = store.asset_path(context.project_id, asset_id)
            except Exception:
                continue
            image_embedding = None
            if text_embedding is not None:
                image_embedding = _clip_image_embedding_for_asset(
                    store,
                    context.project_id,
                    asset_id,
                    score_details,
                )
            score = _cosine_similarity(text_embedding, image_embedding) if image_embedding is not None else 0.0
            selection = {
                "score": score,
                "newly_covered_element_ids": [],
                "already_covered_element_ids": [],
                "should_reference": [],
                "should_exclude": [],
                "optional_or_uncertain": [],
            }
            candidates.append(
                {
                    "reference_id": f"naive-{shot['shot_id']}-{attempt_id}-{asset_id}",
                    "asset_id": asset_id,
                    "source_path": str(source_path),
                    "source_shot_id": shot["shot_id"],
                    "source_scene_num": shot.get("scene_num"),
                    "source_shot_num": shot.get("shot_num"),
                    "source_prompt": source_prompt,
                    "file": source_path.name,
                    "roles": ["naive_top_k_memory"],
                    "rank": memory.get("rank"),
                    "score": selection["score"],
                    "reference_intent": "Naive top-k baseline selected this frame by CLIP text-image similarity.",
                    "visual_element_selection": selection,
                    "holistic_description": "",
                    "reference_guidance": "",
                    "_source_order_index": shot.get("order_index", 0),
                }
            )
    return candidates


def _clip_text_embedding(prompt: str) -> List[float]:
    import clip
    import torch

    model, device, dtype = _naive_clip_model()
    tokens = clip.tokenize([str(prompt or "")], truncate=True).to(device)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
    return [float(value) for value in embedding[0].detach().cpu().tolist()]


def _clip_image_embedding_for_asset(
    store: ProjectStore,
    project_id: str,
    asset_id: str,
    score_details: Dict[str, Any] | None = None,
) -> List[float] | None:
    project_dir = store.project_dir(project_id)
    image_path = store.asset_path(project_id, asset_id)
    cache_path = _clip_embedding_cache_path(project_dir, asset_id)
    cached = read_json(cache_path, {})
    image_stat = image_path.stat()
    if (
        isinstance(cached, dict)
        and cached.get("model") == "ViT-B/32"
        and cached.get("asset_id") == asset_id
        and cached.get("source_mtime_ns") == image_stat.st_mtime_ns
        and isinstance(cached.get("embedding"), list)
    ):
        if score_details is not None:
            score_details["cache_hits"] = int(score_details.get("cache_hits") or 0) + 1
        return [float(value) for value in cached["embedding"]]
    try:
        embedding = _clip_image_embedding(image_path)
    except Exception as exc:
        if score_details is not None:
            failures = score_details.setdefault("score_failures", [])
            if isinstance(failures, list):
                failures.append({"asset_id": asset_id, "type": type(exc).__name__, "message": str(exc)})
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        cache_path,
        {
            "schema_version": SCHEMA_VERSION,
            "asset_id": asset_id,
            "model": "ViT-B/32",
            "source_path": _relative(project_dir, image_path),
            "source_mtime_ns": image_stat.st_mtime_ns,
            "embedding": embedding,
            "created_at": _now(),
        },
    )
    if score_details is not None:
        score_details["cache_writes"] = int(score_details.get("cache_writes") or 0) + 1
    return embedding


def _clip_image_embedding(image_path: Path) -> List[float]:
    import numpy as np
    import torch
    from PIL import Image
    from extract_keyframes import _clip_preprocess_tensor

    model, device, dtype = _naive_clip_model()
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        array = np.asarray(image).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    tensor = _clip_preprocess_tensor(tensor).to(device, dtype=dtype)
    with torch.no_grad():
        embedding = model.encode_image(tensor)
        embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
    return [float(value) for value in embedding[0].detach().cpu().tolist()]


def _naive_clip_model():
    from extract_keyframes import _get_clip_model

    device = os.environ.get("VIDEOGEN_NOTEBOOK_NAIVE_CLIP_DEVICE", "cpu")
    use_half = device != "cpu" and _env_flag("VIDEOGEN_NOTEBOOK_NAIVE_CLIP_HALF", default=False)
    return _get_clip_model(device=device, use_half=use_half)


def _clip_embedding_cache_path(project_dir: Path, asset_id: str) -> Path:
    safe_asset_id = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in asset_id)
    return project_dir / "assets" / "embeddings" / f"{safe_asset_id}_clip_vit_b_32.json"


def _cosine_similarity(left: List[float] | None, right: List[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(float(a) * float(b) for a, b in zip(left, right)))


def _naive_produced_memory_from_keyframes(
    keyframe_paths: List[str],
    shot_id: str,
    attempt_id: str,
) -> List[Dict[str, Any]]:
    produced: List[Dict[str, Any]] = []
    for index, path in enumerate(keyframe_paths, start=1):
        produced.append(
            {
                "asset_id": Path(path).stem,
                "rank": index,
                "visible_elements": [],
                "elements": [],
                "annotation": "",
                "holistic_description": "",
                "active_in_memory_pool": True,
                "source_shot_id": shot_id,
                "attempt_id": attempt_id,
            }
        )
    return produced


def _fake_references(previous_shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    references = []
    for shot in previous_shots[-2:]:
        attempt_id = shot.get("state", {}).get("current_attempt_id")
        if not attempt_id:
            continue
        references.append(
            {
                "reference_id": f"ref-{shot['shot_id']}-{attempt_id}",
                "source_shot_id": shot["shot_id"],
                "roles": ["fake_historical_reference"],
                "covered_elements": [f"element-{shot['shot_id']}-001"],
                "conflict_elements": [],
                "score": 1.0,
                "reason": "Previous completed shot used as fake historical evidence.",
                "holistic_description": f"Fake holistic description for source shot {shot['shot_id']}.",
                "reference_guidance": f"Use source shot {shot['shot_id']} as a broad fake visual reference.",
            }
        )
    return references


def _shot_spec_from_bundle_shot(shot: Dict[str, Any]) -> ShotSpec:
    inputs = shot["inputs"]
    return ShotSpec(
        scene={},
        scene_num=int(shot["scene_num"]),
        shot_num=int(shot["shot_num"]),
        prompt=str(inputs["video_prompt"]),
        is_cut=bool(inputs["is_cut"]),
        duration_seconds=int(inputs["duration_seconds"]),
    )


def _story_script(bundle: Dict[str, Any]) -> Dict[str, Any]:
    story = bundle.get("story") or {}
    source = story.get("source")
    if isinstance(source, dict) and source.get("scenes"):
        return source
    scenes_by_num: Dict[int, List[str]] = {}
    for shot in bundle.get("shots", []):
        scenes_by_num.setdefault(int(shot["scene_num"]), []).append(
            str(shot.get("inputs", {}).get("video_prompt", ""))
        )
    return {
        "story_name": bundle.get("project", {}).get("name", ""),
        "scenes": [
            {"scene_num": scene_num, "video_prompts": prompts}
            for scene_num, prompts in sorted(scenes_by_num.items())
        ],
    }


def _story_script_through_shot(bundle: Dict[str, Any], shot: Dict[str, Any]) -> Dict[str, Any]:
    try:
        current_order = int(shot.get("order_index", 0))
    except (TypeError, ValueError):
        current_order = 0
    scenes_by_num: Dict[int, List[str]] = {}
    for item in bundle.get("shots", []):
        try:
            order_index = int(item.get("order_index", 0))
        except (TypeError, ValueError):
            order_index = 0
        if order_index > current_order:
            continue
        try:
            scene_num = int(item.get("scene_num", 1))
        except (TypeError, ValueError):
            scene_num = 1
        prompt = str(item.get("inputs", {}).get("video_prompt", "")).strip()
        if prompt:
            scenes_by_num.setdefault(scene_num, []).append(prompt)
    return {
        "story_name": bundle.get("project", {}).get("name", ""),
        "scenes": [
            {"scene_num": scene_num, "video_prompts": prompts}
            for scene_num, prompts in sorted(scenes_by_num.items())
        ],
    }


def _fake_produced_memory(shot: Dict[str, Any], attempt_id: str) -> List[Dict[str, Any]]:
    return [
        {
            "asset_id": f"img_{shot['shot_id']}_{attempt_id}_001",
            "rank": 1,
            "visible_elements": [f"element-{shot['shot_id']}-001"],
            "annotation": "Fake keyframe annotation.",
            "holistic_description": "Fake holistic keyframe description with composition and story context.",
            "active_in_memory_pool": True,
        }
    ]


def _placeholder_produced_memory(
    asset_id: str,
    visual_status: List[Dict[str, Any]],
    annotation: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "asset_id": asset_id,
            "rank": 1,
            "visible_elements": [item["element_id"] for item in visual_status],
            "annotation": annotation,
            "holistic_description": "",
            "active_in_memory_pool": True,
        }
    ]


def _seedance_content(
    project_id: str,
    store: ProjectStore,
    submitted_prompt: str,
    references: List[Dict[str, Any]],
    *,
    include_text: bool = True,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if include_text:
        content.append({"type": "text", "text": submitted_prompt})
    for reference in references:
        asset_id = reference.get("asset_id")
        if asset_id:
            path = store.asset_path(project_id, str(asset_id))
        elif reference.get("source_path"):
            path = Path(str(reference["source_path"]))
        else:
            continue
        content.append(image_item_with_metadata(str(path), reference))
    return content


def _run_config_for_bundle(bundle: Dict[str, Any], output_dir: Path) -> RunConfig:
    settings = bundle.get("settings") or {}
    generation = settings.get("generation") or {}
    visual = settings.get("visual_element_memory") or {}
    scoring = visual.get("scoring") or {}
    seedance = settings.get("seedance") or {}
    return RunConfig(
        output_dir=str(output_dir),
        duration=int(generation.get("default_duration_seconds") or 8),
        ratio=str(seedance.get("ratio") or "16:9"),
        resolution=str(seedance.get("resolution") or "720p"),
        generate_audio=bool(generation.get("audio", True)),
        enhanced_text_prompt=True,
        seedance_model=str(seedance.get("model") or DEFAULT_SEEDANCE_MODEL),
        visual_element_memory=True,
        visual_element_selection_mode=str(visual.get("selection_mode") or "greedy_coverage"),
        visual_element_sink_frame_count=int(visual.get("sink_frame_count") or 0),
        visual_element_max_retrieved_frames=int(visual.get("max_retrieved_frames") or 4),
        visual_element_weight_character=float(scoring.get("character_weight", 3.0)),
        visual_element_weight_scene=float(scoring.get("scene_weight", 2.0)),
        visual_element_weight_object=float(scoring.get("object_weight", 1.5)),
        visual_element_weight_reference_uncovered=float(scoring.get("reference_uncovered_weight", 1.0)),
        visual_element_weight_reference_covered=float(scoring.get("reference_covered_weight", 0.2)),
        visual_element_weight_optional=float(scoring.get("optional_weight", 0.1)),
        visual_element_weight_exclude=float(scoring.get("exclude_weight", -0.1)),
        visual_element_weight_quality_full=float(scoring.get("quality_full_weight", 1.0)),
        visual_element_weight_quality_partial=float(scoring.get("quality_partial_weight", 0.2)),
        visual_element_weight_quality_weak=float(scoring.get("quality_weak_weight", 0.1)),
        smooth_reference_seconds=float(generation.get("smooth_reference_seconds", 2.0)),
    )


def _smooth_reference_seconds(context: ShotExecutionContext) -> float:
    settings = context.input_snapshot.get("project_settings")
    if not isinstance(settings, dict):
        settings = context.bundle.get("settings") if isinstance(context.bundle, dict) else {}
    generation = settings.get("generation") if isinstance(settings, dict) else {}
    try:
        value = float((generation or {}).get("smooth_reference_seconds", 2.0))
    except (TypeError, ValueError):
        value = 2.0
    return min(max(value, 0.5), 10.0)


def _visual_memory_snapshot_from_shots(shots: List[Dict[str, Any]]) -> Dict[str, Any]:
    registry_by_id: Dict[str, Dict[str, Any]] = {}
    annotations: List[Dict[str, Any]] = []
    for shot in shots:
        if shot.get("state", {}).get("status") != "completed":
            continue
        attempt = shot.get("attempt") or {}
        for row in attempt.get("visual_element_status") or []:
            element_id = row.get("element_id") or row.get("id")
            if not element_id or element_id in registry_by_id:
                continue
            registry_by_id[str(element_id)] = {
                "id": str(element_id),
                "name": str(row.get("name") or element_id),
                "type": _normalize_visual_element_type(row.get("type")),
                "introduced_at": str(row.get("introduced_at") or row.get("source_shot_id") or shot["shot_id"]),
                "notes": str(row.get("notes") or ""),
            }
        for memory in attempt.get("produced_visual_memory") or []:
            annotation = memory.get("frame_annotation")
            if isinstance(annotation, dict):
                item = dict(annotation)
                item.setdefault("holistic_description", memory.get("holistic_description") or "")
                annotations.append(item)
    return {
        "registry": list(registry_by_id.values()),
        "annotations": annotations,
        "records": [],
    }


def _normalize_visual_element_type(value: object) -> str:
    raw = str(value or "object").strip()
    aliases = {
        "environment": "scene",
        "location": "scene",
        "style": "scene",
        "action": "object",
        "other": "object",
    }
    normalized = aliases.get(raw, raw)
    if normalized in {"character", "scene", "object"}:
        return normalized
    return "object"


def _visual_status_rows(report_or_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    plan = report_or_plan.get("visual_element_plan") or report_or_plan
    registry_by_id = {
        str(item.get("id")): item
        for item in (report_or_plan.get("visual_element_registry_after") or report_or_plan.get("registry_after") or [])
        if item.get("id")
    }
    inserted_by_id = {
        str(item.get("id")): item
        for item in (report_or_plan.get("visual_element_inserted") or [])
        if item.get("id")
    }
    element_by_id = {**registry_by_id, **inserted_by_id}
    rows: List[Dict[str, Any]] = []
    for status in ("should_reference", "should_exclude", "optional_or_uncertain"):
        for item in plan.get(status) or []:
            element_id = str(item.get("id") or "")
            metadata = element_by_id.get(element_id, {})
            rows.append(
                {
                    "element_id": item.get("id"),
                    "name": item.get("name"),
                    "type": item.get("type"),
                    "introduced_at": item.get("introduced_at") or metadata.get("introduced_at") or "",
                    "notes": item.get("notes") or metadata.get("notes") or "",
                    "status": status,
                    "reason": item.get("reason") or "",
                }
            )
    for item in plan.get("new_elements") or []:
        inserted = inserted_by_id.get(str(item.get("id"))) or {}
        rows.append(
            {
                "element_id": item.get("id"),
                "name": item.get("name"),
                "type": item.get("type"),
                "introduced_at": inserted.get("introduced_at") or item.get("introduced_at") or item.get("id"),
                "notes": inserted.get("notes") or item.get("notes") or item.get("reason") or "",
                "status": "new",
                "reason": item.get("reason") or "",
            }
        )
    return rows


def _visual_reference_rows(
    *,
    store: ProjectStore,
    project_id: str,
    shots: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    shot_id_by_scene_shot = {
        (int(shot["scene_num"]), int(shot["shot_num"])): shot["shot_id"]
        for shot in shots
    }
    for index, reference in enumerate(references, start=1):
        item = dict(reference)
        source_path = str(item.get("source_path") or "")
        selection = item.get("visual_element_selection") or {}
        roles = item.get("roles") or []
        source_scene = item.get("source_scene_num")
        source_shot = item.get("source_shot_num")
        source_shot_id = item.get("source_shot_id")
        if not source_shot_id and source_scene and source_shot:
            source_shot_id = shot_id_by_scene_shot.get((int(source_scene), int(source_shot)))
        row = {
            **item,
            "reference_id": item.get("reference_id") or f"ref-{index:03d}",
            "asset_id": item.get("asset_id") or _asset_id_for_path(store, project_id, source_path),
            "source_shot_id": source_shot_id,
            "roles": roles,
            "covered_elements": _covered_selection_element_ids(selection),
            "conflict_elements": _selection_element_ids(selection, "should_exclude"),
            "score": item.get("score"),
            "reason": item.get("reference_intent") or "",
            "holistic_description": str(item.get("holistic_description") or ""),
            "reference_guidance": str(item.get("reference_guidance") or ""),
        }
        rows.append(row)
    return rows


def _covered_selection_element_ids(selection: Dict[str, Any]) -> List[str]:
    values = [
        *(selection.get("newly_covered_element_ids") or []),
        *(selection.get("already_covered_element_ids") or []),
    ]
    return [str(item) for item in values if item]


def _selection_element_ids(selection: Dict[str, Any], key: str) -> List[str]:
    values = selection.get(key) or []
    result = []
    for item in values:
        if isinstance(item, dict):
            result.append(str(item.get("id") or item.get("name") or ""))
    return [item for item in result if item]


def _asset_id_for_path(store: ProjectStore, project_id: str, source_path: str) -> str | None:
    if not source_path:
        return None
    project_dir = store.project_dir(project_id).resolve()
    try:
        source = Path(source_path).resolve()
    except OSError:
        return None
    assets = store.get_project(project_id).get("assets", {})
    for asset_id, asset in assets.items():
        try:
            path = (project_dir / str(asset.get("path"))).resolve()
        except OSError:
            continue
        if path == source:
            return str(asset_id)
    return None


def _image_asset_record(
    project_dir: Path,
    target_path: Path,
    asset_id: str,
    *,
    source_type: str,
    shot_id: str,
    attempt_id: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "kind": "image",
        "path": _relative(project_dir, target_path),
        "created_at": _now(),
        "source": {
            "type": source_type,
            "shot_id": shot_id,
            "attempt_id": attempt_id,
        },
        "metadata": metadata,
    }


def _existing_memory_paths(
    store: ProjectStore,
    project_id: str,
    shots: List[Dict[str, Any]],
    current_shot_id: str,
) -> List[str]:
    paths: List[str] = []
    for shot in shots:
        if shot["shot_id"] == current_shot_id:
            break
        if shot.get("state", {}).get("status") != "completed":
            continue
        attempt = shot.get("attempt") or {}
        for memory in attempt.get("produced_visual_memory") or []:
            asset_id = memory.get("asset_id")
            if not asset_id:
                continue
            try:
                paths.append(str(store.asset_path(project_id, str(asset_id))))
            except Exception:
                continue
    return paths


def _storymem_existing_memory_paths(store: ProjectStore, context: VideoPostprocessContext) -> List[str]:
    selected_path = context.attempt_dir / "reference_selection" / "selected_references.json"
    selected = read_json(selected_path, None)
    if isinstance(selected, list):
        return _asset_paths_from_reference_rows(store, context.project_id, selected)
    candidates = _storymem_memory_candidates(
        ReferenceSelectionContext(
            project_id=context.project_id,
            bundle=context.bundle,
            shot=context.shot,
            previous_shots=[
                shot
                for shot in context.bundle["shots"]
                if int(shot.get("order_index") or 0) < int(context.shot.get("order_index") or 0)
            ],
            attempt_id=context.attempt_id,
            attempt_dir=context.attempt_dir,
            input_snapshot=dict(context.shot.get("inputs") or {}),
            visual_element_status=context.visual_element_status,
            visual_element_details={},
        ),
        store,
    )
    memory_bank = _storymem_default_memory_bank(
        [str(item["source_path"]) for item in candidates],
        STORYMEM_MEMORY_MAX_SIZE,
        STORYMEM_MEMORY_FIX,
    )
    return memory_bank


def _asset_paths_from_reference_rows(
    store: ProjectStore,
    project_id: str,
    references: List[Dict[str, Any]],
) -> List[str]:
    paths: List[str] = []
    for reference in references:
        if not isinstance(reference, dict):
            continue
        asset_id = reference.get("asset_id")
        if asset_id:
            try:
                paths.append(str(store.asset_path(project_id, str(asset_id))))
                continue
            except Exception:
                pass
        source_path = reference.get("source_path")
        if source_path and Path(str(source_path)).is_file():
            paths.append(str(Path(str(source_path))))
    return paths


def _produced_memory_from_annotations(
    annotations: List[Dict[str, Any]],
    shot_id: str,
    attempt_id: str,
) -> List[Dict[str, Any]]:
    produced = []
    for index, annotation in enumerate(annotations, start=1):
        asset_id = Path(str(annotation.get("frame_path") or f"img_{shot_id}_{attempt_id}_{index:03d}")).stem
        elements = annotation.get("elements") or []
        visible_ids = [str(item.get("id")) for item in elements if item.get("id")]
        visible_names = [str(item.get("name")) for item in elements if item.get("name")]
        produced.append(
            {
                "asset_id": asset_id,
                "rank": index,
                "visible_elements": visible_ids,
                "annotation": "Visible: " + ", ".join(visible_names) if visible_names else "No known visual elements visible.",
                "holistic_description": str(annotation.get("holistic_description") or ""),
                "active_in_memory_pool": True,
                "frame_annotation": annotation,
            }
        )
    return produced


def _uses_default_last_frame_continuity(shot: Dict[str, Any], input_snapshot: Dict[str, Any]) -> bool:
    if str(input_snapshot.get("generation_mode") or "default") != "default":
        return False
    if int(shot.get("order_index") or 0) <= 0:
        return False
    return not bool(input_snapshot.get("is_cut", shot.get("inputs", {}).get("is_cut", False)))


def _uses_smooth_reference_video(shot: Dict[str, Any], input_snapshot: Dict[str, Any]) -> bool:
    if str(input_snapshot.get("generation_mode") or "default") != "smooth":
        return False
    if int(shot.get("order_index") or 0) <= 0:
        return False
    return not bool(input_snapshot.get("is_cut", shot.get("inputs", {}).get("is_cut", False)))


def _prompt_for_generation_mode(
    prompt: str,
    generation_mode: str,
    *,
    default_last_frame_continuity: bool = False,
) -> str:
    if generation_mode == "smooth":
        return (
            f"{SMOOTH_CONTINUATION_INSTRUCTION}\n\n"
            "Input media numbering for this Smooth shot:\n"
            "- Video 1 是上一段原始视频的尾部片段。当前 shot 必须向后延长 Video 1，从 Video 1 最后一帧之后的下一时刻自然继续。\n"
            "- Static reference images are numbered separately as Image 1, Image 2, ... after Video 1. "
            "When the prompt below says Image 1, it means the first static image reference, not Video 1.\n\n"
            f"{prompt}"
        )
    if generation_mode == "last_frame_only":
        return (
            "Use Image 1 as the exact first-frame continuity input from the previous Seedance output. "
            "Continue naturally from that image; do not restart or replay the previous shot.\n\n"
            f"{prompt}"
        )
    if generation_mode == "default" and default_last_frame_continuity:
        return (
            "The final reference image is the previous shot's ending frame. "
            "Use this final reference image as a first-frame continuity constraint for the new video; "
            "earlier static images are visual memory references.\n\n"
            f"{prompt}"
        )
    return prompt


def _add_reference_guidance(
    context: ReferenceSelectionContext,
    references: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    rows = [{**item, "reference_guidance": str(item.get("reference_guidance") or "")} for item in references]
    eligible = [item for item in rows if str(item.get("holistic_description") or "").strip()]
    if not eligible:
        _append_style_reference_guidance(rows)
        return rows, {"status": "skipped", "reason": "no selected references with holistic_description"}, []

    prompt = _reference_guidance_prompt(context, eligible)
    try:
        raw_text, metadata = _call_ark_chat(
            messages=[{"role": "user", "content": prompt}],
            model=DEFAULT_VISUAL_ELEMENT_MODEL,
            max_tokens=VISUAL_ELEMENT_MAX_OUTPUT_TOKENS,
            timeout_seconds=120,
        )
        parsed = _extract_json_object(raw_text)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("references"), list):
            raise ValueError("reference guidance output must contain a references list")
        guidance_by_id: Dict[str, str] = {}
        for item in parsed["references"]:
            if not isinstance(item, dict):
                continue
            reference_id = str(item.get("reference_id") or "")
            if reference_id:
                guidance_by_id[reference_id] = str(item.get("reference_guidance") or "").strip()
        for row in rows:
            if str(row.get("holistic_description") or "").strip():
                row["reference_guidance"] = guidance_by_id.get(str(row.get("reference_id") or ""), "")
            else:
                row["reference_guidance"] = ""
        _append_style_reference_guidance(rows)
        details = {
            "status": "completed",
            "prompt": prompt,
            "raw_response": raw_text,
            "metadata": metadata,
            "parsed": parsed,
        }
        return rows, details, [_log("selecting_visual_references", "reference guidance generated for selected references")]
    except Exception as exc:
        for row in rows:
            row["reference_guidance"] = ""
        _append_style_reference_guidance(rows)
        details = {
            "status": "failed",
            "prompt": prompt,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        return rows, details, [_log("selecting_visual_references", f"reference guidance failed without blocking selection: {exc}")]


def _append_style_reference_guidance(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        if _reference_covers_needed_elements(row):
            continue
        guidance = str(row.get("reference_guidance") or "").strip()
        if STYLE_REFERENCE_GUIDANCE in guidance:
            continue
        row["reference_guidance"] = f"{guidance}\n{STYLE_REFERENCE_GUIDANCE}".strip()


def _reference_covers_needed_elements(row: Dict[str, Any]) -> bool:
    if row.get("covered_elements"):
        return True
    selection = row.get("visual_element_selection") if isinstance(row.get("visual_element_selection"), dict) else {}
    return bool(selection.get("should_reference"))


def _reference_guidance_prompt(context: ReferenceSelectionContext, references: List[Dict[str, Any]]) -> str:
    payload = {
        "story_so_far": _shot_context_rows(context.bundle["shots"][: int(context.shot["order_index"]) + 1]),
        "current_shot": {
            "scene_num": context.shot.get("scene_num"),
            "shot_num": context.shot.get("shot_num"),
            "video_prompt": context.input_snapshot.get("video_prompt"),
        },
        "visual_element_status": context.visual_element_status,
        "selected_references": [
            {
                "reference_id": item.get("reference_id"),
                "source_scene_num": item.get("source_scene_num"),
                "source_shot_num": item.get("source_shot_num"),
                "holistic_description": item.get("holistic_description") or "",
                "visible_elements": item.get("covered_elements") or item.get("visible_elements") or [],
                "coverage": item.get("visual_element_selection", {}).get("should_reference", []),
                "exclude": item.get("visual_element_selection", {}).get("should_exclude", []),
                "optional_visible": item.get("visual_element_selection", {}).get("optional_or_uncertain", []),
                "score": item.get("score"),
            }
            for item in references
        ],
    }
    return (
        "你是长视频生成项目中的历史参考帧解释助手。系统已经用客观元素覆盖规则选好了参考帧，"
        "你不能新增、删除、替换或重排参考帧，只能为每个已有 reference_id 补充 reference_guidance。\n"
        "reference_guidance 应用中文简洁说明：为什么这张图适合作为当前 shot 的整体参考，以及生成时应如何使用其整体构图、氛围或剧情关系。"
        "不要重复列出客观覆盖分数，不要引入画面中不存在的信息。若 holistic_description 为空，reference_guidance 必须为空字符串。\n"
        "只返回 JSON 对象，格式为：\n"
        '{"references":[{"reference_id":"ref-001","reference_guidance":"..."}]}\n\n'
        "输入数据：\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _visual_reference_prompt_context(
    bundle: Dict[str, Any],
    shot: Dict[str, Any],
    input_snapshot: Dict[str, Any],
    visual_status: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
) -> str:
    full_script_enabled = _prompt_module_enabled(input_snapshot, bundle, "full_script_context")
    visual_plan_enabled = _prompt_module_enabled(input_snapshot, bundle, "visual_element_plan")
    holistic_guidance_enabled = _prompt_module_enabled(input_snapshot, bundle, "holistic_guidance")
    should_reference_enabled = _prompt_module_enabled(input_snapshot, bundle, "should_reference")
    should_exclude_enabled = _prompt_module_enabled(input_snapshot, bundle, "should_exclude")
    predefined_references = _predefined_references_from_input(input_snapshot)
    predefined_count = len(predefined_references)
    predefined_lines = _predefined_reference_prompt_lines(predefined_references)
    reference_lines: List[str] = []
    if references:
        for index, ref in enumerate(references, start=1):
            ref_index = int(ref.get("reference_index") or index) + predefined_count
            reference_lines.append(
                f"参考图片 Image {ref_index} 来自 Scene {ref.get('source_scene_num')} / Shot {ref.get('source_shot_num')}："
            )
            if "visual_sink_memory" in (ref.get("roles") or []):
                reference_lines.extend(
                    [
                        "这是一张早期参考锚点图，仅用于稳定整体角色身份、画面风格、色彩气质和长期视觉一致性。",
                        "不要因为这张图中的旧场景、旧动作或未在当前镜头要求中出现的物体，就把它们带入当前画面。",
                    ]
                )
                continue
            holistic = str(ref.get("holistic_description") or "").strip()
            guidance = str(ref.get("reference_guidance") or "").strip()
            if holistic_guidance_enabled and holistic:
                reference_lines.append(holistic)
            if holistic_guidance_enabled and guidance:
                reference_lines.append(guidance)
            selection = ref.get("visual_element_selection") or {}
            objective_lines: List[str] = []
            if should_reference_enabled:
                objective_lines.extend(
                    [
                        "应参考其中的：",
                        _reference_names(selection, "should_reference", ref.get("covered_elements")),
                    ]
                )
            if should_exclude_enabled:
                objective_lines.extend(
                    [
                        "不应引入其中的：",
                        _reference_names(selection, "should_exclude", ref.get("conflict_elements")),
                    ]
                )
            if objective_lines:
                reference_lines.append("针对视觉计划中各元素的客观参考约束如下:")
                reference_lines.extend(objective_lines)
    else:
        reference_lines.append("本镜头不提供历史参考图。")

    scene_num = shot.get("scene_num")
    shot_num = shot.get("shot_num")
    prompt = str(input_snapshot.get("video_prompt") or shot.get("inputs", {}).get("video_prompt") or "")
    force_animation = _force_animation_enabled(input_snapshot, bundle)
    reference_video_instruction = (
        "本次输入媒体包含参考视频，请生成参考视频的延长视频：生成的视频应从参考视频的最后一帧之后的下一时刻开始，自然延续其中的人物动作、物体运动、镜头运动方向和速度，并保持整体画面风格、色彩气质和视觉元素的一致性。"
    )
    current_task_lines = [
        "[当前 shot 的生成任务]",
        *([FORCE_ANIMATION_PROMPT_PREFIX] if force_animation else []),
        f"现在请只生成 Scene {scene_num} / Shot {shot_num}：",
        escape_prompt_line(prompt),
        *([reference_video_instruction] if _uses_smooth_reference_video(shot, input_snapshot) else []),
    ]
    overall_constraint_lines = [
        "[总体约束]",
        *([FORCE_ANIMATION_PROMPT_PREFIX] if force_animation else []),
        "当前 shot 的生成任务优先级最高。",
    ]
    if full_script_enabled:
        overall_constraint_lines.append(
            "前序完整剧本仅用于上下文理解，不要生成其他镜头的故事内容。对于本镜头shot中未被详细描述的历史元素，可参考前序镜头中的描述及对应的参考图。"
        )
    if visual_plan_enabled:
        overall_constraint_lines.extend(
            [
                "请只参考视觉元素计划中明确指定的视觉元素，不要将参考图中的与本镜头无关的元素加入当前画面。",
                "如果参考图说明与当前 shot 描述冲突，以当前 shot 描述和本 shot 的视觉元素计划为准。",
            ]
        )
    else:
        overall_constraint_lines.append("如果参考图说明与当前 shot 描述冲突，以当前 shot 描述为准。")
    overall_constraint_lines.extend(
        [
            "除非剧本明确要求，不要在视频结束时添加镜头远离、淡出、变暗等转场效果。",
            "Maintain a stable shot scale through the end of the clip unless the script explicitly asks otherwise.",
            "Do NOT zoom out, pull the camera back, fade out, dim the image, or add an ending transition at the end of video, unless the script explicitly asks for it, as next shot may continue the same scene.",
        ]
    )
    prompt_sections: List[str] = ["\n".join(current_task_lines)]
    if full_script_enabled:
        prompt_sections.append(
            "\n".join(
                [
                    "[前序完整剧本]",
                    "以下是当前 shot 及其之前的 shot 级描述，仅用于理解已发生剧情、人物关系、场景切换，以及理解哪些历史元素不应出现在当前镜头中。",
                    "当前任务只生成指定 shot。不要生成其他 shot 的内容，也不要引入尚未在前序剧本或当前 shot 中出现的人物、场景或物体。",
                    story_prompt_outline(_story_script_through_shot(bundle, shot)),
                ]
            )
        )
    if visual_plan_enabled:
        prompt_sections.append(
            "\n".join(
                [
                    "[本 shot 的视觉元素计划]",
                    "本镜头应参考并保持一致的历史元素：",
                    _status_names(visual_status, "should_reference"),
                    "本镜头不应引入的历史元素：",
                    _status_names(visual_status, "should_exclude"),
                    "不确定的视觉元素：",
                    _status_names(visual_status, "optional_or_uncertain"),
                    "本镜头新引入的视觉元素：",
                    _status_names(visual_status, "new"),
                ]
            )
        )
    if predefined_lines:
        prompt_sections.append(
            "\n".join(
                [
                    "[预定义参考图说明]",
                    "以下参考图片是当前 shot 的预定义输入，由用户或剧本指定。请按每张图的说明使用，不要把未被当前 shot 要求的内容机械复刻进画面。",
                    *predefined_lines,
                ]
            )
        )
    prompt_sections.append(
        "\n".join(
            [
                "[历史参考图说明]",
                "以下参考图片是从已生成的历史Shot视频中提取的关键帧，仅用于保持指定视觉元素或画面风格的一致性，不代表整张图都应被复刻。",
                *reference_lines,
            ]
        )
    )
    prompt_sections.append("\n".join(overall_constraint_lines))
    prompt_text = "\n\n".join(prompt_sections)
    return prompt_text


def _predefined_references_from_input(input_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    references = input_snapshot.get("predefined_references") if isinstance(input_snapshot, dict) else []
    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(references or [], start=1):
        if not isinstance(item, dict):
            continue
        asset_id = str(item.get("asset_id") or "").strip()
        image_path = str(item.get("image_path") or item.get("source_path") or "").strip()
        if not asset_id and not image_path:
            continue
        normalized.append(
            {
                "id": str(item.get("id") or f"pref-{index:04d}"),
                "reference_id": str(item.get("id") or f"pref-{index:04d}"),
                "asset_id": asset_id,
                "source_path": image_path,
                "image_path": image_path,
                "label": str(item.get("label") or f"Predefined reference {index}").strip(),
                "guidance": str(item.get("guidance") or "").strip(),
                "roles": ["predefined_reference"],
                "source_type": "predefined_reference",
            }
        )
    return normalized


def _predefined_reference_prompt_lines(references: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for index, reference in enumerate(references, start=1):
        label = str(reference.get("label") or f"Predefined reference {index}").strip()
        guidance = str(reference.get("guidance") or "").strip()
        lines.append(f"参考图片 Image {index}: {label}")
        if guidance:
            lines.append(guidance)
    return lines


def _combined_static_references(context: Any, selected_references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    predefined = _predefined_references_from_input(context.input_snapshot)
    if len(predefined) > MAX_REFERENCE_IMAGES:
        raise BackendExecutionError(
            f"Predefined references exceed Seedance's {MAX_REFERENCE_IMAGES}-image limit"
        )
    combined: List[Dict[str, Any]] = []
    for index, reference in enumerate(predefined, start=1):
        item = dict(reference)
        item["reference_index"] = index
        combined.append(item)
    offset = len(combined)
    for index, reference in enumerate(selected_references or [], start=1):
        item = dict(reference)
        item["reference_index"] = offset + index
        combined.append(item)
    if len(combined) > MAX_REFERENCE_IMAGES:
        raise BackendExecutionError(
            f"Static reference images exceed Seedance's {MAX_REFERENCE_IMAGES}-image limit"
        )
    return combined


def _effective_reference_budget(context: ReferenceSelectionContext) -> Dict[str, Any]:
    predefined_count = len(_predefined_references_from_input(context.input_snapshot))
    if predefined_count > MAX_REFERENCE_IMAGES:
        raise BackendExecutionError(
            f"Predefined references exceed Seedance's {MAX_REFERENCE_IMAGES}-image limit"
        )
    memory_settings = (context.bundle.get("settings") or {}).get("visual_element_memory") or {}
    try:
        configured_sink = int(memory_settings.get("sink_frame_count", 0))
    except (TypeError, ValueError):
        configured_sink = 0
    try:
        configured_retrieved = int(memory_settings.get("max_retrieved_frames", 4))
    except (TypeError, ValueError):
        configured_retrieved = 4
    remaining = max(0, MAX_REFERENCE_IMAGES - predefined_count)
    effective_sink = min(max(configured_sink, 0), remaining)
    effective_retrieved = min(max(configured_retrieved, 0), max(0, remaining - effective_sink))
    return {
        "max_static_reference_images": MAX_REFERENCE_IMAGES,
        "predefined_reference_count": predefined_count,
        "configured_sink_frame_count": max(configured_sink, 0),
        "configured_max_retrieved_frames": max(configured_retrieved, 0),
        "remaining_slots_after_predefined": remaining,
        "effective_sink_frame_count": effective_sink,
        "effective_max_retrieved_frames": effective_retrieved,
    }


def _naive_reference_budget(context: ReferenceSelectionContext) -> Dict[str, Any]:
    budget = _effective_reference_budget(context)
    budget["effective_sink_frame_count"] = 0
    budget["effective_max_retrieved_frames"] = min(
        int(budget["configured_max_retrieved_frames"]),
        int(budget["remaining_slots_after_predefined"]),
    )
    return budget


def _storymem_reference_budget(context: ReferenceSelectionContext) -> Dict[str, Any]:
    predefined_count = len(_predefined_references_from_input(context.input_snapshot))
    if predefined_count > MAX_REFERENCE_IMAGES:
        raise BackendExecutionError(
            f"Predefined references exceed Seedance's {MAX_REFERENCE_IMAGES}-image limit"
        )
    default_last_frame_reserved = 1 if _uses_default_last_frame_continuity(context.shot, context.input_snapshot) else 0
    remaining = max(0, MAX_REFERENCE_IMAGES - predefined_count - default_last_frame_reserved)
    effective_max_memory_size = min(STORYMEM_MEMORY_MAX_SIZE, remaining)
    return {
        "max_static_reference_images": MAX_REFERENCE_IMAGES,
        "predefined_reference_count": predefined_count,
        "default_last_frame_reserved": default_last_frame_reserved,
        "configured_storymem_max_memory_size": STORYMEM_MEMORY_MAX_SIZE,
        "configured_storymem_fix": STORYMEM_MEMORY_FIX,
        "remaining_slots_after_predefined_and_continuity": remaining,
        "effective_max_memory_size": effective_max_memory_size,
        "effective_sink_frame_count": min(STORYMEM_MEMORY_FIX, effective_max_memory_size),
        "effective_recent_frame_count": max(0, effective_max_memory_size - min(STORYMEM_MEMORY_FIX, effective_max_memory_size)),
    }


def _maybe_force_animation_prompt(prompt: str, input_snapshot: Dict[str, Any], bundle: Dict[str, Any] | None = None) -> str:
    if not _force_animation_enabled(input_snapshot, bundle):
        return prompt
    text = str(prompt or "").lstrip()
    if FORCE_ANIMATION_PROMPT_PREFIX in text:
        return str(prompt or "")
    return f"{FORCE_ANIMATION_PROMPT_PREFIX}\n\n{prompt}"


def _force_animation_enabled(input_snapshot: Dict[str, Any], bundle: Dict[str, Any] | None = None) -> bool:
    settings = input_snapshot.get("project_settings") if isinstance(input_snapshot, dict) else None
    if not isinstance(settings, dict) and isinstance(bundle, dict):
        settings = bundle.get("settings")
    generation = settings.get("generation") if isinstance(settings, dict) else {}
    return bool((generation or {}).get("force_animation_style", True))


def _prompt_module_enabled(input_snapshot: Dict[str, Any], bundle: Dict[str, Any] | None, key: str) -> bool:
    settings = input_snapshot.get("project_settings") if isinstance(input_snapshot, dict) else None
    if not isinstance(settings, dict) and isinstance(bundle, dict):
        settings = bundle.get("settings")
    modules = settings.get("prompt_modules") if isinstance(settings, dict) else {}
    return bool((modules or {}).get(key, True))


def _shot_context_rows(shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "scene_num": item.get("scene_num"),
            "shot_num": item.get("shot_num"),
            "video_prompt": item.get("inputs", {}).get("video_prompt", ""),
        }
        for item in shots
    ]


def _status_names(rows: List[Dict[str, Any]], status: str) -> str:
    names = [str(item.get("name") or item.get("element_id")) for item in rows if item.get("status") == status]
    return "\n".join(f"- {name}" for name in names) if names else "- none"


def _reference_names(selection: Dict[str, Any], key: str, fallback: Any = None) -> str:
    values = selection.get(key) if isinstance(selection, dict) else None
    fallback_ids = {str(item) for item in fallback} if key == "should_reference" and isinstance(fallback, list) else set()
    names: List[str] = []
    if isinstance(values, list):
        for item in values:
            if isinstance(item, dict):
                if fallback_ids and str(item.get("id") or "") not in fallback_ids:
                    continue
                name = item.get("name") or item.get("id")
            else:
                name = item
                if fallback_ids and str(name) not in fallback_ids:
                    continue
            if name:
                names.append(str(name))
    if not names and isinstance(fallback, list):
        names = [str(item) for item in fallback if item]
    return "\n".join(f"- {name}" for name in names) if names else "- none"


def _previous_completed_shot(context: ShotExecutionContext) -> Dict[str, Any]:
    if not context.previous_shots:
        raise BackendExecutionError("This generation mode requires a completed predecessor")
    previous = context.previous_shots[-1]
    if previous.get("state", {}).get("status") != "completed" or not previous.get("attempt"):
        raise BackendExecutionError("Previous shot is not completed")
    return previous


def _content_summary_for_storage(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for item in content:
        current = {key: value for key, value in item.items() if key not in {"image_url", "video_url"}}
        if item.get("type") == "text":
            current["text"] = item.get("text", "")
        if "image_url" in item:
            current["image_url"] = _image_url_summary_for_storage(item)
        if "video_url" in item:
            current["video_url"] = {"url": str(item["video_url"].get("url") or "")}
        summary.append(current)
    return summary


_SEEDANCE_CONTENT_INDEX_RE = re.compile(r"content\[(\d+)\]")


def _seedance_rejected_content_index(exc: Exception) -> int | None:
    text = str(exc)
    data = _json_object_from_error_text(text)
    if isinstance(data, dict):
        error = data.get("error") if isinstance(data.get("error"), dict) else {}
        haystack = "\n".join(
            str(value or "")
            for value in (error.get("param"), error.get("message"), text)
        )
    else:
        haystack = text
    match = _SEEDANCE_CONTENT_INDEX_RE.search(haystack)
    if not match:
        return None
    return int(match.group(1))


def _json_object_from_error_text(text: str) -> Dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        data = json.loads(text[start:])
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _sensitive_image_drop_record(
    content_index: int,
    item: Dict[str, Any],
    exc: Exception,
) -> Dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return {
        "content_index": content_index,
        "type": item.get("type"),
        "role": item.get("role"),
        "local_source": metadata.get("source_path") or metadata.get("file") or "",
        "metadata": metadata,
        "content_summary": _content_summary_for_storage([item])[0],
        "seedance_error": str(exc),
    }


def _image_url_summary_for_storage(item: Dict[str, Any]) -> Dict[str, Any]:
    image_url = item.get("image_url") if isinstance(item.get("image_url"), dict) else {}
    url = str(image_url.get("url") or "")
    if not url.startswith("data:"):
        return {"url": url}
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    local_source = metadata.get("source_path") or metadata.get("file") or "[embedded image data]"
    result = {
        "url": str(local_source),
        "embedded_data_url": True,
    }
    if url.startswith("data:") and ";" in url[:80]:
        result["mime"] = url[5 : url.index(";")]
    return result


def _media_debug_for_submission(content: List[Dict[str, Any]]) -> Dict[str, Any]:
    reference_videos: List[Dict[str, Any]] = []
    for index, item in enumerate(content, start=1):
        if item.get("type") != "video_url":
            continue
        video_url = item.get("video_url") if isinstance(item.get("video_url"), dict) else {}
        url = str(video_url.get("url") or "")
        reference_videos.append(
            {
                "index": index,
                "role": item.get("role"),
                "url": url,
                "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                "preflight": _preflight_media_url(url),
            }
        )
    return {"reference_videos": reference_videos}


def _preflight_media_url(url: str) -> Dict[str, Any]:
    if not url.startswith(("http://", "https://")):
        return {"status": "skipped", "reason": "not_http_url"}
    try:
        response = requests.get(
            url,
            allow_redirects=True,
            headers={"Range": "bytes=0-63", "User-Agent": "videogen-notebook-debug/1.0"},
            stream=True,
            timeout=MEDIA_PREFLIGHT_TIMEOUT_SECONDS,
        )
        first = b""
        try:
            for chunk in response.iter_content(chunk_size=64):
                if chunk:
                    first = chunk[:64]
                    break
        finally:
            response.close()
        content_type = response.headers.get("content-type", "")
        first_ascii = "".join(chr(value) if 32 <= value < 127 else "." for value in first[:32])
        return {
            "status": "ok",
            "status_code": response.status_code,
            "final_url": response.url,
            "content_type": content_type,
            "content_length": response.headers.get("content-length"),
            "history": [
                {
                    "status_code": item.status_code,
                    "url": item.url,
                    "location": item.headers.get("location"),
                }
                for item in response.history
            ],
            "first_bytes_hex": first[:32].hex(),
            "first_bytes_ascii": first_ascii,
            "is_probable_mp4": "video/mp4" in content_type.lower() or b"ftyp" in first[:32],
        }
    except Exception as exc:
        return {
            "status": "error",
            "type": type(exc).__name__,
            "message": str(exc),
        }


def _redact_urls(value: Any) -> Any:
    return value


def _fake_prompt(
    input_snapshot: Dict[str, Any],
    visual_status: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
) -> str:
    lines = [
        input_snapshot["video_prompt"],
        "",
        "Visual element guidance:",
    ]
    for element in visual_status:
        lines.append(f"- {element['name']}: {element['status']} ({element['reason']})")
    predefined = _predefined_references_from_input(input_snapshot)
    if predefined:
        lines.append("")
        lines.append("Predefined references:")
        for index, reference in enumerate(predefined, start=1):
            lines.append(f"- Image {index}: {reference.get('label')}; {reference.get('guidance')}")
    if references:
        lines.append("")
        lines.append("Historical references:")
        for index, reference in enumerate(references, start=1):
            image_index = len(predefined) + index
            lines.append(
                f"- Image {image_index}: source shot {reference['source_shot_id']}, "
                f"use {', '.join(reference['covered_elements'])}."
            )
    return "\n".join(lines)


def _log(step: str, message: str) -> Dict[str, Any]:
    return {"time": _now(), "step": step, "message": message}


def _algorithm_step_max_attempts(bundle: Dict[str, Any]) -> int:
    raw = (
        bundle.get("settings", {})
        .get("generation", {})
        .get("algorithm_step_max_attempts", DEFAULT_ALGORITHM_STEP_MAX_ATTEMPTS)
    )
    try:
        attempts = int(raw)
    except (TypeError, ValueError):
        attempts = DEFAULT_ALGORITHM_STEP_MAX_ATTEMPTS
    return min(max(attempts, 1), 10)


def _reference_selection_disabled(bundle: Dict[str, Any]) -> bool:
    return _reference_selection_mode(bundle) == "none"


def _reference_selection_skips_visual_plan(bundle: Dict[str, Any]) -> bool:
    return _reference_selection_mode(bundle) in {NAIVE_TOP_K_SELECTION_MODE, STORYMEM_MEMORY_SELECTION_MODE}


def _reference_selection_naive(bundle: Dict[str, Any]) -> bool:
    return _reference_selection_mode(bundle) == NAIVE_TOP_K_SELECTION_MODE


def _reference_selection_storymem(bundle: Dict[str, Any]) -> bool:
    return _reference_selection_mode(bundle) == STORYMEM_MEMORY_SELECTION_MODE


def _reference_selection_mode(bundle: Dict[str, Any]) -> str:
    return str(
        bundle.get("settings", {})
        .get("visual_element_memory", {})
        .get("selection_mode", "greedy_coverage")
    )


def _retry_attempt_record(attempt_index: int, exc: Exception) -> Dict[str, Any]:
    details = getattr(exc, "details", None)
    return {
        "attempt": attempt_index,
        "type": type(exc).__name__,
        "message": str(exc),
        "details": details if isinstance(details, dict) else None,
    }


def _attach_step_retry_details(exc: Exception, retry_attempts: List[Dict[str, Any]]) -> None:
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        details = {}
    details = {**details, "step_retry_attempts": retry_attempts}
    try:
        setattr(exc, "details", details)
    except Exception:
        pass


def _attach_keyframe_retry_details(exc: Exception, retry_attempts: List[Dict[str, Any]]) -> None:
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        details = {}
    details = {**details, "keyframe_maintaining_full_attempts": retry_attempts}
    try:
        setattr(exc, "details", details)
    except Exception:
        pass


def _step_retry_label(step: str) -> str:
    return {
        "visual_plan": "Visual Elements Plan",
        "reference_selection": "Historical Reference Selection",
        "seedance_generation": "Seedance Video Generation",
        "keyframe_maintaining": "Keyframe Maintaining",
    }.get(step, step.replace("_", " ").title())


def _terminal_log_step_for_step(step: str, status: str) -> str:
    if step == "visual_plan":
        return "planning_visual_elements"
    if step == "reference_selection":
        return "selecting_visual_references"
    if step == "seedance_prompt":
        return "composing_prompt"
    if step == "keyframe_maintaining":
        return "annotating_visual_memory"
    if step == "seedance_generation":
        return "submitting"
    return status


def _attempt_diagnostic_logs(details: Dict[str, Any], step: str) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    last_error = details.get("last_error")
    if last_error:
        logs.append(_log(step, f"Failure point: {last_error}"))
    for attempts_key in ("step_retry_attempts", "keyframe_maintaining_full_attempts"):
        attempts = details.get(attempts_key)
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            message = (
                f"Full step attempt {attempt.get('attempt', '?')} failed: "
                f"{attempt.get('type', 'Error')}: {attempt.get('message', '')}"
            )
            logs.append(_log(step, message))
    for label, attempts_key in (("LLM", "llm_attempts"), ("VLM", "vlm_attempts")):
        attempts = details.get(attempts_key)
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            attempt_number = attempt.get("attempt", "?")
            prompt = _attempt_prompt(attempt)
            if prompt:
                logs.append(_log(step, f"{label} attempt {attempt_number} submitted prompt:\n{prompt}"))
            raw_response = attempt.get("raw_response")
            if raw_response is not None:
                logs.append(_log(step, f"{label} attempt {attempt_number} raw response:\n{raw_response}"))
            parse_error = attempt.get("parse_error")
            if parse_error:
                logs.append(_log(step, f"{label} attempt {attempt_number} parse error:\n{parse_error}"))
    return logs


def _attempt_prompt(attempt: Dict[str, Any]) -> str:
    prompt = attempt.get("prompt")
    if isinstance(prompt, str):
        return prompt
    metadata = attempt.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("prompt"), str):
        return str(metadata["prompt"])
    return ""


def _short_name(prompt: str) -> str:
    clean = " ".join(prompt.split())
    return clean[:80] + ("..." if len(clean) > 80 else "")


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _prefix_video_assets(shots: List[Dict[str, Any]]) -> Dict[str, str]:
    assets: Dict[str, str] = {}
    for shot in sorted(shots, key=lambda item: item.get("order_index", 0)):
        if shot.get("state", {}).get("status") != "completed":
            break
        attempt = shot.get("attempt") or {}
        asset_id = attempt.get("outputs", {}).get("raw_video_asset_id")
        if asset_id:
            assets[shot["shot_id"]] = asset_id
    return assets


def _thumbnail_from_memory(produced_memory: List[Dict[str, Any]]) -> str | None:
    for memory in produced_memory:
        asset_id = memory.get("asset_id")
        if asset_id:
            return str(asset_id)
    return None


def _assembly_clips(
    store: ProjectStore,
    project_id: str,
    shots: List[Dict[str, Any]],
    assets: Dict[str, Any],
) -> List[Dict[str, Any]]:
    clips: List[Dict[str, Any]] = []
    for shot in shots:
        attempt = shot.get("attempt") or {}
        asset_id = attempt.get("outputs", {}).get("raw_video_asset_id")
        if not asset_id:
            raise BackendExecutionError(f"Completed shot {shot['shot_id']} has no raw video asset")
        path = store.asset_path(project_id, str(asset_id))
        asset = assets.get(str(asset_id), {})
        clips.append(
            {
                "shot_id": shot["shot_id"],
                "attempt_id": attempt.get("attempt_id"),
                "attempt_dir": str(store.attempt_dir(project_id, shot["shot_id"], str(attempt.get("attempt_id")))),
                "output_video": str(path),
                "raw_video_asset_id": str(asset_id),
                "generation_mode": shot.get("inputs", {}).get("generation_mode", "default"),
                "metadata": asset.get("metadata") or {},
            }
        )
    return clips


def _contains_placeholder_clip(clips: List[Dict[str, Any]]) -> bool:
    for clip in clips:
        metadata = clip.get("metadata") or {}
        if metadata.get("fake") or metadata.get("dry_run"):
            return True
        if not _looks_like_mp4(Path(str(clip.get("output_video") or ""))):
            return True
    return False


def _write_placeholder_final(path: Path, clips: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["videogen_notebook placeholder final assembly"]
    for clip in clips:
        lines.append(f"{clip['shot_id']} {clip['raw_video_asset_id']} {clip['output_video']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _concat_videos_ffmpeg(paths: List[Path], output_path: Path) -> None:
    if not paths:
        raise BackendExecutionError("No videos available for final assembly")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        shutil.copyfile(paths[0], output_path)
        return
    ffmpeg = _ffmpeg_executable()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        list_path = Path(handle.name)
        for path in paths:
            handle.write(f"file '{_ffmpeg_concat_escape(path)}'\n")
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        temporary.replace(output_path)
    except subprocess.CalledProcessError as exc:
        raise BackendExecutionError(f"Final video concatenation failed: {exc}") from exc
    finally:
        list_path.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def _ffmpeg_executable() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ModuleNotFoundError as exc:
        raise BackendExecutionError("ffmpeg executable is required for final assembly") from exc


def _ffmpeg_concat_escape(path: Path) -> str:
    return str(path).replace("'", "'\\''")


def _looks_like_mp4(path: Path) -> bool:
    try:
        header = path.read_bytes()[:32]
    except OSError:
        return False
    return b"ftyp" in header


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import json

        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
