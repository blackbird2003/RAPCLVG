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

from ..generators.seedance_client import DEFAULT_MODEL as DEFAULT_SEEDANCE_MODEL
from ..generators.seedance_client import (
    SeedanceClient,
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

from .domain import MAX_REFERENCE_IMAGES, MIN_SMOOTH_REFERENCE_SECONDS, SCHEMA_VERSION
from .domain import STEP_SEQUENCE
from .gpu_lock import GpuLockCancelled, KeyframeGpuLock
from .json_store import read_json, write_json_atomic, write_jsonl_atomic
from .locks import GlobalRunLock
from .project_store import InvalidProjectError, ProjectStore, _completed_prefix, _now
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
from .runner_helpers import *
from .backends import (
    FakeExecutionBackend,
    VisualElementMemoryAdapter,
    RealExecutionBackend,
)
from .step_flow import StepFlowMixin
from .attempts import AttemptLifecycleMixin
from .assembly import AssemblyMixin


def create_runner(
    store: ProjectStore,
    backend_name: str = "fake",
    *,
    real_client: SeedanceLikeClient | None = None,
    real_visual_planner: VisualMemoryPlanner | None = None,
    real_postprocessor: VideoPostprocessor | None = None,
    real_submit: bool | None = None,
) -> "VideoGenRunner":
    normalized = (backend_name or "fake").strip().lower()
    if normalized == "fake":
        return VideoGenRunner(store, FakeExecutionBackend(store))
    if normalized == "real":
        submit_enabled = (
            _env_flag("VIDEOGEN_REAL_SUBMIT", default=True)
            if real_submit is None
            else bool(real_submit)
        )
        coordinator = (
            VisualElementMemoryAdapter(store)
            if submit_enabled and (real_visual_planner is None or real_postprocessor is None)
            else None
        )
        return VideoGenRunner(
            store,
            RealExecutionBackend(
                store,
                client=real_client,
                visual_planner=real_visual_planner or coordinator,
                postprocessor=real_postprocessor or coordinator,
                dry_run=not submit_enabled,
            ),
        )
    raise ValueError(f"Unknown video generation runner backend: {backend_name}")


class VideoGenRunner(StepFlowMixin, AttemptLifecycleMixin, AssemblyMixin):
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



class FakeVideoGenRunner(VideoGenRunner):
    """Compatibility wrapper that keeps the low-cost fake runner as default."""

    def __init__(self, store: ProjectStore) -> None:
        super().__init__(store, FakeExecutionBackend(store))
