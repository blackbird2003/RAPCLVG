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
    DEFAULT_MODEL as DEFAULT_SEEDANCE_MODEL,
    SeedanceClient,
    SeedanceError,
    extract_last_frame_url,
    extract_task_id,
    extract_video_url,
)
from .adapters import PlaceholderVideoPostprocessor, PlaceholderVisualMemoryPlanner
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
    _content_summary_for_storage,
    _log,
    _maybe_force_animation_prompt,
    _media_debug_for_submission,
    _predefined_media_summary_items,
    _predefined_non_image_references,
    _predefined_references_from_input,
    _previous_completed_shot,
    _relative,
    _seedance_content,
    _seedance_rejected_content_index,
    _sensitive_image_drop_record,
    _shot_spec_from_bundle_shot,
    _smooth_reference_seconds,
    _story_script,
    _uses_default_last_frame_continuity,
    _uses_smooth_reference_video,
)


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
            "ratio": (
                "adaptive"
                if _uses_smooth_reference_video(context.shot, context.input_snapshot)
                else settings.get("seedance", {}).get("ratio")
            ),
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
            "ratio": (
                "adaptive"
                if _uses_smooth_reference_video(context.shot, context.input_snapshot)
                else settings.get("seedance", {}).get("ratio")
            ),
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
        content.extend(_predefined_media_summary_items(context.input_snapshot))
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
        content.extend(self._predefined_media_items(context))
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

    def _predefined_media_items(self, context: ShotExecutionContext) -> List[Dict[str, Any]]:
        references = _predefined_non_image_references(context.input_snapshot)
        if not references:
            return []
        from ...media.reference_video import configured_reference_video_publisher, require_public_https_url

        publisher = configured_reference_video_publisher()
        items: List[Dict[str, Any]] = []
        for reference in references:
            asset_id = str(reference.get("asset_id") or "")
            if asset_id:
                source_path = self.store.asset_path(context.project_id, asset_id)
            else:
                source_path = Path(str(reference.get("source_path") or ""))
                if not source_path.is_absolute():
                    source_path = self.store.project_dir(context.project_id) / source_path
            if not source_path.is_file():
                raise BackendExecutionError(f"Predefined {reference['media_type']} file is missing: {source_path}")
            public_url = require_public_https_url(publisher.publish(source_path))
            metadata = dict(reference)
            metadata.update(
                {
                    "input_origin": "published_predefined_media",
                    "source_path": str(source_path),
                    "file": source_path.name,
                }
            )
            if reference["media_type"] == "video":
                items.append(video_item_with_metadata(public_url, metadata))
            else:
                items.append(audio_item_with_metadata(public_url, metadata))
        return items

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
        from ...media.reference_video import (
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
