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
    _add_reference_guidance,
    _dry_run_references,
    _dry_run_visual_status,
    _effective_reference_budget,
    _existing_memory_paths,
    _image_asset_record,
    _log,
    _naive_produced_memory_from_keyframes,
    _placeholder_produced_memory,
    _plan_result_from_details,
    _produced_memory_from_annotations,
    _prompt_for_generation_mode,
    _reference_selection_mode,
    _reference_selection_sink_recent,
    _reference_selection_skips_visual_plan,
    _relative,
    _run_config_for_bundle,
    _shot_spec_from_bundle_shot,
    _sink_recent_existing_memory_paths,
    _story_script,
    _uses_default_last_frame_continuity,
    _visual_memory_snapshot_from_shots,
    _visual_reference_prompt_context,
    _visual_reference_rows,
    _visual_status_rows,
)


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


class VisualElementMemoryAdapter:
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
        from ...generators.seedance_executor import extract_shot_memory

        bundle = context.bundle
        settings = bundle["settings"]
        project_dir = self.store.project_dir(context.project_id)
        key = (context.project_id, context.shot["shot_id"], context.attempt_id)
        memory = self._planned.pop(key, None) or self._memory_for_current_postprocess(context)
        sink_recent_mode = _reference_selection_sink_recent(bundle)
        existing_memory_paths = (
            _sink_recent_existing_memory_paths(self.store, context)
            if sink_recent_mode
            else _existing_memory_paths(
                self.store,
                context.project_id,
                bundle["shots"],
                context.shot["shot_id"],
            )
        )
        profile = (
            SINK_RECENT_KEYFRAME_PROFILE
            if sink_recent_mode
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
                    "selected_sink_recent_memory_bank" if sink_recent_mode else "all_previous_memory"
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
        from pipeline.agentic.visual_element_memory import VisualElementMemory

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
        from ...generators.seedance_executor import extract_shot_memory

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
