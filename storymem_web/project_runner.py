from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from memory_query_llm import MemoryQueryGenerator
from seedance_client import SeedanceCancelledError, SeedanceClient, extract_last_frame_url
from storymem_seedance.artifacts import concat_video_paths
from storymem_seedance.executor import complete_shot, extract_shot_memory, prepare_shot, submit_shot
from storymem_seedance.models import RunConfig, ShotSpec
from storymem_seedance.prompting import (
    SMOOTH_CONTINUATION_INSTRUCTION,
    image_item_with_metadata,
    video_item_with_metadata,
)
from storymem_seedance.visual_element_memory import VisualElementMemory

from .reference_video import (
    ReferenceVideoPublisher,
    configured_reference_video_publisher,
    extract_video_tail,
    require_public_https_url,
)
from .repository import InvalidStateError, ProjectRepository
from .settings import build_run_config, merged_generation_config


MemorySelector = Callable[..., List[Dict[str, Any]]]
MemoryExtractor = Callable[..., Dict[str, Any]]
ConcatFunction = Callable[[Iterable[str], str], Optional[str]]
TailExtractor = Callable[..., str]
SmoothAssembler = Callable[[List[Dict[str, Any]], str], str]
VisualMemoryFactory = Callable[[RunConfig, Dict[str, Any], List[ShotSpec]], VisualElementMemory]


class _MemoryStoreView:
    def __init__(self, paths: Iterable[str]) -> None:
        self._paths = list(paths)

    def memory_keyframes(self) -> List[str]:
        return list(self._paths)


class ProjectRunner:
    def __init__(
        self,
        repository: ProjectRepository,
        job_id: str,
        *,
        client: Optional[SeedanceClient] = None,
        memory_selector: Optional[MemorySelector] = None,
        memory_extractor: Optional[MemoryExtractor] = None,
        concat_function: Optional[ConcatFunction] = None,
        reference_video_publisher: Optional[ReferenceVideoPublisher] = None,
        tail_extractor: Optional[TailExtractor] = None,
        smooth_assembler: Optional[SmoothAssembler] = None,
        visual_memory_factory: Optional[VisualMemoryFactory] = None,
    ) -> None:
        self.repository = repository
        self.job_id = job_id
        self.client = client
        self.memory_selector = memory_selector
        self.memory_extractor = memory_extractor or extract_shot_memory
        self.concat_function = concat_function or concat_video_paths
        self.reference_video_publisher = reference_video_publisher
        self.tail_extractor = tail_extractor or extract_video_tail
        self.smooth_assembler = smooth_assembler
        self.visual_memory_factory = visual_memory_factory or VisualElementMemory
        self._memory_query_generator: Optional[MemoryQueryGenerator] = None
        self._active_attempt_id: Optional[str] = None

    def run(self) -> Dict[str, Any]:
        job = self.repository.get_job(self.job_id)
        project_id = job["project_id"]
        self.repository.update_job(self.job_id, status="running", pid=os.getpid())
        self.repository.set_project_state(
            project_id,
            "running",
            run_mode=job["mode"],
            active_shot_id=job.get("shot_id"),
        )
        try:
            if job["mode"] == "keyframes":
                if not job.get("shot_id"):
                    raise InvalidStateError("Keyframe retry requires a shot")
                self._retry_keyframes(project_id, job["shot_id"])
            elif job["mode"] == "smooth_postprocess":
                if not job.get("shot_id"):
                    raise InvalidStateError("Smooth retry requires a shot")
                self._retry_smoothing(project_id, job["shot_id"])
            else:
                shots = self._shots_for_job(job)
                for shot in shots:
                    if self.repository.job_cancel_requested(self.job_id):
                        raise SeedanceCancelledError("Project execution was interrupted")
                    self._execute_shot(project_id, shot["shot_id"])
            project = self.repository.get_project(project_id)
            completed = all(shot["state"] == "completed" for shot in project["shots"])
            final_status = "completed" if completed else "partial"
            self.repository.update_job(self.job_id, status="completed")
            return self.repository.set_project_state(
                project_id,
                final_status,
                active_shot_id=None,
                run_mode=None,
            )
        except SeedanceCancelledError as exc:
            if self._active_attempt_id:
                self.repository.update_attempt(
                    self._active_attempt_id,
                    status="interrupted",
                    error={"type": type(exc).__name__, "message": str(exc)},
                    finished=True,
                )
            self.repository.update_job(self.job_id, status="interrupted")
            return self.repository.set_project_state(
                project_id,
                "interrupted",
                active_shot_id=None,
                run_mode=None,
            )
        except Exception as exc:
            logging.exception("Project runner failed for job %s", self.job_id)
            if self._active_attempt_id:
                self.repository.update_attempt(
                    self._active_attempt_id,
                    status="failed",
                    error={"type": type(exc).__name__, "message": str(exc)},
                    finished=True,
                )
            self.repository.update_job(self.job_id, status="failed")
            self.repository.set_project_state(
                project_id,
                "failed",
                active_shot_id=None,
                run_mode=None,
            )
            raise

    def _shots_for_job(self, job: Dict[str, Any]) -> List[Dict[str, Any]]:
        project = self.repository.get_project(job["project_id"])
        shots = project["shots"]
        if job["mode"] == "single":
            selected = next((shot for shot in shots if shot["shot_id"] == job["shot_id"]), None)
            if selected is None:
                raise InvalidStateError("Selected shot does not belong to this project")
            predecessors = shots[: selected["order_index"]]
            if any(shot["state"] != "completed" for shot in predecessors):
                raise InvalidStateError("All preceding shots must be completed first")
            if selected["state"] == "completed":
                raise InvalidStateError("Reset the completed shot before rerunning it")
            return [selected]
        if job["mode"] != "all":
            raise InvalidStateError(f"Unknown job mode: {job['mode']}")
        first_incomplete = next(
            (index for index, shot in enumerate(shots) if shot["state"] != "completed"),
            len(shots),
        )
        return shots[first_incomplete:]

    def _execute_shot(self, project_id: str, shot_id: str) -> str:
        project = self.repository.get_project(project_id)
        shot = next(item for item in project["shots"] if item["shot_id"] == shot_id)
        story_script = _story_from_project(project)
        shot_spec = _shot_spec(shot, story_script)
        generation_config = merged_generation_config(project["generation_config"])
        generation_config["duration"] = shot["duration_seconds"]
        memory_policy = _memory_policy(shot)
        preliminary_snapshot = {
            "video_prompt": shot["video_prompt"],
            "is_cut": shot["is_cut"],
            "generation_mode": shot["generation_mode"],
            "duration_seconds": shot["duration_seconds"],
            "memory_policy": memory_policy,
            "generation_config": generation_config,
            "references": [],
        }
        attempt = self.repository.create_attempt(shot_id, preliminary_snapshot)
        attempt_id = attempt["attempt_id"]
        self._active_attempt_id = attempt_id
        attempt_dir = self.repository.attempt_dir(project_id, shot_id, attempt_id)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        config = build_run_config(generation_config, str(attempt_dir))
        client = self._client(config)
        self.repository.set_project_state(project_id, "running", active_shot_id=shot_id)

        references, memory_selection = self._prepare_references(
            project=project,
            shot=shot,
            shot_spec=shot_spec,
            story_script=story_script,
            config=config,
            attempt_dir=attempt_dir,
        )
        if shot["generation_mode"] == "smooth":
            references = self._prepare_smooth_reference(
                project=project,
                shot=shot,
                attempt_dir=attempt_dir,
                references=references,
            )
        elif not shot["is_cut"] and _has_previous_ending_reference(references):
            previous_last_frame_url = self._previous_seedance_last_frame_url(
                project, shot, client
            )
            references = _use_original_previous_ending_url(
                references, previous_last_frame_url
            )
        if shot["generation_mode"] == "last_frame_only":
            references = _use_previous_ending_as_first_frame(references)
        prompt_override = memory_selection.get("prompt_override")
        if _pipeline_version(project) == "classic" and shot["generation_mode"] == "default":
            prompt_override = _classic_seedance_prompt(shot, references)
        if prompt_override and shot["generation_mode"] == "smooth":
            prompt_override = (
                f"{SMOOTH_CONTINUATION_INSTRUCTION}\n\n"
                "Input media numbering for this Smooth shot:\n"
                "- Video 1 是上一段原始视频的尾部片段。当前 shot 必须向后延长 Video 1，从 Video 1 最后一帧之后的下一时刻自然继续。\n"
                "- Static reference images are numbered separately as Image 1, Image 2, ... after Video 1. "
                "When the prompt below says Image 1, it means the first static image reference, not Video 1.\n\n"
                f"{prompt_override}"
            )
        prepared = prepare_shot(
            config=config,
            story_script=story_script,
            shot=shot_spec,
            references=references,
            is_first=shot["order_index"] == 0,
            generation_mode=shot["generation_mode"],
            prompt_override=prompt_override,
        )
        memory_selection = {
            key: value
            for key, value in memory_selection.items()
            if key != "prompt_override"
        }
        input_snapshot = {
            "video_prompt": shot["video_prompt"],
            "is_cut": shot["is_cut"],
            "generation_mode": shot["generation_mode"],
            "duration_seconds": shot["duration_seconds"],
            "memory_policy": memory_policy,
            "generation_config": generation_config,
            "references": prepared.reference_metadata,
        }
        self.repository.update_attempt(
            attempt_id,
            input_snapshot=input_snapshot,
            memory_selection=memory_selection,
            submitted_prompt=prepared.submitted_prompt,
        )
        self.repository.replace_references(attempt_id, prepared.reference_metadata)
        _atomic_json(attempt_dir / "input.json", input_snapshot | {"submitted_prompt": prepared.submitted_prompt})
        _atomic_json(attempt_dir / "memory.json", memory_selection)

        self.repository.update_attempt(attempt_id, status="submitting")
        submission = submit_shot(config=config, client=client, prepared=prepared)
        self.repository.update_attempt(attempt_id, task_id=submission.task_id)
        _atomic_json(
            attempt_dir / "seedance_task.json",
            {"create_response": _redact_response(submission.response)},
        )

        output_video = str(attempt_dir / f"{shot['scene_num']:02d}_{shot['shot_num']:02d}.mp4")
        completed = complete_shot(
            config=config,
            client=client,
            task_id=submission.task_id,
            output_video=output_video,
            should_cancel=lambda: self.repository.job_cancel_requested(self.job_id),
            on_status=lambda status: self.repository.update_attempt(attempt_id, status=status),
        )
        usage = _usage_from_response(completed.response)
        self.repository.update_attempt(
            attempt_id,
            usage=usage,
            output_video=output_video,
        )
        _atomic_json(
            attempt_dir / "seedance_task.json",
            {"create_response": submission.response, "result_response": _redact_response(completed.response)},
        )

        if not config.skip_keyframes:
            produced = self._extract_and_register_memory(
                project_id,
                shot_id,
                attempt_id,
                output_video,
                config=config,
            )
            if _pipeline_version(project) == "visual_element_v1":
                self._annotate_visual_memory(
                    project=project,
                    shot=shot,
                    shot_spec=shot_spec,
                    story_script=story_script,
                    config=config,
                    attempt_id=attempt_id,
                    keyframe_paths=produced.get("keyframe_paths") or [],
                )
        elif _pipeline_version(project) == "visual_element_v1":
            self._finalize_visual_memory_without_annotations(attempt_id)

        if shot["generation_mode"] == "smooth":
            self.repository.update_attempt(attempt_id, status="smoothing_transition")
        self.repository.update_attempt(attempt_id, status="completed", finished=True)
        self._update_final_video(project_id)
        self._active_attempt_id = None
        return attempt_id

    def _retry_smoothing(self, project_id: str, shot_id: str) -> str:
        shot = self.repository.get_shot(shot_id)
        if shot["project_id"] != project_id or not shot.get("current_attempt_id"):
            raise InvalidStateError("Shot has no current Smooth attempt to repair")
        attempt_id = shot["current_attempt_id"]
        attempt = self.repository.get_attempt(attempt_id)
        if attempt["status"] not in {"failed", "interrupted"}:
            raise InvalidStateError("Only failed Smooth post-processing can be retried")
        if (attempt.get("error") or {}).get("type") != "SmoothTransitionError":
            raise InvalidStateError("Attempt did not fail during Smooth post-processing")
        if not attempt.get("output_video") or not Path(attempt["output_video"]).is_file():
            raise InvalidStateError("Attempt has no raw video for Smooth post-processing")

        self._active_attempt_id = attempt_id
        self.repository.set_project_state(project_id, "running", active_shot_id=shot_id)
        self.repository.update_attempt(
            attempt_id, status="smoothing_transition", error={}
        )
        self.repository.update_attempt(
            attempt_id, status="completed", error={}, finished=True
        )
        self._update_final_video(project_id)
        self._active_attempt_id = None
        return attempt_id

    def _retry_keyframes(self, project_id: str, shot_id: str) -> str:
        project = self.repository.get_project(project_id)
        shot = next(
            (item for item in project["shots"] if item["shot_id"] == shot_id),
            None,
        )
        if shot is None:
            raise InvalidStateError("Selected shot does not belong to this project")
        attempt_id = shot.get("current_attempt_id")
        if not attempt_id:
            raise InvalidStateError("Shot has no current attempt to repair")
        attempt = self.repository.get_attempt(attempt_id)
        if attempt["status"] not in {"failed", "interrupted"}:
            raise InvalidStateError("Only failed or interrupted attempts can retry keyframes")
        output_video = attempt.get("output_video")
        if not output_video or not Path(output_video).is_file():
            raise InvalidStateError("Attempt has no generated video to post-process")
        generation_config = merged_generation_config(project["generation_config"])
        generation_config["duration"] = shot["duration_seconds"]
        config = build_run_config(
            generation_config,
            str(self.repository.attempt_dir(project_id, shot_id, attempt_id)),
        )

        self._active_attempt_id = attempt_id
        self.repository.set_project_state(project_id, "running", active_shot_id=shot_id)
        self.repository.update_attempt(
            attempt_id,
            status="extracting_keyframes",
            error={},
        )
        produced = self._extract_and_register_memory(
            project_id,
            shot_id,
            attempt_id,
            output_video,
            config=config,
        )
        if _pipeline_version(project) == "visual_element_v1":
            story_script = _story_from_project(project)
            shot_spec = _shot_spec(shot, story_script)
            self._annotate_visual_memory(
                project=project,
                shot=shot,
                shot_spec=shot_spec,
                story_script=story_script,
                config=config,
                attempt_id=attempt_id,
                keyframe_paths=produced.get("keyframe_paths") or [],
            )
        if self.repository.job_cancel_requested(self.job_id):
            raise SeedanceCancelledError("Project execution was interrupted")
        self.repository.update_attempt(
            attempt_id,
            status="completed",
            error={},
            finished=True,
        )
        self._update_final_video(project_id)
        self._active_attempt_id = None
        return attempt_id

    def _extract_and_register_memory(
        self,
        project_id: str,
        shot_id: str,
        attempt_id: str,
        output_video: str,
        config: RunConfig,
    ) -> Dict[str, Any]:
        self.repository.update_attempt(attempt_id, status="extracting_keyframes")
        existing_paths = [
            asset["source_path"]
            for asset in self.repository.list_memory_assets(
                project_id,
                asset_type="retrieval_keyframe",
            )
            if asset["source_shot_id"] != shot_id
        ]
        produced = self.memory_extractor(
            output_video,
            existing_memory_paths=existing_paths,
            keyframe_profile=config.keyframe_profile,
            keyframe_config_path=config.keyframe_config_path,
        )
        assets = _memory_assets_from_result(produced)
        self.repository.add_memory_assets(project_id, shot_id, attempt_id, assets)
        _atomic_json(
            self.repository.attempt_dir(project_id, shot_id, attempt_id)
            / "memory_assets.json",
            assets,
        )
        return produced

    def _prepare_references(
        self,
        *,
        project: Dict[str, Any],
        shot: Dict[str, Any],
        shot_spec: ShotSpec,
        story_script: Dict[str, Any],
        config: RunConfig,
        attempt_dir: Path,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if _pipeline_version(project) == "visual_element_v1":
            return self._prepare_visual_element_references(
                project=project,
                shot=shot,
                shot_spec=shot_spec,
                story_script=story_script,
                config=config,
            )
        if shot["order_index"] == 0:
            return [], {"policy": "first_shot", "references": []}
        active_assets = self.repository.list_memory_assets(
            project["project_id"], asset_type="retrieval_keyframe"
        )
        paths = [asset["source_path"] for asset in active_assets]
        asset_by_path = {asset["source_path"]: asset for asset in active_assets}
        selector = self.memory_selector or _default_memory_selector
        memory_records = selector(
            config=config,
            store=_MemoryStoreView(paths),
            story_script=story_script,
            shot=shot_spec,
            memory_query_generator=self._query_generator(project, config),
            memory_policy=_memory_policy(shot),
        )
        memory_records = _filter_memory_records(
            memory_records, _memory_policy(shot)
        )
        metadata = []
        if not shot["is_cut"]:
            previous = project["shots"][shot["order_index"] - 1]
            ending_assets = self.repository.list_memory_assets(
                project["project_id"],
                shot_id=previous["shot_id"],
                asset_type="ending_frame",
            )
            if ending_assets:
                ending = ending_assets[-1]
                metadata.append(
                    {
                        "label": "previous_last_frame",
                        "roles": ["previous_last_frame"],
                        "source_type": "auto",
                        "source_path": ending["source_path"],
                        "file": os.path.basename(ending["source_path"]),
                        "source_shot_id": previous["shot_id"],
                        "source_scene_num": previous["scene_num"],
                        "source_shot_num": previous["shot_num"],
                        "source_prompt": previous["video_prompt"],
                    }
                )
        available_memory_slots = max(0, config.max_reference_images - len(metadata))
        memory_records = _fit_storymem_memory_to_slots(
            memory_records, available_memory_slots
        )
        for record in memory_records:
            if len(metadata) >= config.max_reference_images:
                break
            item = dict(record)
            asset = asset_by_path.get(item.get("source_path"), {})
            item["source_type"] = "auto"
            item["source_shot_id"] = asset.get("source_shot_id")
            metadata.append(item)
        for index, item in enumerate(metadata, start=1):
            item["reference_index"] = index
        references = [image_item_with_metadata(item["source_path"], item) for item in metadata]
        selection = _read_last_jsonl(attempt_dir / "prompt_retrieval_log.jsonl") or {
            "policy": "default_memory",
            "final_memory": [os.path.basename(item["source_path"]) for item in metadata],
        }
        selection["references"] = metadata
        selection["memory_policy"] = _memory_policy(shot)
        submitted_counts = _memory_role_counts(metadata)
        if selection.get("counts"):
            selection["selection_counts"] = dict(selection["counts"])
        selection["counts"] = {
            **{
                key: value
                for key, value in (selection.get("counts") or {}).items()
                if key not in {"sink", "recent", "selected", "final_memory"}
            },
            "sink": submitted_counts["sink"],
            "retrieved": submitted_counts["retrieve"],
            "recent": submitted_counts["recent"],
            "final_memory": len(
                {
                    item["source_path"]
                    for item in metadata
                    if set(item.get("roles") or [])
                    & {
                        "early_sink_memory",
                        "prompt_retrieved_memory",
                        "recent_window_memory",
                    }
                }
            ),
        }
        return references, selection

    def _prepare_visual_element_references(
        self,
        *,
        project: Dict[str, Any],
        shot: Dict[str, Any],
        shot_spec: ShotSpec,
        story_script: Dict[str, Any],
        config: RunConfig,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        self.repository.update_attempt(
            self._active_attempt_id,
            status="planning_visual_elements",
        )
        visual_config = self._visual_config_for_shot(config, shot)
        visual_memory = self._visual_memory_before_shot(
            project=project,
            story_script=story_script,
            config=visual_config,
            shot_index=shot["order_index"],
        )
        selection = visual_memory.prepare_references_for_shot(
            shot["order_index"],
            shot_spec,
        )
        self.repository.update_attempt(
            self._active_attempt_id,
            status="selecting_visual_references",
        )
        visual_refs = list(selection.selected_references)
        visual_metadata = [item.get("metadata", {}) for item in visual_refs]
        continuity_refs = self._continuity_references(project, shot)
        references = [*continuity_refs, *visual_refs]
        memory_selection = {
            "policy": "visual_element_v1",
            "references": [item.get("metadata", {}) for item in references],
            "visual_element": {
                "settings": {
                    "selection_mode": visual_config.visual_element_selection_mode,
                    "sink_frame_count": visual_config.visual_element_sink_frame_count,
                    "max_retrieved_frames": visual_config.visual_element_max_retrieved_frames,
                    "effective_static_reference_budget": visual_config.max_reference_images
                    - len(continuity_refs),
                },
                "plan": selection.report_record.get("visual_element_plan", {}),
                "decision": selection.report_record.get("visual_element_decision", {}),
                "inserted": selection.report_record.get("visual_element_inserted", []),
                "selected_references": visual_metadata,
                "retrieved_references": selection.report_record.get(
                    "visual_element_retrieved_references",
                    [],
                ),
                "sink_references": selection.report_record.get(
                    "visual_element_sink_references",
                    [],
                ),
                "record": visual_memory.records[-1] if visual_memory.records else {},
                "state_before": self._visual_snapshot_before(project, shot["order_index"]),
                "state_after_planning": visual_memory.snapshot(),
            },
            "prompt_override": selection.prompt_context,
        }
        return references, memory_selection

    def _visual_memory_before_shot(
        self,
        *,
        project: Dict[str, Any],
        story_script: Dict[str, Any],
        config: RunConfig,
        shot_index: int,
    ) -> VisualElementMemory:
        all_specs = [_shot_spec(item, story_script) for item in project["shots"]]
        memory = self.visual_memory_factory(config, story_script, all_specs)
        snapshot = self._visual_snapshot_before(project, shot_index)
        if snapshot:
            memory.load_snapshot(snapshot)
        return memory

    def _visual_snapshot_before(
        self,
        project: Dict[str, Any],
        shot_index: int,
    ) -> Dict[str, Any]:
        for predecessor in reversed(project["shots"][:shot_index]):
            if predecessor["state"] != "completed" or not predecessor.get("current_attempt_id"):
                continue
            attempt = self.repository.get_attempt(predecessor["current_attempt_id"])
            visual = (attempt.get("memory_selection") or {}).get("visual_element") or {}
            snapshot = visual.get("state_after")
            if snapshot:
                return snapshot
        return {}

    def _visual_config_for_shot(self, config: RunConfig, shot: Dict[str, Any]) -> RunConfig:
        continuity_image_count = (
            1
            if shot["order_index"] > 0
            and not shot["is_cut"]
            and shot["generation_mode"] != "smooth"
            else 0
        )
        available = max(0, int(config.max_reference_images) - continuity_image_count)
        sink_count = min(int(config.visual_element_sink_frame_count), available)
        max_retrieved = min(
            int(config.visual_element_max_retrieved_frames),
            max(0, available - sink_count),
        )
        return replace(
            config,
            visual_element_memory=True,
            visual_element_sink_frame_count=sink_count,
            visual_element_max_retrieved_frames=max_retrieved,
        )

    def _continuity_references(
        self,
        project: Dict[str, Any],
        shot: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if shot["order_index"] == 0 or shot["is_cut"]:
            return []
        previous = project["shots"][shot["order_index"] - 1]
        ending_assets = self.repository.list_memory_assets(
            project["project_id"],
            shot_id=previous["shot_id"],
            asset_type="ending_frame",
        )
        if not ending_assets:
            return []
        ending = ending_assets[-1]
        metadata = {
            "label": "previous_last_frame",
            "roles": ["previous_last_frame"],
            "source_type": "auto",
            "source_path": ending["source_path"],
            "file": os.path.basename(ending["source_path"]),
            "source_shot_id": previous["shot_id"],
            "source_scene_num": previous["scene_num"],
            "source_shot_num": previous["shot_num"],
            "source_prompt": previous["video_prompt"],
        }
        return [image_item_with_metadata(ending["source_path"], metadata)]

    def _annotate_visual_memory(
        self,
        *,
        project: Dict[str, Any],
        shot: Dict[str, Any],
        shot_spec: ShotSpec,
        story_script: Dict[str, Any],
        config: RunConfig,
        attempt_id: str,
        keyframe_paths: List[str],
    ) -> None:
        self.repository.update_attempt(attempt_id, status="annotating_visual_memory")
        attempt = self.repository.get_attempt(attempt_id)
        memory_selection = attempt.get("memory_selection") or {}
        visual = dict(memory_selection.get("visual_element") or {})
        memory = self._visual_memory_before_shot(
            project=project,
            story_script=story_script,
            config=self._visual_config_for_shot(config, shot),
            shot_index=shot["order_index"],
        )
        planning_snapshot = visual.get("state_after_planning")
        if planning_snapshot:
            memory.load_snapshot(planning_snapshot)
        annotations = memory.annotate_completed_shot(
            shot_index=shot["order_index"] + 1,
            shot=shot_spec,
            keyframe_paths=keyframe_paths,
        )
        visual["current_annotations"] = annotations
        if memory.records:
            visual["current_annotation_visualizations"] = memory.records[-1].get(
                "current_annotation_visualizations",
                [],
            )
            visual["record"] = memory.records[-1]
        visual["state_after"] = memory.snapshot()
        visual.pop("state_after_planning", None)
        memory_selection["visual_element"] = visual
        self.repository.update_attempt(attempt_id, memory_selection=memory_selection)

    def _finalize_visual_memory_without_annotations(self, attempt_id: str) -> None:
        attempt = self.repository.get_attempt(attempt_id)
        memory_selection = attempt.get("memory_selection") or {}
        visual = dict(memory_selection.get("visual_element") or {})
        planning_snapshot = visual.get("state_after_planning")
        if planning_snapshot:
            visual["state_after"] = planning_snapshot
            visual["current_annotations"] = []
            visual["current_annotation_visualizations"] = []
            visual.pop("state_after_planning", None)
            memory_selection["visual_element"] = visual
            self.repository.update_attempt(attempt_id, memory_selection=memory_selection)

    def _prepare_smooth_reference(
        self,
        *,
        project: Dict[str, Any],
        shot: Dict[str, Any],
        attempt_dir: Path,
        references: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if shot["order_index"] == 0 or shot["is_cut"]:
            raise InvalidStateError("Smooth mode requires a non-cut predecessor")
        previous = project["shots"][shot["order_index"] - 1]
        if previous["state"] != "completed" or not previous.get("current_attempt_id"):
            raise InvalidStateError("Smooth mode requires a completed predecessor")
        previous_attempt = self.repository.get_attempt(previous["current_attempt_id"])
        raw_video = previous_attempt.get("output_video")
        if not raw_video or not Path(raw_video).is_file():
            raise InvalidStateError("Previous shot has no immutable raw output video")

        tail_path = attempt_dir / "smooth_reference_tail.mp4"
        self.tail_extractor(raw_video, tail_path, seconds=1.0)
        publisher = self.reference_video_publisher
        if publisher is None:
            publisher = configured_reference_video_publisher()
            self.reference_video_publisher = publisher
        public_url = require_public_https_url(publisher.publish(tail_path))
        metadata = {
            "label": "previous_tail_video",
            "roles": ["previous_tail_video"],
            "source_type": "auto",
            "source_path": str(tail_path),
            "file": tail_path.name,
            "source_shot_id": previous["shot_id"],
            "source_attempt_id": previous_attempt["attempt_id"],
            "source_scene_num": previous["scene_num"],
            "source_shot_num": previous["shot_num"],
            "source_prompt": previous["video_prompt"],
            "input_origin": "published_previous_raw_tail",
            "duration_seconds": 1.0,
        }
        memory_references = [
            item for item in references if not _is_previous_ending_reference(item)
        ]
        return [video_item_with_metadata(public_url, metadata), *memory_references]

    def _previous_seedance_last_frame_url(
        self,
        project: Dict[str, Any],
        shot: Dict[str, Any],
        client: SeedanceClient,
    ) -> str:
        if shot["order_index"] == 0:
            raise InvalidStateError("Last-frame-only mode requires a previous shot")
        previous = project["shots"][shot["order_index"] - 1]
        attempt_id = previous.get("current_attempt_id")
        if not attempt_id:
            raise InvalidStateError("Previous shot has no current Seedance attempt")
        attempt = self.repository.get_attempt(attempt_id)
        task_id = attempt.get("task_id")
        if not task_id:
            raise InvalidStateError("Previous shot has no Seedance task ID")
        response = client.get_task(task_id)
        return extract_last_frame_url(response)

    def _query_generator(
        self, project: Dict[str, Any], config: RunConfig
    ) -> Optional[MemoryQueryGenerator]:
        if not config.llm_memory_query:
            return None
        if self._memory_query_generator is None:
            self._memory_query_generator = MemoryQueryGenerator(
                cache_path=self.repository.project_dir(project["project_id"]) / "llm_memory_queries.jsonl",
                base_url=config.llm_memory_query_base_url,
                model=config.llm_memory_query_model,
            )
        return self._memory_query_generator

    def _client(self, config: RunConfig) -> SeedanceClient:
        if self.client is None:
            self.client = SeedanceClient(api_base=config.seedance_api_base, model=config.seedance_model)
        return self.client

    def _update_final_video(self, project_id: str) -> Optional[str]:
        prefix = self.repository.valid_completed_prefix(project_id)
        if not prefix:
            self.repository.set_project_state(project_id, "running", current_final_video=None)
            return None
        output = self.repository.project_dir(project_id) / "final" / "current.mp4"
        clips = []
        for item in prefix:
            snapshot = item.get("input_snapshot") or {}
            clips.append(
                {
                    "attempt_id": item["attempt_id"],
                    "output_video": item["output_video"],
                    "generation_mode": snapshot.get("generation_mode", "default"),
                    "attempt_dir": str(
                        self.repository.attempt_dir(
                            project_id, item["shot_id"], item["attempt_id"]
                        )
                    ),
                }
            )
        if any(item["generation_mode"] == "smooth" for item in clips):
            assembler = self.smooth_assembler
            if assembler is None:
                from .smooth_transition import SmoothVideoAssembler

                assembler = SmoothVideoAssembler()
                self.smooth_assembler = assembler
            result = assembler(clips, str(output))
        else:
            result = self.concat_function(
                [item["output_video"] for item in prefix], str(output)
            )
        self.repository.set_project_state(project_id, "running", current_final_video=result)
        return result


def _default_memory_selector(**kwargs: Any) -> List[Dict[str, Any]]:
    from storymem_seedance.runner import select_memory

    return select_memory(**kwargs)


def _memory_policy(shot: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "sink": bool(shot["memory_sink"]),
        "retrieve": bool(shot["memory_retrieve"]),
        "recent": bool(shot["memory_recent"]),
    }


def _pipeline_version(project: Dict[str, Any]) -> str:
    return str(
        (project.get("generation_config") or {}).get("pipeline_version") or "classic"
    )


def _filter_memory_records(
    records: List[Dict[str, Any]], policy: Dict[str, bool]
) -> List[Dict[str, Any]]:
    enabled_roles = {
        role
        for key, role in (
            ("sink", "early_sink_memory"),
            ("retrieve", "prompt_retrieved_memory"),
            ("recent", "recent_window_memory"),
        )
        if policy[key]
    }
    filtered = []
    for record in records:
        roles = [role for role in record.get("roles", []) if role in enabled_roles]
        if not roles:
            continue
        item = dict(record)
        item["roles"] = roles
        filtered.append(item)
    return filtered


def _memory_role_counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        key: sum(role in (item.get("roles") or []) for item in records)
        for key, role in (
            ("sink", "early_sink_memory"),
            ("retrieve", "prompt_retrieved_memory"),
            ("recent", "recent_window_memory"),
        )
    }


def _fit_storymem_memory_to_slots(
    records: List[Dict[str, Any]], slots: int
) -> List[Dict[str, Any]]:
    if slots <= 0:
        return []
    if len(records) <= slots:
        return records
    sink = [
        item
        for item in records
        if "early_sink_memory" in (item.get("roles") or [])
    ]
    non_sink = [
        item
        for item in records
        if "early_sink_memory" not in (item.get("roles") or [])
    ]
    if len(sink) >= slots:
        return sink[:slots]
    return [*sink, *non_sink[-(slots - len(sink)) :]]


def _classic_seedance_prompt(
    shot: Dict[str, Any], references: List[Dict[str, Any]]
) -> str:
    prompt = str(shot.get("video_prompt") or "").strip()
    if shot.get("is_cut") or not _has_previous_ending_reference(references):
        return prompt
    ending_index = next(
        (
            index
            for index, item in enumerate(references, start=1)
            if _is_previous_ending_reference(item)
        ),
        1,
    )
    return (
        f"参考图 Image {ending_index} 是上一段视频的最后一帧。"
        f"新视频的第一帧必须与 Image {ending_index} 保持一致，并从该画面自然继续。\n\n"
        f"{prompt}"
    )


def _story_from_project(project: Dict[str, Any]) -> Dict[str, Any]:
    source = dict(project.get("source_story") or {})
    scenes: Dict[int, Dict[str, Any]] = {}
    for shot in project["shots"]:
        scene = scenes.setdefault(
            shot["scene_num"],
            {
                "scene_num": shot["scene_num"],
                "video_prompts": [],
                "first_frame_prompt": [],
                "memory_queries": [],
                "cut": [],
                "durations": [],
            },
        )
        scene["video_prompts"].append(shot["video_prompt"])
        scene["first_frame_prompt"].append(shot["first_frame_prompt"])
        scene["memory_queries"].append(shot["memory_query"])
        scene["cut"].append(shot["is_cut"])
        scene["durations"].append(shot["duration_seconds"])
    source["story_name"] = project["name"]
    source["scenes"] = list(scenes.values())
    return source


def _shot_spec(shot: Dict[str, Any], story_script: Dict[str, Any]) -> ShotSpec:
    scene = next(item for item in story_script["scenes"] if item["scene_num"] == shot["scene_num"])
    return ShotSpec(
        scene=scene,
        scene_num=shot["scene_num"],
        shot_num=shot["shot_num"],
        prompt=shot["video_prompt"],
        is_cut=shot["is_cut"],
        first_frame_prompt=shot["first_frame_prompt"],
    )


def _usage_from_response(response: Dict[str, Any]) -> Dict[str, Any]:
    usage = dict(response.get("usage") or {})
    usage["duration_seconds"] = int(response.get("duration") or 0)
    usage["resolution"] = response.get("resolution")
    usage["model"] = response.get("model")
    return usage


def _memory_assets_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    assets = [
        {
            "asset_type": "retrieval_keyframe",
            "source_path": path,
            "rank": index,
            "active_in_memory_pool": True,
        }
        for index, path in enumerate(result.get("keyframe_paths") or [])
    ]
    if result.get("last_frame_path"):
        assets.append(
            {
                "asset_type": "ending_frame",
                "source_path": result["last_frame_path"],
                "rank": 0,
                "active_in_memory_pool": True,
            }
        )
    if result.get("motion_frames_path"):
        assets.append(
            {
                "asset_type": "motion_preview",
                "source_path": result["motion_frames_path"],
                "rank": 0,
                "active_in_memory_pool": False,
            }
        )
    return assets


def _read_last_jsonl(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    last = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def _has_previous_ending_reference(references: List[Dict[str, Any]]) -> bool:
    return any(_is_previous_ending_reference(item) for item in references)


def _is_previous_ending_reference(item: Dict[str, Any]) -> bool:
    return "previous_last_frame" in (item.get("metadata", {}).get("roles") or [])


def _previous_ending_reference(
    references: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    return next(
        (item for item in references if _is_previous_ending_reference(item)),
        None,
    )


def _use_original_previous_ending_url(
    references: List[Dict[str, Any]], last_frame_url: str
) -> List[Dict[str, Any]]:
    updated = []
    for item in references:
        if not _is_previous_ending_reference(item):
            updated.append(item)
            continue
        original = dict(item)
        original["image_url"] = {"url": last_frame_url}
        original["metadata"] = dict(item.get("metadata", {}))
        original["metadata"]["input_origin"] = "seedance_last_frame_url"
        updated.append(original)
    return updated


def _use_previous_ending_as_first_frame(
    references: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ending = _previous_ending_reference(references)
    if ending is None:
        raise InvalidStateError(
            "Last-frame-only mode requires an ending frame from the previous shot"
        )
    first_frame = dict(ending)
    first_frame["role"] = "first_frame"
    return [first_frame]


def _redact_response(value: Any) -> Any:
    from storymem_seedance.artifacts import redact_urls

    return redact_urls(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
