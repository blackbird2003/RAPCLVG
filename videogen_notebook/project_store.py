from __future__ import annotations

import json
import shutil
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import quote

from .domain import (
    DEFAULT_PROJECT_NAME,
    MAX_REFERENCE_IMAGES,
    PIPELINE_NAME,
    SCHEMA_VERSION,
    STEP_SEQUENCE,
    default_shot_inputs,
    default_settings,
    empty_story,
    normalize_story,
    validate_duration,
    validate_generation_mode,
    validate_settings,
)
from .json_store import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic


PREDEFINED_REFERENCE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}


class ProjectStoreError(RuntimeError):
    pass


class NotFoundError(ProjectStoreError):
    pass


class InvalidProjectError(ProjectStoreError):
    pass


class ProjectStore:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.projects_dir = self.workspace / "projects"
        self.locks_dir = self.workspace / "locks"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> List[Dict[str, Any]]:
        projects = []
        for project_file in sorted(self.projects_dir.glob("*/project.json")):
            try:
                project = read_json(project_file)
                if isinstance(project, dict):
                    projects.append(project)
            except (OSError, json.JSONDecodeError):
                continue
        projects.sort(key=lambda item: item.get("created_at") or item.get("project_id") or "", reverse=True)
        return projects

    def recover_interrupted_runs(self) -> List[str]:
        recovered: List[str] = []
        for project in self.list_projects():
            project_id = str(project.get("project_id") or "")
            if not project_id:
                continue
            has_running_project = project.get("status") == "running" or bool(project.get("active_shot_id"))
            running_shots = [
                shot for shot in self.list_shots(project_id)
                if _is_running_status(shot.get("state", {}).get("status"))
            ]
            if not has_running_project and not running_shots:
                continue
            for shot in running_shots:
                self._interrupt_shot(project_id, shot)
            self.update_project_json(project_id, {"status": "interrupted", "active_shot_id": None})
            recovered.append(project_id)
        return recovered

    def create_project(self, name: str | None = None) -> Dict[str, Any]:
        settings = self.get_default_project_settings()
        story = empty_story((name or DEFAULT_PROJECT_NAME).strip() or DEFAULT_PROJECT_NAME)
        story["scenes"][0]["durations"] = [settings["generation"]["default_duration_seconds"]]
        return self.import_story(story, name=name)

    def import_story(self, story: Dict[str, Any], *, name: str | None = None) -> Dict[str, Any]:
        settings = self.get_default_project_settings()
        normalized = normalize_story(
            story,
            default_duration=settings["generation"]["default_duration_seconds"],
            default_non_cut_mode=settings["generation"].get("default_non_cut_mode", "smooth"),
        )
        project_id = self._new_project_id()
        project_dir = self.project_dir(project_id)
        project_dir.mkdir(parents=True)
        now = _now()
        project_name = (name or normalized.title or DEFAULT_PROJECT_NAME).strip() or DEFAULT_PROJECT_NAME
        project = {
            "schema_version": SCHEMA_VERSION,
            "project_id": project_id,
            "name": project_name,
            "pipeline": PIPELINE_NAME,
            "status": "draft",
            "active_shot_id": None,
            "created_at": now,
            "updated_at": now,
            "thumbnail_asset_id": None,
            "shot_count": len(normalized.shots),
            "completed_prefix": 0,
            "current_final_video_asset_id": None,
            "cost_summary": {"estimated_cny": 0.0, "seedance_tasks": 0},
            "run_state": {"status": "idle"},
        }
        self._initialize_project_dirs(project_dir)
        write_json_atomic(project_dir / "project.json", project)
        write_json_atomic(project_dir / "settings.json", settings)
        write_json_atomic(project_dir / "story.json", normalized.to_dict())
        write_json_atomic(project_dir / "assets" / "manifest.json", {"schema_version": SCHEMA_VERSION, "assets": {}})
        write_json_atomic(project_dir / "memory" / "visual_state.json", {"schema_version": SCHEMA_VERSION, "elements": []})
        write_json_atomic(project_dir / "memory" / "lineage.json", {"schema_version": SCHEMA_VERSION, "completed_prefix": 0})
        for shot in normalized.shots:
            self._write_initial_shot(project_dir, shot.to_story_dict())
        return self.get_project(project_id)

    def get_default_project_settings(self) -> Dict[str, Any]:
        return validate_settings(read_json(self.workspace / "default_project_settings.json", default_settings()))

    def save_default_project_settings(self, settings_update: Dict[str, Any]) -> Dict[str, Any]:
        settings = validate_settings(_merged(self.get_default_project_settings(), settings_update or {}))
        write_json_atomic(self.workspace / "default_project_settings.json", settings)
        return settings

    def get_project(self, project_id: str) -> Dict[str, Any]:
        project_dir = self.project_dir(project_id)
        project = read_json(project_dir / "project.json")
        if not isinstance(project, dict):
            raise NotFoundError(f"Project not found: {project_id}")
        settings = validate_settings(read_json(project_dir / "settings.json", default_settings()))
        story = read_json(project_dir / "story.json", {})
        shots = self.list_shots(project_id)
        assets = read_json(project_dir / "assets" / "manifest.json", {"assets": {}})
        assembly = read_json(project_dir / "final" / "assembly.json", {})
        for shot in shots:
            self._hydrate_predefined_references(project_id, shot)
            current_attempt_id = shot.get("state", {}).get("current_attempt_id")
            if current_attempt_id:
                try:
                    shot["attempt"] = self.get_attempt(project_id, shot["shot_id"], current_attempt_id)
                except NotFoundError:
                    if shot.get("state", {}).get("status") == "running":
                        shot["attempt"] = None
                    else:
                        raise
            else:
                shot["attempt"] = None
        self._hydrate_visual_status_metadata(shots)
        return {
            "project": project,
            "settings": settings,
            "story": story,
            "shots": shots,
            "assets": assets.get("assets", {}) if isinstance(assets, dict) else {},
            "assembly": assembly if isinstance(assembly, dict) else {},
            "project_dir": str(project_dir),
        }

    def list_shots(self, project_id: str) -> List[Dict[str, Any]]:
        project_dir = self.project_dir(project_id)
        shots = []
        for shot_file in sorted((project_dir / "shots").glob("*/shot.json")):
            shot = read_json(shot_file)
            if isinstance(shot, dict):
                shot.setdefault("steps", _initial_step_state())
                shots.append(shot)
        shots.sort(key=lambda item: item.get("order_index", 0))
        return shots

    def save_project(
        self,
        project_id: str,
        *,
        name: str | None = None,
        settings_update: Dict[str, Any] | None = None,
        shot_updates: Iterable[Dict[str, Any]] = (),
    ) -> Dict[str, Any]:
        bundle = self.get_project(project_id)
        project_dir = self.project_dir(project_id)
        project = bundle["project"]
        settings = validate_settings(_merged(bundle["settings"], settings_update or {}))
        if name is not None:
            project["name"] = name.strip() or DEFAULT_PROJECT_NAME
        write_json_atomic(project_dir / "settings.json", settings)

        shots_by_id = {shot["shot_id"]: shot for shot in bundle["shots"]}
        for update in shot_updates:
            shot_id = str(update.get("shot_id") or "")
            if shot_id not in shots_by_id:
                raise NotFoundError(f"Shot not found: {shot_id}")
            shot = shots_by_id[shot_id]
            inputs = shot["inputs"]
            prompt = _normalize_textarea_text(update.get("video_prompt", inputs.get("video_prompt", "")))
            if not prompt:
                raise InvalidProjectError(f"Shot {shot_id} video prompt cannot be empty")
            is_cut = bool(update.get("is_cut", inputs.get("is_cut", False)))
            has_predecessor = int(shot["order_index"]) > 0
            mode = validate_generation_mode(
                update.get("generation_mode", inputs.get("generation_mode", "default")),
                is_cut=is_cut,
                has_predecessor=has_predecessor,
            )
            duration = validate_duration(update.get("duration_seconds", inputs.get("duration_seconds")))
            next_inputs = {
                "video_prompt": prompt,
                "is_cut": is_cut,
                "generation_mode": mode,
                "duration_seconds": duration,
                "predefined_references": _normalize_predefined_references_update(
                    update.get("predefined_references", inputs.get("predefined_references", [])),
                    inputs.get("predefined_references", []),
                ),
            }
            current_inputs = {
                **inputs,
                "video_prompt": _normalize_textarea_text(inputs.get("video_prompt", "")),
                "predefined_references": _normalize_predefined_references_update(
                    inputs.get("predefined_references", []),
                    inputs.get("predefined_references", []),
                ),
            }
            if shot.get("state", {}).get("status") not in {"draft", "failed", "interrupted"}:
                if next_inputs == current_inputs:
                    continue
                raise InvalidProjectError(f"Shot {shot_id} is locked in state {shot.get('state', {}).get('status')}")
            inputs.update(
                next_inputs
            )
            shot["state"]["updated_at"] = _now()
            write_json_atomic(self.shot_dir(project_id, shot_id) / "shot.json", shot)

        project["updated_at"] = _now()
        project["shot_count"] = len(shots_by_id)
        write_json_atomic(project_dir / "project.json", project)
        self._write_story_design(project_id)
        return self.get_project(project_id)

    def add_predefined_reference(
        self,
        project_id: str,
        shot_id: str,
        *,
        filename: str,
        data: bytes,
        label: str = "",
        guidance: str = "",
    ) -> Dict[str, Any]:
        bundle = self.get_project(project_id)
        project = bundle["project"]
        if project.get("status") == "running":
            raise InvalidProjectError("Cannot add predefined references while the project is running")
        shot = next((item for item in bundle["shots"] if item["shot_id"] == shot_id), None)
        if shot is None:
            raise NotFoundError(f"Shot not found: {shot_id}")
        if shot.get("state", {}).get("status") not in {"draft", "failed", "interrupted"}:
            raise InvalidProjectError(f"Shot {shot_id} is locked in state {shot.get('state', {}).get('status')}")
        if not data:
            raise InvalidProjectError("Uploaded reference image is empty")
        suffix = Path(filename or "").suffix.lower() or ".jpg"
        if suffix not in PREDEFINED_REFERENCE_EXTENSIONS:
            raise InvalidProjectError(f"Unsupported reference image extension: {suffix}")
        existing = _normalize_predefined_references_update(
            shot.get("inputs", {}).get("predefined_references", []),
            shot.get("inputs", {}).get("predefined_references", []),
        )
        if len(existing) >= MAX_REFERENCE_IMAGES:
            raise InvalidProjectError(f"Shot {shot_id} already has {MAX_REFERENCE_IMAGES} predefined references")
        reference_id = _next_predefined_reference_id(existing)
        asset_id = _predefined_asset_id(shot_id, reference_id)
        target = self.project_dir(project_id) / "assets" / "predefined_references" / f"{asset_id}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        record = {
            "id": reference_id,
            "asset_id": asset_id,
            "image_path": _relative(self.project_dir(project_id), target),
            "label": str(label or "").strip() or Path(filename or asset_id).stem,
            "guidance": _normalize_textarea_text(guidance),
        }
        existing.append(record)
        shot["inputs"]["predefined_references"] = existing
        shot["state"]["updated_at"] = _now()
        self.write_shot(project_id, shot)
        self.update_asset_manifest(project_id, {asset_id: _predefined_asset_record(self.project_dir(project_id), target, asset_id, shot_id, record)})
        self._write_story_design(project_id)
        return self.get_project(project_id)

    def add_shot(self, project_id: str) -> Dict[str, Any]:
        bundle = self.get_project(project_id)
        project = bundle["project"]
        if project.get("status") == "running":
            raise InvalidProjectError("Cannot add a shot while the project is running")
        shots = bundle["shots"]
        next_index = len(shots)
        last = shots[-1] if shots else None
        scene_num = int(last.get("scene_num", 1)) if last else 1
        shot_num = int(last.get("shot_num", 0)) + 1 if last else 1
        shot_id = f"{next_index + 1:04d}"
        inputs = default_shot_inputs(
            has_predecessor=bool(shots),
            default_non_cut_mode=bundle["settings"].get("generation", {}).get("default_non_cut_mode", "smooth"),
        )
        shot = {
            "shot_id": shot_id,
            "order_index": next_index,
            "scene_num": scene_num,
            "shot_num": shot_num,
            "video_prompt": inputs["video_prompt"],
            "is_cut": inputs["is_cut"],
            "generation_mode": inputs["generation_mode"],
            "duration_seconds": inputs["duration_seconds"],
        }
        self._write_initial_shot(self.project_dir(project_id), shot)
        project["shot_count"] = next_index + 1
        project["updated_at"] = _now()
        write_json_atomic(self.project_dir(project_id) / "project.json", project)
        self._write_story_design(project_id)
        return self.get_project(project_id)

    def save_visual_element_status(self, project_id: str, shot_id: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        bundle = self.get_project(project_id)
        project = bundle["project"]
        if project.get("status") == "running":
            raise InvalidProjectError("Cannot edit visual elements while the project is running")
        shots = bundle["shots"]
        shot = next((item for item in shots if item["shot_id"] == shot_id), None)
        if shot is None:
            raise NotFoundError(f"Shot not found: {shot_id}")
        if _is_running_status(shot.get("state", {}).get("status")):
            raise InvalidProjectError(f"Shot {shot_id} is currently running")
        attempt_id = shot.get("state", {}).get("current_attempt_id")
        if not attempt_id:
            raise InvalidProjectError(f"Shot {shot_id} has no visual plan attempt to edit")

        return self._write_visual_element_status_update(
            project_id,
            shot,
            rows,
            metadata_key="manual_edit",
            metadata={"updated_at": _now(), "row_count": len(rows)},
            log_message=f"manual visual element plan saved with {len(rows)} rows",
        )

    def apply_visual_plan_reflection(
        self,
        project_id: str,
        shot_id: str,
        rows: List[Dict[str, Any]],
        reflection: Dict[str, Any],
        *,
        allow_running_project: bool = False,
    ) -> Dict[str, Any]:
        bundle = self.get_project(project_id)
        project = bundle["project"]
        if project.get("status") == "running" and not allow_running_project:
            raise InvalidProjectError("Cannot reflect visual elements while the project is running")
        shot = next((item for item in bundle["shots"] if item["shot_id"] == shot_id), None)
        if shot is None:
            raise NotFoundError(f"Shot not found: {shot_id}")
        if _is_running_status(shot.get("state", {}).get("status")):
            raise InvalidProjectError(f"Shot {shot_id} is currently running")
        steps = shot.get("steps") or _initial_step_state()
        if steps.get("visual_plan", {}).get("status") != "visual_plan_completed":
            raise InvalidProjectError(f"Shot {shot_id} has no completed visual plan to reflect")
        return self._write_visual_element_status_update(
            project_id,
            shot,
            rows,
            metadata_key="reflection",
            metadata=reflection,
            log_message=f"visual plan reflection applied with {len(rows)} rows",
        )

    def _write_visual_element_status_update(
        self,
        project_id: str,
        shot: Dict[str, Any],
        rows: List[Dict[str, Any]],
        *,
        metadata_key: str,
        metadata: Dict[str, Any],
        log_message: str,
    ) -> Dict[str, Any]:
        shot_id = shot["shot_id"]
        attempt_id = shot.get("state", {}).get("current_attempt_id")
        if not attempt_id:
            raise InvalidProjectError(f"Shot {shot_id} has no visual plan attempt to update")

        attempt_dir = self.attempt_dir(project_id, shot_id, str(attempt_id))
        normalized = _normalize_visual_element_rows(rows, shot_id)
        write_json_atomic(attempt_dir / "visual_plan" / "visual_element_status.json", normalized)
        write_json_atomic(attempt_dir / "visual_element_status.json", normalized)

        details = read_json(attempt_dir / "visual_plan" / "details.json", {})
        if not isinstance(details, dict):
            details = {}
        existing_record = details.get("visual_element_record") if isinstance(details.get("visual_element_record"), dict) else {}
        record = _visual_element_record_from_rows(normalized, shot, existing_record if isinstance(existing_record, dict) else {})
        details["visual_element_record"] = record
        details[metadata_key] = metadata
        write_json_atomic(attempt_dir / "visual_plan" / "details.json", details)
        write_json_atomic(attempt_dir / "visual_element_details.json", details)

        logs = read_jsonl(attempt_dir / "visual_plan" / "logs.jsonl")
        logs.append({"time": _now(), "step": "planning_visual_elements", "message": log_message})
        write_jsonl_atomic(attempt_dir / "visual_plan" / "logs.jsonl", logs)

        _clear_downstream_attempt_outputs(attempt_dir)
        attempt = read_json(attempt_dir / "attempt.json", {})
        if not isinstance(attempt, dict):
            attempt = {"schema_version": SCHEMA_VERSION, "attempt_id": str(attempt_id), "shot_id": shot_id}
        attempt.update(
            {
                "status": "visual_plan_completed",
                "updated_at": _now(),
                "visual_element_status": normalized,
                "selected_references": [],
                "produced_visual_memory": [],
                "prompt": {"submitted_prompt": ""},
                "seedance": {"usage": {}},
                "outputs": {},
                "postprocess": {},
                "error": None,
            }
        )
        write_json_atomic(attempt_dir / "attempt.json", attempt)

        shot["steps"] = shot.get("steps") or _initial_step_state()
        shot["steps"]["visual_plan"] = _completed_step_state("visual_plan_completed")
        for step in STEP_SEQUENCE[1:]:
            shot["steps"][step] = _draft_step_state()
        shot["state"]["status"] = "visual_plan_completed"
        shot["state"]["updated_at"] = _now()
        self.write_shot(project_id, shot)

        self._reset_from_index(project_id, int(shot["order_index"]) + 1)
        return self.get_project(project_id)

    def save_seedance_prompt(self, project_id: str, shot_id: str, prompt: str) -> Dict[str, Any]:
        bundle = self.get_project(project_id)
        if bundle["project"].get("status") == "running":
            raise InvalidProjectError("Cannot edit Seedance prompt while the project is running")
        shot = next((item for item in bundle["shots"] if item["shot_id"] == shot_id), None)
        if shot is None:
            raise NotFoundError(f"Shot not found: {shot_id}")
        if _is_running_status(shot.get("state", {}).get("status")):
            raise InvalidProjectError(f"Shot {shot_id} is currently running")
        prompt = str(prompt or "").strip()
        if not prompt:
            raise InvalidProjectError("Seedance prompt cannot be empty")
        attempt_id = shot.get("state", {}).get("current_attempt_id")
        if not attempt_id:
            raise InvalidProjectError(f"Shot {shot_id} has no Seedance prompt attempt to edit")
        steps = shot.get("steps") or _initial_step_state()
        if steps.get("seedance_prompt", {}).get("status") not in {"seedance_prompt_completed", "seedance_prompt_failed"}:
            raise InvalidProjectError(f"Shot {shot_id} has no completed Seedance prompt to edit")

        attempt_dir = self.attempt_dir(project_id, shot_id, str(attempt_id))
        (attempt_dir / "seedance_prompt").mkdir(parents=True, exist_ok=True)
        write_json_atomic(attempt_dir / "seedance_prompt" / "prompt.json", {"submitted_prompt": prompt})
        (attempt_dir / "seedance_prompt" / "prompt.txt").write_text(prompt, encoding="utf-8")
        logs = read_jsonl(attempt_dir / "seedance_prompt" / "logs.jsonl")
        logs.append({"time": _now(), "step": "composing_prompt", "message": "manual Seedance prompt saved"})
        write_jsonl_atomic(attempt_dir / "seedance_prompt" / "logs.jsonl", logs)

        _clear_attempt_outputs_after_step(attempt_dir, "seedance_prompt")
        attempt = read_json(attempt_dir / "attempt.json", {})
        if not isinstance(attempt, dict):
            attempt = {"schema_version": SCHEMA_VERSION, "attempt_id": str(attempt_id), "shot_id": shot_id}
        attempt.update(
            {
                "status": "seedance_prompt_completed",
                "updated_at": _now(),
                "prompt": {"submitted_prompt": prompt, "manual_edit": True},
                "seedance": {"usage": {}},
                "outputs": {},
                "postprocess": {},
                "produced_visual_memory": [],
                "error": None,
            }
        )
        write_json_atomic(attempt_dir / "attempt.json", attempt)
        write_jsonl_atomic(attempt_dir / "logs.jsonl", _all_step_logs_for_store(attempt_dir))

        steps["seedance_prompt"] = _completed_step_state("seedance_prompt_completed")
        for step in STEP_SEQUENCE[STEP_SEQUENCE.index("seedance_prompt") + 1:]:
            steps[step] = _draft_step_state()
        shot["steps"] = steps
        shot["state"]["status"] = "seedance_prompt_completed"
        shot["state"]["updated_at"] = _now()
        self.write_shot(project_id, shot)

        self._reset_from_index(project_id, int(shot["order_index"]) + 1)
        return self.get_project(project_id)

    def save_selected_references(self, project_id: str, shot_id: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        bundle = self.get_project(project_id)
        if bundle["project"].get("status") == "running":
            raise InvalidProjectError("Cannot edit references while the project is running")
        shot = next((item for item in bundle["shots"] if item["shot_id"] == shot_id), None)
        if shot is None:
            raise NotFoundError(f"Shot not found: {shot_id}")
        if _is_running_status(shot.get("state", {}).get("status")):
            raise InvalidProjectError(f"Shot {shot_id} is currently running")
        attempt_id = shot.get("state", {}).get("current_attempt_id")
        if not attempt_id:
            raise InvalidProjectError(f"Shot {shot_id} has no reference selection attempt to edit")
        steps = shot.get("steps") or _initial_step_state()
        if steps.get("reference_selection", {}).get("status") not in {"reference_selection_completed", "reference_selection_failed"}:
            raise InvalidProjectError(f"Shot {shot_id} has no completed reference selection to edit")

        attempt_dir = self.attempt_dir(project_id, shot_id, str(attempt_id))
        existing = read_json(attempt_dir / "reference_selection" / "selected_references.json", [])
        existing_by_id = {
            str(item.get("reference_id")): item
            for item in existing
            if isinstance(item, dict) and item.get("reference_id")
        }
        updated: List[Dict[str, Any]] = []
        for row in rows:
            reference_id = str(row.get("reference_id") or "")
            if not reference_id or not row.get("keep", True):
                continue
            base = deepcopy(existing_by_id.get(reference_id))
            if not isinstance(base, dict):
                continue
            base["reference_guidance"] = str(row.get("reference_guidance") or "").strip()
            updated.append(base)
        for index, item in enumerate(updated, start=1):
            item["reference_index"] = index

        write_json_atomic(attempt_dir / "reference_selection" / "selected_references.json", updated)
        write_json_atomic(attempt_dir / "selected_references.json", updated)
        logs = read_jsonl(attempt_dir / "reference_selection" / "logs.jsonl")
        logs.append({"time": _now(), "step": "selecting_visual_references", "message": "manual reference guidance saved"})
        write_jsonl_atomic(attempt_dir / "reference_selection" / "logs.jsonl", logs)

        details = read_json(attempt_dir / "reference_selection" / "details.json", {})
        if not isinstance(details, dict):
            details = {}
        details["manual_edit"] = {"updated_at": _now(), "edited_fields": ["reference_guidance"], "reference_count": len(updated)}
        write_json_atomic(attempt_dir / "reference_selection" / "details.json", details)

        _clear_attempt_outputs_after_step(attempt_dir, "reference_selection")
        attempt = read_json(attempt_dir / "attempt.json", {})
        if not isinstance(attempt, dict):
            attempt = {"schema_version": SCHEMA_VERSION, "attempt_id": str(attempt_id), "shot_id": shot_id}
        attempt.update(
            {
                "status": "reference_selection_completed",
                "updated_at": _now(),
                "selected_references": updated,
                "prompt": {"submitted_prompt": ""},
                "seedance": {"usage": {}},
                "outputs": {},
                "postprocess": {},
                "produced_visual_memory": [],
                "error": None,
            }
        )
        write_json_atomic(attempt_dir / "attempt.json", attempt)
        write_jsonl_atomic(attempt_dir / "logs.jsonl", _all_step_logs_for_store(attempt_dir))

        steps["reference_selection"] = _completed_step_state("reference_selection_completed")
        for step in STEP_SEQUENCE[STEP_SEQUENCE.index("reference_selection") + 1:]:
            steps[step] = _draft_step_state()
        shot["steps"] = steps
        shot["state"]["status"] = "reference_selection_completed"
        shot["state"]["updated_at"] = _now()
        self.write_shot(project_id, shot)

        self._reset_from_index(project_id, int(shot["order_index"]) + 1)
        return self.get_project(project_id)

    def save_produced_visual_memory(self, project_id: str, shot_id: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        bundle = self.get_project(project_id)
        if bundle["project"].get("status") == "running":
            raise InvalidProjectError("Cannot edit produced visual memory while the project is running")
        shot = next((item for item in bundle["shots"] if item["shot_id"] == shot_id), None)
        if shot is None:
            raise NotFoundError(f"Shot not found: {shot_id}")
        if _is_running_status(shot.get("state", {}).get("status")):
            raise InvalidProjectError(f"Shot {shot_id} is currently running")
        attempt_id = shot.get("state", {}).get("current_attempt_id")
        if not attempt_id:
            raise InvalidProjectError(f"Shot {shot_id} has no keyframe memory attempt to edit")
        steps = shot.get("steps") or _initial_step_state()
        if steps.get("keyframe_maintaining", {}).get("status") not in {"keyframe_maintaining_completed", "keyframe_maintaining_failed"}:
            raise InvalidProjectError(f"Shot {shot_id} has no completed keyframe memory to edit")

        attempt_dir = self.attempt_dir(project_id, shot_id, str(attempt_id))
        existing = read_json(attempt_dir / "keyframe_maintaining" / "produced_visual_memory.json", [])
        existing_by_id = {
            str(item.get("asset_id")): item
            for item in existing
            if isinstance(item, dict) and item.get("asset_id")
        }
        updated: List[Dict[str, Any]] = []
        for row in rows:
            asset_id = str(row.get("asset_id") or "")
            if not asset_id or not row.get("keep", True):
                continue
            base = deepcopy(existing_by_id.get(asset_id))
            if not isinstance(base, dict):
                continue
            holistic = str(row.get("holistic_description") or "").strip()
            base["holistic_description"] = holistic
            frame_annotation = base.get("frame_annotation")
            if isinstance(frame_annotation, dict):
                frame_annotation["holistic_description"] = holistic
            updated.append(base)
        for index, item in enumerate(updated, start=1):
            item["rank"] = index

        write_json_atomic(attempt_dir / "keyframe_maintaining" / "produced_visual_memory.json", updated)
        write_json_atomic(attempt_dir / "produced_visual_memory.json", updated)
        logs = read_jsonl(attempt_dir / "keyframe_maintaining" / "logs.jsonl")
        logs.append({"time": _now(), "step": "annotating_visual_memory", "message": "manual holistic description saved"})
        write_jsonl_atomic(attempt_dir / "keyframe_maintaining" / "logs.jsonl", logs)

        attempt = read_json(attempt_dir / "attempt.json", {})
        if not isinstance(attempt, dict):
            attempt = {"schema_version": SCHEMA_VERSION, "attempt_id": str(attempt_id), "shot_id": shot_id}
        attempt.update(
            {
                "status": "completed",
                "updated_at": _now(),
                "produced_visual_memory": updated,
                "error": None,
            }
        )
        write_json_atomic(attempt_dir / "attempt.json", attempt)
        write_jsonl_atomic(attempt_dir / "logs.jsonl", _all_step_logs_for_store(attempt_dir))

        steps["keyframe_maintaining"] = _completed_step_state("keyframe_maintaining_completed")
        shot["steps"] = steps
        shot["state"]["status"] = "completed"
        shot["state"]["updated_at"] = _now()
        self.write_shot(project_id, shot)

        self._reset_from_index(project_id, int(shot["order_index"]) + 1)
        self.refresh_memory_state(project_id)
        return self.get_project(project_id)

    def export_story_design(self, project_id: str) -> Dict[str, Any]:
        bundle = self.get_project(project_id)
        scenes: List[Dict[str, Any]] = []
        by_scene: Dict[int, List[Dict[str, Any]]] = {}
        for shot in bundle["shots"]:
            by_scene.setdefault(int(shot["scene_num"]), []).append(shot)
        for scene_num, shots in sorted(by_scene.items()):
            ordered = sorted(shots, key=lambda item: int(item["shot_num"]))
            scenes.append(
                {
                    "scene_num": scene_num,
                    "video_prompts": [shot["inputs"]["video_prompt"] for shot in ordered],
                    "cut": [bool(shot["inputs"]["is_cut"]) for shot in ordered],
                    "generation_modes": [shot["inputs"]["generation_mode"] for shot in ordered],
                    "durations": [int(shot["inputs"]["duration_seconds"]) for shot in ordered],
                    "predefined_references": [
                        _export_predefined_references(self.project_dir(project_id), shot["inputs"].get("predefined_references", []))
                        for shot in ordered
                    ],
                }
            )
        story = bundle.get("story") or {}
        return {
            "story_name": bundle["project"].get("name") or story.get("title") or DEFAULT_PROJECT_NAME,
            "story_overview": story.get("overview") or "",
            "scenes": scenes,
        }

    def duplicate_project(self, project_id: str, *, name: str | None = None) -> Dict[str, Any]:
        source_dir = self.project_dir(project_id)
        if not (source_dir / "project.json").exists():
            raise NotFoundError(f"Project not found: {project_id}")
        new_id = self._new_project_id()
        target_dir = self.project_dir(new_id)
        shutil.copytree(source_dir, target_dir)
        project = read_json(target_dir / "project.json")
        now = _now()
        project["project_id"] = new_id
        project["name"] = (name or f"{project.get('name', DEFAULT_PROJECT_NAME)} copy").strip()
        copied_running = project.get("status") == "running"
        project["status"] = "interrupted" if copied_running else project.get("status", "draft")
        project["active_shot_id"] = None
        project["created_at"] = now
        project["updated_at"] = now
        write_json_atomic(target_dir / "project.json", project)
        if copied_running:
            self._interrupt_running_shots(new_id)
        self._remove_runtime_files(target_dir)
        return self.get_project(new_id)

    def fork_after_shot(self, project_id: str, shot_id: str, *, name: str | None = None) -> Dict[str, Any]:
        original = self.get_project(project_id)
        shot_ids = [shot["shot_id"] for shot in original["shots"]]
        if shot_id not in shot_ids:
            raise NotFoundError(f"Shot not found: {shot_id}")
        fork = self.duplicate_project(
            project_id,
            name=name or f"{original['project']['name']} - fork after shot {shot_id}",
        )
        self._reset_after(fork["project"]["project_id"], shot_id)
        return self.get_project(fork["project"]["project_id"])

    def reset_from_shot(self, project_id: str, shot_id: str) -> Dict[str, Any]:
        bundle = self.get_project(project_id)
        shot_ids = [shot["shot_id"] for shot in bundle["shots"]]
        if shot_id not in shot_ids:
            raise NotFoundError(f"Shot not found: {shot_id}")
        reset_index = next(
            int(shot["order_index"]) for shot in bundle["shots"] if shot["shot_id"] == shot_id
        )
        self._reset_from_index(project_id, reset_index, reset_from_shot_id=shot_id)
        return self.get_project(project_id)

    def delete_project(self, project_id: str) -> None:
        project_dir = self.project_dir(project_id)
        if not (project_dir / "project.json").exists():
            raise NotFoundError(f"Project not found: {project_id}")
        project = read_json(project_dir / "project.json", {})
        if project.get("status") == "running":
            raise InvalidProjectError("Cannot delete a running project")
        shutil.rmtree(project_dir)

    def get_attempt(self, project_id: str, shot_id: str, attempt_id: str) -> Dict[str, Any]:
        attempt_dir = self.attempt_dir(project_id, shot_id, attempt_id)
        attempt = read_json(attempt_dir / "attempt.json")
        if not isinstance(attempt, dict):
            raise NotFoundError(f"Attempt not found: {attempt_id}")
        for key, filename in (
            ("visual_element_status", "visual_element_status.json"),
            ("selected_references", "selected_references.json"),
            ("produced_visual_memory", "produced_visual_memory.json"),
        ):
            value = read_json(_step_file(attempt_dir, filename), [])
            attempt[key] = value if isinstance(value, list) else []
        request = read_json(_step_file(attempt_dir, "request.json"), {})
        response = read_json(_step_file(attempt_dir, "response.json"), {})
        attempt["request"] = _request_for_display(request) if isinstance(request, dict) else {}
        attempt["response"] = response if isinstance(response, dict) else {}
        seedance_details = read_json(attempt_dir / "seedance_generation" / "details.json", {})
        if not isinstance(seedance_details, dict):
            seedance_details = {}
        media_debug = read_json(attempt_dir / "seedance_generation" / "media_debug.json", {})
        if isinstance(media_debug, dict) and media_debug:
            seedance_details["media_debug"] = media_debug
        attempt["seedance_details"] = seedance_details
        visual_details = _merged_details(
            read_json(attempt_dir / "visual_plan" / "details.json", {}),
            read_json(attempt_dir / "visual_element_details.json", {}),
            read_json(attempt_dir / "reference_selection" / "details.json", {}),
            read_json(attempt_dir / "keyframe_maintaining" / "details.json", {}),
        )
        attempt["visual_element_details"] = visual_details
        logs = []
        for step in STEP_SEQUENCE:
            logs.extend(read_jsonl(attempt_dir / step / "logs.jsonl"))
        if not logs:
            logs = read_jsonl(attempt_dir / "logs.jsonl")
        attempt["logs"] = logs
        prompt_text = read_json(attempt_dir / "seedance_prompt" / "prompt.json", {})
        if not isinstance(prompt_text, dict) or not prompt_text.get("submitted_prompt"):
            prompt_text = read_json(attempt_dir / "seedance_generation" / "prompt.json", {})
        if isinstance(prompt_text, dict) and prompt_text.get("submitted_prompt"):
            attempt.setdefault("prompt", {})["submitted_prompt"] = prompt_text["submitted_prompt"]
        step_state = read_json(attempt_dir / "steps.json", {})
        attempt["steps"] = step_state if isinstance(step_state, dict) else {}
        self._hydrate_attempt_media(project_id, shot_id, attempt)
        return attempt

    def next_attempt_id(self, project_id: str, shot_id: str) -> str:
        attempts_dir = self.shot_dir(project_id, shot_id) / "attempts"
        max_number = 0
        for attempt_dir in attempts_dir.glob("a[0-9][0-9][0-9]"):
            try:
                max_number = max(max_number, int(attempt_dir.name[1:]))
            except ValueError:
                continue
        return f"a{max_number + 1:03d}"

    def attempt_dir(self, project_id: str, shot_id: str, attempt_id: str) -> Path:
        _validate_local_id(attempt_id, "attempt_id")
        return self.shot_dir(project_id, shot_id) / "attempts" / attempt_id

    def update_project_json(self, project_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        project_dir = self.project_dir(project_id)
        project = read_json(project_dir / "project.json")
        if not isinstance(project, dict):
            raise NotFoundError(f"Project not found: {project_id}")
        project.update(updates)
        project["updated_at"] = _now()
        write_json_atomic(project_dir / "project.json", project)
        return project

    def write_shot(self, project_id: str, shot: Dict[str, Any]) -> None:
        write_json_atomic(self.shot_dir(project_id, shot["shot_id"]) / "shot.json", shot)

    def update_asset_manifest(self, project_id: str, assets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        manifest_path = self.project_dir(project_id) / "assets" / "manifest.json"
        manifest = read_json(manifest_path, {"schema_version": SCHEMA_VERSION, "assets": {}})
        if not isinstance(manifest, dict):
            manifest = {"schema_version": SCHEMA_VERSION, "assets": {}}
        manifest.setdefault("schema_version", SCHEMA_VERSION)
        manifest.setdefault("assets", {}).update(assets)
        write_json_atomic(manifest_path, manifest)
        return manifest

    def refresh_memory_state(self, project_id: str) -> Dict[str, Any]:
        bundle = self.get_project(project_id)
        project_dir = self.project_dir(project_id)
        completed_prefix = _completed_prefix(bundle["shots"])
        elements: List[Dict[str, Any]] = []
        selected_references: List[Dict[str, Any]] = []
        produced_memory: List[Dict[str, Any]] = []
        lineage_shots: List[Dict[str, Any]] = []
        for shot in bundle["shots"][:completed_prefix]:
            attempt = shot.get("attempt") or {}
            attempt_id = attempt.get("attempt_id")
            for element in attempt.get("visual_element_status") or []:
                item = dict(element)
                item.setdefault("source_shot_id", shot["shot_id"])
                item.setdefault("attempt_id", attempt_id)
                elements.append(item)
            for reference in attempt.get("selected_references") or []:
                item = dict(reference)
                item.setdefault("target_shot_id", shot["shot_id"])
                item.setdefault("attempt_id", attempt_id)
                selected_references.append(item)
            for memory in attempt.get("produced_visual_memory") or []:
                item = dict(memory)
                item.setdefault("source_shot_id", shot["shot_id"])
                item.setdefault("attempt_id", attempt_id)
                produced_memory.append(item)
            lineage_shots.append(
                {
                    "shot_id": shot["shot_id"],
                    "attempt_id": attempt_id,
                    "raw_video_asset_id": attempt.get("outputs", {}).get("raw_video_asset_id"),
                }
            )
        now = _now()
        visual_state = {
            "schema_version": SCHEMA_VERSION,
            "completed_prefix": completed_prefix,
            "updated_at": now,
            "elements": elements,
            "selected_references": selected_references,
            "produced_visual_memory": produced_memory,
        }
        write_json_atomic(project_dir / "memory" / "visual_state.json", visual_state)
        existing_lineage = read_json(project_dir / "memory" / "lineage.json", {})
        lineage = {
            "schema_version": SCHEMA_VERSION,
            "completed_prefix": completed_prefix,
            "updated_at": now,
            "shots": lineage_shots,
        }
        if isinstance(existing_lineage, dict):
            for key in ("fork_after_shot_id", "reset_from_shot_id"):
                if existing_lineage.get(key):
                    lineage[key] = existing_lineage[key]
        write_json_atomic(project_dir / "memory" / "lineage.json", lineage)
        return visual_state

    def asset_path(self, project_id: str, asset_id: str) -> Path:
        _validate_local_id(asset_id, "asset_id")
        project_dir = self.project_dir(project_id)
        manifest = read_json(project_dir / "assets" / "manifest.json", {"assets": {}})
        if not isinstance(manifest, dict):
            raise NotFoundError(f"Asset not found: {asset_id}")
        asset = manifest.get("assets", {}).get(asset_id)
        if not isinstance(asset, dict) or not asset.get("path"):
            raise NotFoundError(f"Asset not found: {asset_id}")
        path = (project_dir / str(asset["path"])).resolve()
        try:
            path.relative_to(project_dir.resolve())
        except ValueError as exc:
            raise InvalidProjectError(f"Asset path escapes project bundle: {asset_id}") from exc
        if not path.exists() or not path.is_file():
            raise NotFoundError(f"Asset file not found: {asset_id}")
        return path

    def project_file_path(self, project_id: str, relative_path: str) -> Path:
        project_dir = self.project_dir(project_id).resolve()
        path = (project_dir / relative_path).resolve()
        try:
            path.relative_to(project_dir)
        except ValueError as exc:
            raise InvalidProjectError(f"Project file path escapes project bundle: {relative_path}") from exc
        if not path.exists() or not path.is_file():
            raise NotFoundError(f"Project file not found: {relative_path}")
        return path

    def project_dir(self, project_id: str) -> Path:
        _validate_local_id(project_id, "project_id")
        return self.projects_dir / project_id

    def shot_dir(self, project_id: str, shot_id: str) -> Path:
        _validate_local_id(shot_id, "shot_id")
        return self.project_dir(project_id) / "shots" / shot_id

    def _reset_after(self, project_id: str, shot_id: str) -> None:
        bundle = self.get_project(project_id)
        fork_index = next(int(shot["order_index"]) for shot in bundle["shots"] if shot["shot_id"] == shot_id)
        self._reset_from_index(project_id, fork_index + 1, fork_after_shot_id=shot_id)

    def _reset_from_index(
        self,
        project_id: str,
        start_index: int,
        *,
        fork_after_shot_id: str | None = None,
        reset_from_shot_id: str | None = None,
    ) -> None:
        bundle = self.get_project(project_id)
        project = bundle["project"]
        project_dir = self.project_dir(project_id)
        for shot in bundle["shots"]:
            if int(shot["order_index"]) < start_index:
                continue
            state = shot["state"]
            current_attempt_id = state.get("current_attempt_id")
            if current_attempt_id:
                archived = shot.setdefault("archived_attempt_ids", [])
                if current_attempt_id not in archived:
                    archived.append(current_attempt_id)
            state.update({"status": "draft", "current_attempt_id": None, "updated_at": _now()})
            state.pop("current_final_video_asset_id", None)
            shot["steps"] = _initial_step_state()
            write_json_atomic(self.shot_dir(project_id, shot["shot_id"]) / "shot.json", shot)
        updated_shots = self.list_shots(project_id)
        completed_prefix = _completed_prefix(updated_shots)
        prefix_shot = updated_shots[completed_prefix - 1] if completed_prefix else None
        prefix_shot_id = prefix_shot["shot_id"] if prefix_shot else None
        project["status"] = "draft"
        project["active_shot_id"] = None
        project["completed_prefix"] = completed_prefix
        project["updated_at"] = _now()
        project["current_final_video_asset_id"] = (
            (_shot_current_final_asset_id(prefix_shot) or _prefix_final_asset(project_dir, prefix_shot_id))
            if prefix_shot_id and prefix_shot
            else None
        )
        write_json_atomic(project_dir / "project.json", project)
        lineage = {"schema_version": SCHEMA_VERSION, "completed_prefix": completed_prefix}
        if fork_after_shot_id:
            lineage["fork_after_shot_id"] = fork_after_shot_id
        if reset_from_shot_id:
            lineage["reset_from_shot_id"] = reset_from_shot_id
        write_json_atomic(project_dir / "memory" / "lineage.json", lineage)
        self.refresh_memory_state(project_id)

    def _write_initial_shot(self, project_dir: Path, shot: Dict[str, Any]) -> None:
        shot_dir = project_dir / "shots" / shot["shot_id"]
        shot_dir.mkdir(parents=True, exist_ok=True)
        now = _now()
        data = {
            "shot_id": shot["shot_id"],
            "order_index": int(shot["shot_id"]) - 1,
            "scene_num": shot["scene_num"],
            "shot_num": shot["shot_num"],
            "inputs": {
                "video_prompt": shot["video_prompt"],
                "is_cut": shot["is_cut"],
                "generation_mode": shot["generation_mode"],
                "duration_seconds": shot["duration_seconds"],
                "predefined_references": self._copy_predefined_references(
                    project_dir,
                    str(shot["shot_id"]),
                    shot.get("predefined_references") or [],
                ),
            },
            "state": {
                "status": "draft",
                "revision": 1,
                "current_attempt_id": None,
                "updated_at": now,
            },
            "steps": _initial_step_state(),
            "archived_attempt_ids": [],
        }
        write_json_atomic(shot_dir / "shot.json", data)
        (shot_dir / "attempts").mkdir(exist_ok=True)

    def _initialize_project_dirs(self, project_dir: Path) -> None:
        for relative in (
            "assets/images",
            "assets/videos",
            "assets/thumbnails",
            "assets/embeddings",
            "assets/predefined_references",
            "assets/temp",
            "memory",
            "shots",
            "final",
            "logs",
        ):
            (project_dir / relative).mkdir(parents=True, exist_ok=True)

    def _remove_runtime_files(self, project_dir: Path) -> None:
        for path in (project_dir / "assets" / "temp").glob("*"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

    def _interrupt_running_shots(self, project_id: str) -> None:
        for shot in self.list_shots(project_id):
            if _is_running_status(shot.get("state", {}).get("status")):
                self._interrupt_shot(project_id, shot)

    def _interrupt_shot(self, project_id: str, shot: Dict[str, Any]) -> None:
        state = shot.get("state") or {}
        current_attempt_id = state.get("current_attempt_id")
        if current_attempt_id:
            archived = shot.setdefault("archived_attempt_ids", [])
            if current_attempt_id not in archived:
                archived.append(current_attempt_id)
        state["status"] = "interrupted"
        state["current_attempt_id"] = None
        state["updated_at"] = _now()
        self.write_shot(project_id, shot)

    def _write_story_design(self, project_id: str) -> None:
        write_json_atomic(self.project_dir(project_id) / "story.json", _normalized_story_from_bundle(self.get_project(project_id)))

    def _hydrate_attempt_media(self, project_id: str, shot_id: str, attempt: Dict[str, Any]) -> None:
        project_dir = self.project_dir(project_id).resolve()
        for reference in attempt.get("selected_references") or []:
            if not isinstance(reference, dict):
                continue
            reference.setdefault("holistic_description", "")
            reference.setdefault("reference_guidance", "")
            if reference.get("asset_id"):
                reference.setdefault("media_url", _asset_url(project_id, str(reference["asset_id"])))
            visualization_url = self._project_file_url(project_id, project_dir, reference.get("visualization_path"))
            if visualization_url:
                reference["visualization_url"] = visualization_url
            source_url = self._project_file_url(project_id, project_dir, reference.get("source_path"))
            if source_url:
                reference.setdefault("media_url", source_url)

        for memory in attempt.get("produced_visual_memory") or []:
            if not isinstance(memory, dict):
                continue
            memory.setdefault("holistic_description", "")
            asset_id = str(memory.get("asset_id") or "")
            if asset_id:
                memory.setdefault("media_url", _asset_url(project_id, asset_id))
                annotation_path = (
                    project_dir
                    / "memory"
                    / "visual_element_memory"
                    / f"shot-{shot_id}"
                    / f"current_{asset_id}_annotation.jpg"
                )
                visualization_url = self._project_file_url(project_id, project_dir, annotation_path)
                if visualization_url:
                    memory["visualization_url"] = visualization_url
            frame_annotation = memory.get("frame_annotation") if isinstance(memory.get("frame_annotation"), dict) else {}
            if frame_annotation:
                frame_annotation.setdefault("holistic_description", memory.get("holistic_description") or "")
                memory.setdefault("source_scene_num", frame_annotation.get("source_scene_num"))
                memory.setdefault("source_shot_num", frame_annotation.get("source_shot_num"))
                elements = frame_annotation.get("elements")
                if isinstance(elements, list):
                    for element in elements:
                        if isinstance(element, dict):
                            element.setdefault("reference_quality", "full")
                    memory["elements"] = elements

    def _hydrate_predefined_references(self, project_id: str, shot: Dict[str, Any]) -> None:
        project_dir = self.project_dir(project_id).resolve()
        references = shot.get("inputs", {}).get("predefined_references")
        if not isinstance(references, list):
            shot.setdefault("inputs", {})["predefined_references"] = []
            return
        for reference in references:
            if not isinstance(reference, dict):
                continue
            asset_id = str(reference.get("asset_id") or "")
            if asset_id:
                reference["media_url"] = _asset_url(project_id, asset_id)
                continue
            url = self._project_file_url(project_id, project_dir, reference.get("image_path"))
            if url:
                reference["media_url"] = url

    def _copy_predefined_references(self, project_dir: Path, shot_id: str, references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = _normalize_imported_predefined_references(references)
        if len(normalized) > MAX_REFERENCE_IMAGES:
            raise InvalidProjectError(f"Shot {shot_id} predefined references exceed Seedance's {MAX_REFERENCE_IMAGES}-image limit")
        copied: List[Dict[str, Any]] = []
        assets: Dict[str, Dict[str, Any]] = {}
        for index, reference in enumerate(normalized, start=1):
            source = _resolve_import_image_path(reference["image_path"])
            if not source.is_file():
                raise InvalidProjectError(f"Predefined reference image not found: {source}")
            suffix = source.suffix.lower() or ".jpg"
            if suffix not in PREDEFINED_REFERENCE_EXTENSIONS:
                raise InvalidProjectError(f"Unsupported predefined reference image extension: {source}")
            reference_id = f"pref-{index:04d}"
            asset_id = _predefined_asset_id(shot_id, reference_id)
            target = project_dir / "assets" / "predefined_references" / f"{asset_id}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            item = {
                "id": reference_id,
                "asset_id": asset_id,
                "image_path": _relative(project_dir, target),
                "label": reference.get("label") or source.stem,
                "guidance": reference.get("guidance") or "",
            }
            copied.append(item)
            assets[asset_id] = _predefined_asset_record(project_dir, target, asset_id, shot_id, item)
        if assets:
            manifest_path = project_dir / "assets" / "manifest.json"
            manifest = read_json(manifest_path, {"schema_version": SCHEMA_VERSION, "assets": {}})
            if not isinstance(manifest, dict):
                manifest = {"schema_version": SCHEMA_VERSION, "assets": {}}
            manifest.setdefault("schema_version", SCHEMA_VERSION)
            manifest.setdefault("assets", {}).update(assets)
            write_json_atomic(manifest_path, manifest)
        return copied

    def _hydrate_visual_status_metadata(self, shots: List[Dict[str, Any]]) -> None:
        registry: Dict[str, Dict[str, Any]] = {}
        for shot in sorted(shots, key=lambda item: item.get("order_index", 0)):
            attempt = shot.get("attempt") or {}
            rows = attempt.get("visual_element_status") or []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                element_id = str(row.get("element_id") or row.get("id") or "")
                if not element_id:
                    continue
                metadata = registry.get(element_id)
                if metadata:
                    introduced_at = str(row.get("introduced_at") or "")
                    if not introduced_at or introduced_at == element_id:
                        row["introduced_at"] = metadata.get("introduced_at") or introduced_at
                    if not row.get("notes"):
                        row["notes"] = metadata.get("notes") or ""
                if element_id not in registry and (row.get("introduced_at") or row.get("notes")):
                    registry[element_id] = {
                        "introduced_at": str(row.get("introduced_at") or ""),
                        "notes": str(row.get("notes") or ""),
                    }

    def _project_file_url(self, project_id: str, project_dir: Path, path_value: Any) -> str | None:
        if not path_value:
            return None
        path = Path(str(path_value))
        if not path.is_absolute():
            path = project_dir / path
        try:
            relative = path.resolve().relative_to(project_dir)
        except (OSError, ValueError):
            return None
        if not (project_dir / relative).is_file():
            return None
        return f"/projects/{project_id}/files/{quote(relative.as_posix(), safe='/')}"

    def _new_project_id(self) -> str:
        while True:
            project_id = f"p_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            if not self.project_dir(project_id).exists():
                return project_id


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _validate_local_id(value: str, field_name: str) -> None:
    if not value or any(part in value for part in ("/", "\\", "..")):
        raise InvalidProjectError(f"Invalid {field_name}: {value}")


def _asset_url(project_id: str, asset_id: str) -> str:
    return f"/projects/{project_id}/assets/{quote(asset_id, safe='')}"


def _merged(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    _merge_into(result, update)
    return result


def _normalize_textarea_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_imported_predefined_references(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise InvalidProjectError("predefined_references must be a list")
    references: List[Dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise InvalidProjectError(f"predefined reference {index} must be an object")
        image_path = str(item.get("image_path") or item.get("path") or "").strip()
        if not image_path:
            raise InvalidProjectError(f"predefined reference {index} missing image_path")
        references.append(
            {
                "image_path": image_path,
                "label": str(item.get("label") or "").strip(),
                "guidance": _normalize_textarea_text(item.get("guidance") or item.get("description") or ""),
            }
        )
    return references


def _normalize_predefined_references_update(value: Any, existing: Any) -> List[Dict[str, Any]]:
    existing_by_id = {
        str(item.get("id") or ""): item
        for item in (existing or [])
        if isinstance(item, dict) and item.get("id")
    }
    if value is None:
        value = existing
    if not isinstance(value, list):
        raise InvalidProjectError("predefined_references must be a list")
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        reference_id = str(item.get("id") or "").strip()
        base = existing_by_id.get(reference_id)
        if not reference_id or not isinstance(base, dict):
            raise InvalidProjectError(f"Unknown predefined reference row {index}: {reference_id}")
        if reference_id in seen:
            raise InvalidProjectError(f"Duplicate predefined reference id: {reference_id}")
        seen.add(reference_id)
        normalized.append(
            {
                "id": reference_id,
                "asset_id": str(base.get("asset_id") or item.get("asset_id") or "").strip(),
                "image_path": str(base.get("image_path") or item.get("image_path") or "").strip(),
                "label": str(item.get("label") or "").strip(),
                "guidance": _normalize_textarea_text(item.get("guidance") or ""),
            }
        )
    if len(normalized) > MAX_REFERENCE_IMAGES:
        raise InvalidProjectError(f"predefined references exceed Seedance's {MAX_REFERENCE_IMAGES}-image limit")
    return normalized


def _next_predefined_reference_id(references: List[Dict[str, Any]]) -> str:
    used = {str(item.get("id") or "") for item in references}
    index = 1
    while True:
        reference_id = f"pref-{index:04d}"
        if reference_id not in used:
            return reference_id
        index += 1


def _predefined_asset_id(shot_id: str, reference_id: str) -> str:
    safe_reference_id = reference_id.replace("-", "_")
    return f"pref_{shot_id}_{safe_reference_id}"


def _resolve_import_image_path(value: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _predefined_asset_record(
    project_dir: Path,
    target_path: Path,
    asset_id: str,
    shot_id: str,
    reference: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "kind": "image",
        "path": _relative(project_dir, target_path),
        "created_at": _now(),
        "source": {
            "type": "predefined_reference",
            "shot_id": shot_id,
            "reference_id": reference.get("id"),
        },
        "metadata": {
            "label": reference.get("label") or "",
            "guidance": reference.get("guidance") or "",
        },
    }


def _export_predefined_references(project_dir: Path, references: Any) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for item in references or []:
        if not isinstance(item, dict):
            continue
        image_path = str(item.get("image_path") or "").strip()
        if image_path and project_dir.is_absolute():
            image_path = str((project_dir / image_path).resolve())
        result.append(
            {
                "image_path": image_path,
                "label": str(item.get("label") or "").strip(),
                "guidance": str(item.get("guidance") or "").strip(),
            }
        )
    return result


def _relative(project_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _merged_details(*items: Any) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for item in items:
        if isinstance(item, dict):
            _merge_into(merged, item)
    return merged


def _request_for_display(request: Dict[str, Any]) -> Dict[str, Any]:
    display = deepcopy(request)
    content = display.get("content")
    if isinstance(content, list):
        display["content"] = [_request_content_item_for_display(item) for item in content]
    return display


def _request_content_item_for_display(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    current = deepcopy(item)
    image_url = current.get("image_url")
    if not isinstance(image_url, dict):
        return current
    url = str(image_url.get("url") or "")
    if not url.startswith("data:"):
        return current
    metadata = current.get("metadata") if isinstance(current.get("metadata"), dict) else {}
    image_url["url"] = str(metadata.get("source_path") or metadata.get("file") or "[embedded image data]")
    image_url["embedded_data_url"] = True
    if ";" in url[:80]:
        image_url["mime"] = url[5 : url.index(";")]
    return current


def _merge_into(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_into(target[key], value)
        else:
            target[key] = value


def _normalize_visual_element_rows(rows: List[Dict[str, Any]], shot_id: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        element_id = str(raw.get("element_id") or raw.get("id") or "").strip()
        element_type = _normalize_visual_element_type(raw.get("type"))
        status = str(raw.get("status") or "optional_or_uncertain").strip() or "optional_or_uncertain"
        introduced_at = str(raw.get("introduced_at") or shot_id).strip() or shot_id
        notes = str(raw.get("notes") or "").strip()
        reason = str(raw.get("reason") or raw.get("state_reason") or "").strip()
        if not name and not element_id and not notes and not reason:
            continue
        if not name:
            raise InvalidProjectError(f"Visual element row {index} name cannot be empty")
        if status not in {"should_reference", "should_exclude", "optional_or_uncertain", "new"}:
            raise InvalidProjectError(f"Visual element row {index} has invalid status: {status}")
        if not element_id:
            element_id = _next_manual_element_id(shot_id, used_ids)
        if element_id in used_ids:
            raise InvalidProjectError(f"Duplicate visual element id: {element_id}")
        used_ids.add(element_id)
        normalized.append(
            {
                "element_id": element_id,
                "name": name,
                "type": element_type,
                "introduced_at": introduced_at,
                "notes": notes,
                "status": status,
                "reason": reason,
            }
        )
    return normalized


def _next_manual_element_id(shot_id: str, used_ids: set[str]) -> str:
    index = 1
    while True:
        element_id = f"element-{shot_id}-manual-{index:03d}"
        if element_id not in used_ids:
            return element_id
        index += 1


def _normalize_visual_element_type(value: Any) -> str:
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


def _visual_element_record_from_rows(rows: List[Dict[str, Any]], shot: Dict[str, Any], existing_record: Dict[str, Any]) -> Dict[str, Any]:
    registry_after = [
        {
            "id": row["element_id"],
            "name": row["name"],
            "type": row["type"],
            "introduced_at": row["introduced_at"],
            "notes": row["notes"],
        }
        for row in rows
    ]
    inserted = [item for item, row in zip(registry_after, rows) if row["status"] == "new"]
    decision = {
        "existing_element_states": [
            {
                "id": row["element_id"],
                "state": row["status"],
                "reason": row["reason"],
            }
            for row in rows
            if row["status"] != "new"
        ],
        "new_elements": [
            {
                "name": row["name"],
                "type": row["type"],
                "notes": row["notes"] or row["reason"],
            }
            for row in rows
            if row["status"] == "new"
        ],
        "shot_notes": list((existing_record.get("decision") or {}).get("shot_notes") or []),
    }
    record = deepcopy(existing_record)
    record.update(
        {
            "shot": f"shot-{shot['shot_id']}",
            "scene_num": shot.get("scene_num"),
            "shot_num": shot.get("shot_num"),
            "decision": decision,
            "inserted": inserted,
            "registry_after": registry_after,
            "manual_edit": True,
            "manual_rows": rows,
        }
    )
    return record


def _clear_downstream_attempt_outputs(attempt_dir: Path) -> None:
    _clear_attempt_outputs_after_step(attempt_dir, "visual_plan")


def _clear_attempt_outputs_after_step(attempt_dir: Path, step: str) -> None:
    remove_after = {
        "visual_plan": (
            "selected_references.json",
            "request.json",
            "response.json",
            "produced_visual_memory.json",
            "reference_selection/selected_references.json",
            "reference_selection/details.json",
            "reference_selection/prompt_context.json",
            "reference_selection/logs.jsonl",
            "seedance_prompt/prompt.json",
            "seedance_prompt/prompt.txt",
            "seedance_prompt/request_content_summary.json",
            "seedance_prompt/details.json",
            "seedance_prompt/logs.jsonl",
            "seedance_generation/prompt.json",
            "seedance_generation/prompt.txt",
            "seedance_generation/request.json",
            "seedance_generation/response.json",
            "seedance_generation/logs.jsonl",
            "keyframe_maintaining/produced_visual_memory.json",
            "keyframe_maintaining/postprocess.json",
            "keyframe_maintaining/details.json",
            "keyframe_maintaining/logs.jsonl",
        ),
        "seedance_prompt": (
            "request.json",
            "response.json",
            "produced_visual_memory.json",
            "seedance_generation/prompt.json",
            "seedance_generation/prompt.txt",
            "seedance_generation/request.json",
            "seedance_generation/response.json",
            "seedance_generation/logs.jsonl",
            "keyframe_maintaining/produced_visual_memory.json",
            "keyframe_maintaining/postprocess.json",
            "keyframe_maintaining/details.json",
            "keyframe_maintaining/logs.jsonl",
        ),
        "reference_selection": (
            "request.json",
            "response.json",
            "produced_visual_memory.json",
            "seedance_prompt/prompt.json",
            "seedance_prompt/prompt.txt",
            "seedance_prompt/request_content_summary.json",
            "seedance_prompt/details.json",
            "seedance_prompt/logs.jsonl",
            "seedance_generation/prompt.json",
            "seedance_generation/prompt.txt",
            "seedance_generation/request.json",
            "seedance_generation/response.json",
            "seedance_generation/logs.jsonl",
            "keyframe_maintaining/produced_visual_memory.json",
            "keyframe_maintaining/postprocess.json",
            "keyframe_maintaining/details.json",
            "keyframe_maintaining/logs.jsonl",
        ),
    }
    for relative in remove_after.get(step, ()):
        (attempt_dir / relative).unlink(missing_ok=True)


def _all_step_logs_for_store(attempt_dir: Path) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    for step in STEP_SEQUENCE:
        logs.extend(read_jsonl(attempt_dir / step / "logs.jsonl"))
    return logs


def _initial_step_state() -> Dict[str, Dict[str, Any]]:
    return {step: _draft_step_state() for step in STEP_SEQUENCE}


def _draft_step_state() -> Dict[str, Any]:
    return {"status": "draft", "started_at": None, "updated_at": None}


def _completed_step_state(status: str) -> Dict[str, Any]:
    now = _now()
    return {"status": status, "started_at": now, "updated_at": now}


def _is_running_status(status: Any) -> bool:
    value = str(status or "")
    return value == "running" or value.endswith("_running") or value.endswith("_queued")


def _step_file(attempt_dir: Path, filename: str) -> Path:
    mapping = {
        "visual_element_status.json": attempt_dir / "visual_plan" / "visual_element_status.json",
        "visual_element_details.json": attempt_dir / "visual_plan" / "details.json",
        "selected_references.json": attempt_dir / "reference_selection" / "selected_references.json",
        "request.json": attempt_dir / "seedance_generation" / "request.json",
        "response.json": attempt_dir / "seedance_generation" / "response.json",
        "produced_visual_memory.json": attempt_dir / "keyframe_maintaining" / "produced_visual_memory.json",
    }
    path = mapping.get(filename, attempt_dir / filename)
    return path if path.exists() else attempt_dir / filename


def _normalized_story_from_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    story = bundle.get("story") or {}
    return {
        "title": bundle.get("project", {}).get("name") or story.get("title") or DEFAULT_PROJECT_NAME,
        "overview": story.get("overview") or "",
        "shots": [
            {
                "shot_id": shot["shot_id"],
                "scene_num": int(shot["scene_num"]),
                "shot_num": int(shot["shot_num"]),
                "video_prompt": shot["inputs"]["video_prompt"],
                "is_cut": bool(shot["inputs"]["is_cut"]),
                "generation_mode": shot["inputs"]["generation_mode"],
                "duration_seconds": int(shot["inputs"]["duration_seconds"]),
                "predefined_references": deepcopy(shot["inputs"].get("predefined_references") or []),
            }
            for shot in bundle.get("shots", [])
        ],
        "source": {
            "story_name": bundle.get("project", {}).get("name") or DEFAULT_PROJECT_NAME,
            "story_overview": story.get("overview") or "",
            "scenes": _story_design_scenes(bundle.get("shots", [])),
        },
    }


def _story_design_scenes(shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_scene: Dict[int, List[Dict[str, Any]]] = {}
    for shot in shots:
        by_scene.setdefault(int(shot["scene_num"]), []).append(shot)
    scenes = []
    for scene_num, scene_shots in sorted(by_scene.items()):
        ordered = sorted(scene_shots, key=lambda item: int(item["shot_num"]))
        scenes.append(
            {
                "scene_num": scene_num,
                "video_prompts": [shot["inputs"]["video_prompt"] for shot in ordered],
                "cut": [bool(shot["inputs"]["is_cut"]) for shot in ordered],
                "generation_modes": [shot["inputs"]["generation_mode"] for shot in ordered],
                "durations": [int(shot["inputs"]["duration_seconds"]) for shot in ordered],
                "predefined_references": [
                    _export_predefined_references(Path(), shot["inputs"].get("predefined_references", []))
                    for shot in ordered
                ],
            }
        )
    return scenes


def _prefix_final_asset(project_dir: Path, shot_id: str) -> str | None:
    if not shot_id:
        return None
    assembly = read_json(project_dir / "final" / "assembly.json", {})
    if not isinstance(assembly, dict):
        return None
    final_assets = assembly.get("prefix_final_video_asset_ids")
    if isinstance(final_assets, dict):
        value = final_assets.get(shot_id)
        if value:
            return str(value)
    prefix_assets = assembly.get("prefix_video_asset_ids")
    if isinstance(prefix_assets, dict):
        value = prefix_assets.get(shot_id)
        return str(value) if value else None
    return None


def _shot_current_final_asset_id(shot: Dict[str, Any] | None) -> str | None:
    if not shot:
        return None
    state_value = (shot.get("state") or {}).get("current_final_video_asset_id")
    if state_value:
        return str(state_value)
    attempt_value = ((shot.get("attempt") or {}).get("outputs") or {}).get("current_final_video_asset_id")
    return str(attempt_value) if attempt_value else None


def _completed_prefix(shots: List[Dict[str, Any]]) -> int:
    count = 0
    for shot in sorted(shots, key=lambda item: item.get("order_index", 0)):
        if shot.get("state", {}).get("status") != "completed":
            break
        count += 1
    return count
