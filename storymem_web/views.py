from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

from .repository import ProjectRepository
from .settings import merged_generation_config


ACTIVE_PROJECT_STATES = {"running"}
EDITABLE_STATES = {"draft", "queued", "stale", "failed", "interrupted"}


def project_view(repository: ProjectRepository, project_id: str) -> Dict[str, Any]:
    project = repository.get_project(project_id)
    active_job = repository.active_job(project_id)
    completed_prefix = repository.valid_completed_prefix(project_id)
    shots = []
    predecessors_complete = True
    for shot in project["shots"]:
        attempt = None
        references = []
        produced = []
        if shot["current_attempt_id"]:
            attempt = repository.get_attempt(shot["current_attempt_id"])
            references = repository.list_references(attempt["attempt_id"])
            produced = repository.list_memory_assets(
                project_id,
                attempt_id=attempt["attempt_id"],
                current_only=False,
            )
            if attempt.get("output_video"):
                attempt["output_url"] = media_url(repository, project_id, attempt["output_video"])
            _augment_visual_element_view(repository, project_id, attempt)
        for reference in references:
            reference["media_url"] = media_url(
                repository, project_id, reference["source_path"], strict=False
            )
            reference["media_kind"] = (
                "video"
                if Path(reference["source_path"]).suffix.lower() in {".mp4", ".webm"}
                else "image"
            )
        for asset in produced:
            asset["media_url"] = media_url(repository, project_id, asset["source_path"])
        can_retry_keyframes = bool(
            active_job is None
            and attempt
            and attempt["status"] in {"failed", "interrupted"}
            and attempt.get("output_url")
            and Path(attempt["output_video"]).is_file()
            and (attempt.get("error") or {}).get("type") != "SmoothTransitionError"
        )
        can_retry_smoothing = bool(
            active_job is None
            and attempt
            and attempt["status"] in {"failed", "interrupted"}
            and attempt.get("output_url")
            and Path(attempt["output_video"]).is_file()
            and (attempt.get("error") or {}).get("type") == "SmoothTransitionError"
        )
        view = dict(shot)
        view.update(
            {
                "attempt": attempt,
                "references": references,
                "produced_memory": produced,
                "can_edit": shot["state"] in EDITABLE_STATES,
                "can_run": (
                    active_job is None
                    and predecessors_complete
                    and shot["state"] in EDITABLE_STATES | {"failed", "interrupted"}
                ),
                "can_reset": shot["state"] in {"completed", "failed", "interrupted", "stale"},
                "can_retry_keyframes": can_retry_keyframes,
                "can_retry_smoothing": can_retry_smoothing,
                "poll": (
                    project["status"] in ACTIVE_PROJECT_STATES
                    and shot["state"] != "completed"
                ),
            }
        )
        shots.append(view)
        predecessors_complete = predecessors_complete and shot["state"] == "completed"
    project["shots"] = shots
    project["active_job"] = active_job
    project["generation_config"] = merged_generation_config(project["generation_config"])
    project["completed_count"] = len(completed_prefix)
    project["shot_count"] = len(shots)
    project["cost"] = repository.cost_summary(project_id)
    if project.get("current_final_video"):
        project["final_video_url"] = media_url(
            repository, project_id, project["current_final_video"]
        )
    else:
        project["final_video_url"] = None
    return project


def shot_view(repository: ProjectRepository, project_id: str, shot_id: str) -> Dict[str, Any]:
    project = project_view(repository, project_id)
    return next(shot for shot in project["shots"] if shot["shot_id"] == shot_id)


def media_url(
    repository: ProjectRepository,
    project_id: str,
    path: str,
    *,
    strict: bool = True,
) -> Optional[str]:
    if not path:
        return None
    project_dir = repository.project_dir(project_id).resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(project_dir)
    except ValueError:
        if strict:
            return None
        return None
    return f"/media/{project_id}/{quote(relative.as_posix())}"


def _augment_visual_element_view(
    repository: ProjectRepository,
    project_id: str,
    attempt: Dict[str, Any],
) -> None:
    visual = (attempt.get("memory_selection") or {}).get("visual_element")
    if not isinstance(visual, dict):
        attempt["visual_element_status"] = []
        attempt["selected_visual_references"] = []
        attempt["produced_visual_memory"] = []
        return

    element_by_id = {}
    for element in (visual.get("state_after_planning") or visual.get("state_after") or {}).get(
        "registry", []
    ):
        if isinstance(element, dict):
            element_by_id[element.get("id")] = element
    status_rows = []
    decision = visual.get("decision") or {}
    for item in decision.get("existing_element_states") or []:
        element = element_by_id.get(item.get("id"), {})
        status_rows.append(
            {
                "id": item.get("id"),
                "name": element.get("name") or item.get("id"),
                "type": element.get("type", ""),
                "introduced_at": element.get("introduced_at", ""),
                "notes": element.get("notes", ""),
                "state": item.get("state"),
                "state_reason": item.get("reason", ""),
            }
        )
    for item in visual.get("inserted") or []:
        status_rows.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "type": item.get("type", ""),
                "introduced_at": item.get("introduced_at", ""),
                "notes": item.get("notes", ""),
                "state": "new",
                "state_reason": item.get("notes", ""),
            }
        )
    attempt["visual_element_status"] = status_rows

    selected = []
    for index, item in enumerate(visual.get("selected_references") or [], start=1):
        row = dict(item)
        row["reference_index"] = row.get("reference_index") or index
        row["media_url"] = media_url(
            repository, project_id, str(row.get("source_path") or ""), strict=False
        )
        if row.get("visualization_path"):
            row["visualization_url"] = media_url(
                repository, project_id, str(row["visualization_path"]), strict=False
            )
        selected.append(row)
    attempt["selected_visual_references"] = selected

    visualizations = visual.get("current_annotation_visualizations") or []
    produced = []
    for index, annotation in enumerate(visual.get("current_annotations") or []):
        row = dict(annotation)
        row["media_url"] = media_url(
            repository, project_id, str(row.get("frame_path") or ""), strict=False
        )
        if index < len(visualizations):
            row["visualization_path"] = visualizations[index]
            row["visualization_url"] = media_url(
                repository, project_id, str(visualizations[index]), strict=False
            )
        produced.append(row)
    attempt["produced_visual_memory"] = produced
