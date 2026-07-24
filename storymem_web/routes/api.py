from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..repository import ConflictError, InvalidStateError, NotFoundError
from ..schemas import (
    ProjectCreateRequest,
    ProjectSettingsUpdateRequest,
    ProjectUpdateRequest,
    RunRequest,
    ShotUpdateRequest,
)
from ..views import project_view, shot_view


router = APIRouter(prefix="/api")


def _repo(request: Request):
    return request.app.state.repository


def _raise_api_error(exc: Exception) -> None:
    if isinstance(exc, NotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, (ConflictError, InvalidStateError)):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


@router.get("/health")
async def health(request: Request):
    return {"status": "ok", **request.app.state.runtime_info}


@router.get("/projects")
def list_projects(request: Request):
    return _repo(request).list_projects()


@router.post("/projects", status_code=201)
def create_project(request: Request, payload: ProjectCreateRequest):
    try:
        return _repo(request).create_empty_project(payload.name)
    except Exception as exc:
        _raise_api_error(exc)


@router.get("/projects/{project_id}")
def get_project(request: Request, project_id: str):
    try:
        return project_view(_repo(request), project_id)
    except Exception as exc:
        _raise_api_error(exc)


@router.patch("/projects/{project_id}")
def update_project(request: Request, project_id: str, payload: ProjectUpdateRequest):
    try:
        return _repo(request).rename_project(project_id, payload.name)
    except Exception as exc:
        _raise_api_error(exc)


@router.patch("/projects/{project_id}/settings")
def update_project_settings(
    request: Request,
    project_id: str,
    payload: ProjectSettingsUpdateRequest,
):
    try:
        return _repo(request).update_project_generation_config(
            project_id,
            payload.dict(),
        )
    except Exception as exc:
        _raise_api_error(exc)


@router.patch("/projects/{project_id}/shots/{shot_id}")
def update_shot(
    request: Request,
    project_id: str,
    shot_id: str,
    payload: ShotUpdateRequest,
):
    try:
        project = _repo(request).get_project(project_id)
        if not any(shot["shot_id"] == shot_id for shot in project["shots"]):
            raise NotFoundError("Shot does not belong to this project")
        return _repo(request).update_shot(
            shot_id,
            video_prompt=payload.video_prompt,
            is_cut=payload.is_cut,
            generation_mode=payload.generation_mode,
            duration_seconds=payload.duration_seconds,
            memory_sink=payload.memory_sink,
            memory_retrieve=payload.memory_retrieve,
            memory_recent=payload.memory_recent,
            expected_row_version=payload.expected_row_version,
        )
    except Exception as exc:
        _raise_api_error(exc)


@router.post("/projects/{project_id}/run", status_code=202)
def run_project(request: Request, project_id: str, payload: RunRequest):
    try:
        return request.app.state.process_manager.start(
            project_id,
            payload.mode,
            payload.shot_id,
        )
    except Exception as exc:
        _raise_api_error(exc)


@router.post("/projects/{project_id}/interrupt", status_code=202)
def interrupt_project(request: Request, project_id: str):
    job = request.app.state.process_manager.interrupt(project_id)
    if job is None:
        raise HTTPException(status_code=409, detail="Project has no active job")
    return job


@router.post("/projects/{project_id}/shots/{shot_id}/retry-keyframes", status_code=202)
def retry_keyframes(request: Request, project_id: str, shot_id: str):
    try:
        shot = shot_view(_repo(request), project_id, shot_id)
        if not shot["can_retry_keyframes"]:
            raise InvalidStateError("Shot has no failed post-processing step to retry")
        return request.app.state.process_manager.start(project_id, "keyframes", shot_id)
    except Exception as exc:
        _raise_api_error(exc)


@router.post("/projects/{project_id}/shots/{shot_id}/retry-smoothing", status_code=202)
def retry_smoothing(request: Request, project_id: str, shot_id: str):
    try:
        shot = shot_view(_repo(request), project_id, shot_id)
        if not shot["can_retry_smoothing"]:
            raise InvalidStateError("Shot has no failed Smooth assembly to retry")
        return request.app.state.process_manager.start(
            project_id, "smooth_postprocess", shot_id
        )
    except Exception as exc:
        _raise_api_error(exc)


@router.post("/projects/{project_id}/shots/{shot_id}/reset")
def reset_from_shot(request: Request, project_id: str, shot_id: str):
    try:
        if _repo(request).active_job(project_id):
            raise ConflictError("Interrupt the active project before resetting a shot")
        return _repo(request).reset_from_shot(project_id, shot_id)
    except Exception as exc:
        _raise_api_error(exc)
