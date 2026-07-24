from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import json5
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from ..repository import InvalidStateError, ProjectRepository
from ..views import project_view, shot_view


router = APIRouter()
MEDIA_SUFFIXES = {".mp4", ".webm", ".jpg", ".jpeg", ".png"}


def _repository(request: Request) -> ProjectRepository:
    return request.app.state.repository


def _redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url=url, status_code=303)


def _error_redirect(url: str, exc: Exception) -> RedirectResponse:
    separator = "&" if "?" in url else "?"
    return _redirect(f"{url}{separator}error={quote(str(exc))}")


@router.get("/")
def projects_page(request: Request):
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="projects.html",
        context={
            "projects": _repository(request).list_projects(),
            "error": request.query_params.get("error"),
        },
    )


@router.post("/projects")
def create_project(request: Request, name: str = Form("Untitled video project")):
    try:
        project = _repository(request).create_empty_project(name)
        return _redirect(f"/projects/{project['project_id']}")
    except Exception as exc:
        return _error_redirect("/", exc)


@router.post("/projects/import")
async def import_project(request: Request, story_file: UploadFile = File(...)):
    try:
        raw = await story_file.read()
        story = json5.loads(raw.decode("utf-8"))
        project = _repository(request).create_project(story)
        return _redirect(f"/projects/{project['project_id']}")
    except Exception as exc:
        return _error_redirect("/", exc)


@router.get("/projects/{project_id}")
def project_page(request: Request, project_id: str):
    try:
        project = project_view(_repository(request), project_id)
    except Exception as exc:
        return _error_redirect("/", exc)
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="project.html",
        context={"project": project, "error": request.query_params.get("error")},
    )


@router.post("/projects/{project_id}/rename")
def rename_project(request: Request, project_id: str, name: str = Form(...)):
    try:
        _repository(request).rename_project(project_id, name)
        return _redirect(f"/projects/{project_id}")
    except Exception as exc:
        return _error_redirect(f"/projects/{project_id}", exc)


@router.post("/projects/{project_id}/settings")
def update_project_settings(
    request: Request,
    project_id: str,
    pipeline_version: str = Form(...),
    visual_element_sink_frame_count: int = Form(...),
    visual_element_max_retrieved_frames: int = Form(...),
):
    try:
        _repository(request).update_project_generation_config(
            project_id,
            {
                "pipeline_version": pipeline_version,
                "visual_element_sink_frame_count": visual_element_sink_frame_count,
                "visual_element_max_retrieved_frames": visual_element_max_retrieved_frames,
            },
        )
        return _redirect(f"/projects/{project_id}")
    except Exception as exc:
        return _error_redirect(f"/projects/{project_id}", exc)


@router.get("/projects/{project_id}/shots/{shot_id}/runtime")
def shot_runtime(request: Request, project_id: str, shot_id: str):
    project = project_view(_repository(request), project_id)
    shot = next((item for item in project["shots"] if item["shot_id"] == shot_id), None)
    if shot is None:
        raise HTTPException(status_code=404, detail="Shot not found")
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="fragments/shot_runtime.html",
        context={"project": project, "shot": shot},
    )


@router.get("/projects/{project_id}/summary")
def project_summary(request: Request, project_id: str):
    project = project_view(_repository(request), project_id)
    return request.app.state.templates.TemplateResponse(
        request=request,
        name="fragments/project_summary.html",
        context={"project": project},
    )


@router.post("/projects/{project_id}/shots/{shot_id}")
def update_shot(
    request: Request,
    project_id: str,
    shot_id: str,
    video_prompt: str = Form(...),
    is_cut: str | None = Form(None),
    generation_mode: str = Form("default"),
    duration_seconds: int = Form(...),
    memory_sink: str | None = Form(None),
    memory_retrieve: str | None = Form(None),
    memory_recent: str | None = Form(None),
    row_version: int = Form(...),
):
    try:
        _repository(request).update_shot(
            shot_id,
            video_prompt=video_prompt,
            is_cut=is_cut == "on",
            generation_mode=generation_mode,
            duration_seconds=duration_seconds,
            memory_sink=None if memory_sink is None else memory_sink == "on",
            memory_retrieve=None if memory_retrieve is None else memory_retrieve == "on",
            memory_recent=None if memory_recent is None else memory_recent == "on",
            expected_row_version=row_version,
        )
        return _redirect(f"/projects/{project_id}")
    except Exception as exc:
        return _error_redirect(f"/projects/{project_id}", exc)


@router.post("/projects/{project_id}/run")
def run_project(request: Request, project_id: str):
    try:
        request.app.state.process_manager.start(project_id, "all")
        return _redirect(f"/projects/{project_id}")
    except Exception as exc:
        return _error_redirect(f"/projects/{project_id}", exc)


@router.post("/projects/{project_id}/shots/{shot_id}/run")
def run_shot(request: Request, project_id: str, shot_id: str):
    try:
        request.app.state.process_manager.start(project_id, "single", shot_id)
        return _redirect(f"/projects/{project_id}")
    except Exception as exc:
        return _error_redirect(f"/projects/{project_id}", exc)


@router.post("/projects/{project_id}/shots/{shot_id}/retry-keyframes")
def retry_keyframes(request: Request, project_id: str, shot_id: str):
    try:
        shot = shot_view(_repository(request), project_id, shot_id)
        if not shot["can_retry_keyframes"]:
            raise InvalidStateError("Shot has no failed post-processing step to retry")
        request.app.state.process_manager.start(project_id, "keyframes", shot_id)
        return _redirect(f"/projects/{project_id}")
    except Exception as exc:
        return _error_redirect(f"/projects/{project_id}", exc)


@router.post("/projects/{project_id}/shots/{shot_id}/retry-smoothing")
def retry_smoothing(request: Request, project_id: str, shot_id: str):
    try:
        shot = shot_view(_repository(request), project_id, shot_id)
        if not shot["can_retry_smoothing"]:
            raise InvalidStateError("Shot has no failed Smooth assembly to retry")
        request.app.state.process_manager.start(
            project_id, "smooth_postprocess", shot_id
        )
        return _redirect(f"/projects/{project_id}")
    except Exception as exc:
        return _error_redirect(f"/projects/{project_id}", exc)


@router.post("/projects/{project_id}/interrupt")
def interrupt_project(request: Request, project_id: str):
    try:
        request.app.state.process_manager.interrupt(project_id)
        return _redirect(f"/projects/{project_id}")
    except Exception as exc:
        return _error_redirect(f"/projects/{project_id}", exc)


@router.post("/projects/{project_id}/shots/{shot_id}/reset")
def reset_from_shot(request: Request, project_id: str, shot_id: str):
    try:
        if _repository(request).active_job(project_id):
            raise RuntimeError("Interrupt the active project before resetting a shot")
        _repository(request).reset_from_shot(project_id, shot_id)
        return _redirect(f"/projects/{project_id}")
    except Exception as exc:
        return _error_redirect(f"/projects/{project_id}", exc)


@router.get("/media/{project_id}/{media_path:path}")
def project_media(request: Request, project_id: str, media_path: str):
    root = _repository(request).project_dir(project_id).resolve()
    candidate = (root / media_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Media not found") from exc
    if candidate.suffix.lower() not in MEDIA_SUFFIXES or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(candidate)
