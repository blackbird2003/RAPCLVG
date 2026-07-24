from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from .domain import StoryValidationError
from .jobs import BackgroundJobManager
from .project_store import InvalidProjectError, NotFoundError, ProjectStore
from .runner import create_runner


PACKAGE_DIR = Path(__file__).resolve().parent


def create_app(workspace: Optional[str] = None, runner_backend: Optional[str] = None) -> FastAPI:
    store = ProjectStore(workspace or os.getenv("VIDEOGEN_NOTEBOOK_WORKSPACE", ".runtime/videogen_notebook"))
    store.recover_interrupted_runs()
    runner = create_runner(
        store,
        runner_backend or os.getenv("VIDEOGEN_NOTEBOOK_RUNNER", "fake"),
    )
    app = FastAPI(title="videogen_notebook")
    app.state.store = store
    app.state.runner = runner
    app.state.jobs = BackgroundJobManager(runner)
    app.state.templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")
    app.state.templates.env.filters["element_names"] = _element_names
    app.state.templates.env.filters["logs_for"] = _logs_for
    app.state.templates.env.filters["pretty_json"] = _pretty_json
    app.state.templates.env.filters["step_elapsed"] = _step_elapsed
    app.state.templates.env.filters["running_like"] = _running_like
    app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")

    @app.get("/")
    async def projects(request: Request):
        projects = store.list_projects()
        return _render(
            request,
            "projects.html",
            {
                "projects": projects,
                "project_groups": _project_date_groups(projects),
                "default_settings": store.get_default_project_settings(),
            },
        )

    @app.post("/settings/defaults")
    async def save_default_settings(request: Request):
        form = await request.form()
        try:
            store.save_default_project_settings(_settings_update_from_form(form))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect("/")

    @app.post("/projects")
    async def create_project(name: str = Form("Untitled video project")):
        project = store.create_project(name)
        return _redirect(f"/projects/{project['project']['project_id']}")

    @app.post("/projects/import")
    async def import_project(story_file: list[UploadFile] = File(...)):
        try:
            for uploaded in story_file:
                payload = json.loads((await uploaded.read()).decode("utf-8"))
                store.import_story(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, StoryValidationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect("/")

    @app.get("/projects/{project_id}")
    async def project_page(project_id: str, request: Request):
        try:
            bundle = store.get_project(project_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _render(request, "project.html", bundle)

    @app.get("/projects/{project_id}/assets/{asset_id}")
    async def project_asset(project_id: str, asset_id: str):
        try:
            path = store.asset_path(project_id, asset_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidProjectError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FileResponse(path)

    @app.get("/projects/{project_id}/files/{relative_path:path}")
    async def project_file(project_id: str, relative_path: str):
        try:
            path = store.project_file_path(project_id, relative_path)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidProjectError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FileResponse(path)

    @app.post("/projects/{project_id}/save")
    async def save_project(request: Request, project_id: str):
        form = await request.form()
        try:
            bundle = store.get_project(project_id)
            shot_updates = []
            for shot in bundle["shots"]:
                shot_id = shot["shot_id"]
                shot_updates.append(
                    {
                        "shot_id": shot_id,
                        "video_prompt": form.get(f"shot-{shot_id}-prompt", ""),
                        "is_cut": form.get(f"shot-{shot_id}-is-cut") == "on",
                        "generation_mode": form.get(f"shot-{shot_id}-generation-mode", "default"),
                        "duration_seconds": form.get(f"shot-{shot_id}-duration", "-1"),
                        "predefined_references": _predefined_reference_rows_from_form(form, shot_id),
                    }
                )
            store.save_project(
                project_id,
                name=str(form.get("project-name", "")),
                settings_update=_settings_update_from_form(form),
                shot_updates=shot_updates,
            )
        except (NotFoundError, InvalidProjectError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect(f"/projects/{project_id}")

    @app.post("/projects/{project_id}/shots/add")
    async def add_shot(project_id: str):
        try:
            store.add_shot(project_id)
        except (NotFoundError, InvalidProjectError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect(f"/projects/{project_id}")

    @app.post("/projects/{project_id}/shots/{shot_id}/predefined-references/add")
    async def add_predefined_reference(
        project_id: str,
        shot_id: str,
        image_file: UploadFile = File(...),
        label: str = Form(""),
        guidance: str = Form(""),
    ):
        try:
            store.add_predefined_reference(
                project_id,
                shot_id,
                filename=image_file.filename or "reference.jpg",
                data=await image_file.read(),
                label=label,
                guidance=guidance,
            )
        except (NotFoundError, InvalidProjectError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect(f"/projects/{project_id}#{shot_id}")

    @app.get("/projects/{project_id}/story-design.json")
    async def export_story_design(project_id: str):
        try:
            payload = store.export_story_design(project_id)
        except (NotFoundError, InvalidProjectError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(
            payload,
            headers={"Content-Disposition": f'attachment; filename="{project_id}_story_design.json"'},
        )

    @app.post("/projects/{project_id}/duplicate")
    async def duplicate_project(project_id: str):
        try:
            project = store.duplicate_project(project_id)
        except (NotFoundError, InvalidProjectError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect(f"/projects/{project['project']['project_id']}")

    @app.post("/projects/{project_id}/delete")
    async def delete_project(project_id: str):
        try:
            store.delete_project(project_id)
        except (NotFoundError, InvalidProjectError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect("/")

    @app.post("/projects/{project_id}/run-all")
    async def run_all(project_id: str):
        try:
            app.state.jobs.start_run_all(project_id)
        except (NotFoundError, InvalidProjectError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect(f"/projects/{project_id}")

    @app.post("/projects/{project_id}/stop")
    async def stop_project(project_id: str):
        try:
            app.state.jobs.stop_project(project_id)
        except (NotFoundError, InvalidProjectError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect(f"/projects/{project_id}")

    @app.post("/projects/{project_id}/rerun-assembly")
    async def rerun_assembly(project_id: str):
        try:
            app.state.jobs.start_rerun_assembly(project_id)
        except (NotFoundError, InvalidProjectError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect(f"/projects/{project_id}")

    @app.post("/projects/{project_id}/shots/{shot_id}/run")
    async def run_shot(project_id: str, shot_id: str):
        try:
            app.state.jobs.start_run_shot(project_id, shot_id)
        except (NotFoundError, InvalidProjectError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect(f"/projects/{project_id}")

    @app.post("/projects/{project_id}/shots/{shot_id}/steps/{step}/run")
    async def run_step(project_id: str, shot_id: str, step: str):
        try:
            app.state.jobs.start_run_step(project_id, shot_id, step)
        except (NotFoundError, InvalidProjectError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect(f"/projects/{project_id}")

    @app.post("/projects/{project_id}/shots/{shot_id}/visual-plan/save")
    async def save_visual_plan(project_id: str, shot_id: str, request: Request):
        form = await request.form()
        try:
            store.save_visual_element_status(project_id, shot_id, _visual_element_rows_from_form(form))
        except (NotFoundError, InvalidProjectError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect(f"/projects/{project_id}#{shot_id}")

    @app.post("/projects/{project_id}/shots/{shot_id}/visual-plan/reflect")
    async def reflect_visual_plan(project_id: str, shot_id: str):
        try:
            app.state.jobs.start_reflect_visual_plan(project_id, shot_id)
        except (NotFoundError, InvalidProjectError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect(f"/projects/{project_id}#{shot_id}")

    @app.post("/projects/{project_id}/shots/{shot_id}/seedance-prompt/save")
    async def save_seedance_prompt(project_id: str, shot_id: str, request: Request):
        form = await request.form()
        try:
            store.save_seedance_prompt(project_id, shot_id, str(form.get("seedance-prompt", "")))
        except (NotFoundError, InvalidProjectError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect(f"/projects/{project_id}#{shot_id}")

    @app.post("/projects/{project_id}/shots/{shot_id}/references/save")
    async def save_selected_references(project_id: str, shot_id: str, request: Request):
        form = await request.form()
        try:
            store.save_selected_references(project_id, shot_id, _reference_rows_from_form(form))
        except (NotFoundError, InvalidProjectError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect(f"/projects/{project_id}#{shot_id}")

    @app.post("/projects/{project_id}/shots/{shot_id}/produced-memory/save")
    async def save_produced_memory(project_id: str, shot_id: str, request: Request):
        form = await request.form()
        try:
            store.save_produced_visual_memory(project_id, shot_id, _produced_memory_rows_from_form(form))
        except (NotFoundError, InvalidProjectError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect(f"/projects/{project_id}#{shot_id}")

    @app.post("/projects/{project_id}/shots/{shot_id}/reset-from")
    async def reset_from_shot(project_id: str, shot_id: str):
        try:
            store.reset_from_shot(project_id, shot_id)
        except (NotFoundError, InvalidProjectError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect(f"/projects/{project_id}")

    @app.post("/projects/{project_id}/shots/{shot_id}/fork-after")
    async def fork_after_shot(project_id: str, shot_id: str):
        try:
            project = store.fork_after_shot(project_id, shot_id)
        except (NotFoundError, InvalidProjectError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _redirect(f"/projects/{project['project']['project_id']}")

    return app


def _render(request: Request, template: str, context: dict):
    return request.app.state.templates.TemplateResponse(
        request,
        template,
        {"request": request, **context},
    )


def _project_date_groups(projects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    current_date: str | None = None
    for project in projects:
        date = _project_created_date(project)
        if date != current_date:
            groups.append({"date": date, "projects": []})
            current_date = date
        groups[-1]["projects"].append(project)
    return groups


def _project_created_date(project: dict[str, Any]) -> str:
    created_at = str(project.get("created_at") or "").strip()
    if len(created_at) >= 10:
        return created_at[:10]
    return "Unknown date"


def _redirect(path: str) -> RedirectResponse:
    return RedirectResponse(path, status_code=303)


def _element_names(element_ids, selection: dict | None = None) -> str:
    ids = [str(item) for item in (element_ids or []) if item]
    if not ids:
        return "none"
    by_id = {}
    if isinstance(selection, dict):
        for key in ("should_reference", "should_exclude", "optional_or_uncertain"):
            for element in selection.get(key) or []:
                if isinstance(element, dict) and element.get("id"):
                    by_id[str(element["id"])] = str(element.get("name") or element["id"])
    return ", ".join(by_id.get(element_id, element_id) for element_id in ids)


_LOG_GROUPS = {
    "visual_plan": {"planning_visual_elements", "visual_plan_reflection", "fake_visual_plan", "visual_plan_failed"},
    "reference_selection": {"selecting_visual_references", "reference_selection_failed"},
    "collection": {"planning_visual_elements", "selecting_visual_references", "fake_visual_plan"},
    "seedance_prompt": {"composing_prompt", "seedance_prompt_failed"},
    "seedance": {"submitting", "waiting_for_seedance", "downloading", "seedance_generation_failed"},
    "postprocess": {
        "waiting_for_gpu_keyframe_slot",
        "extracting_keyframes",
        "annotating_visual_memory",
        "fake_postprocess",
        "keyframe_maintaining_failed",
    },
}


def _logs_for(logs, group: str):
    steps = _LOG_GROUPS.get(str(group), set())
    if not steps:
        return list(logs or [])
    return [log for log in (logs or []) if isinstance(log, dict) and log.get("step") in steps]


def _predefined_reference_rows_from_form(form, shot_id: str) -> list[dict[str, Any]]:
    ids = list(form.getlist(f"shot-{shot_id}-predefined-id"))
    labels = list(form.getlist(f"shot-{shot_id}-predefined-label"))
    guidance_values = list(form.getlist(f"shot-{shot_id}-predefined-guidance"))
    keep_values = {str(value) for value in form.getlist(f"shot-{shot_id}-predefined-keep")}
    rows: list[dict[str, Any]] = []
    for index, reference_id in enumerate(ids):
        if str(reference_id) not in keep_values:
            continue
        rows.append(
            {
                "id": str(reference_id),
                "label": labels[index] if index < len(labels) else "",
                "guidance": guidance_values[index] if index < len(guidance_values) else "",
            }
        )
    return rows


def _running_like(status: object) -> bool:
    value = str(status or "")
    return value == "running" or value.endswith("_running") or value.endswith("_queued")


def _pretty_json(value) -> Markup:
    return Markup(escape(json.dumps(value, ensure_ascii=False, indent=2)))


def _step_elapsed(steps, step_name: str | None = None) -> str:
    total_seconds = 0
    measured = False
    step_values = (steps or {}).values()
    if step_name:
        step_values = [(steps or {}).get(step_name)]
    for step in step_values:
        if not isinstance(step, dict):
            continue
        started = _parse_datetime(step.get("started_at"))
        if started is None:
            continue
        ended = _parse_datetime(step.get("updated_at")) or datetime.now(timezone.utc)
        delta = max(0, int((ended - started).total_seconds()))
        status = str(step.get("status") or "")
        if status != "draft" or delta > 0:
            total_seconds += delta
            measured = True
    if not measured:
        return ""
    minutes, seconds = divmod(total_seconds, 60)
    if minutes:
        return f"in {minutes}min {seconds}s"
    return f"in {seconds}s"


def _parse_datetime(value) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _settings_update_from_form(form: Any) -> dict[str, Any]:
    return {
        "visual_element_memory": {
            "sink_frame_count": form.get("sink-frame-count", "0"),
            "max_retrieved_frames": form.get("max-retrieved-frames", "4"),
            "selection_mode": form.get("selection-mode", "greedy_coverage"),
            "scoring": {
                "character_weight": form.get("score-character-weight", "3.0"),
                "scene_weight": form.get("score-scene-weight", "2.0"),
                "object_weight": form.get("score-object-weight", "1.5"),
                "reference_uncovered_weight": form.get("score-reference-uncovered-weight", "1.0"),
                "reference_covered_weight": form.get("score-reference-covered-weight", "0.2"),
                "optional_weight": form.get("score-optional-weight", "0.1"),
                "exclude_weight": form.get("score-exclude-weight", "-0.1"),
                "quality_full_weight": form.get("score-quality-full-weight", "1.0"),
                "quality_partial_weight": form.get("score-quality-partial-weight", "0.2"),
                "quality_weak_weight": form.get("score-quality-weak-weight", "0.1"),
            },
        },
        "generation": {
            "default_duration_seconds": form.get("default-duration", "-1"),
            "default_non_cut_mode": form.get("default-non-cut-mode", "smooth"),
            "audio": form.get("audio") == "on",
            "auto_run_step_review_delay_seconds": form.get("auto-run-step-review-delay", "10"),
            "algorithm_step_max_attempts": form.get("algorithm-step-max-attempts", "5"),
            "smooth_reference_seconds": form.get("smooth-reference-seconds", "2"),
            "require_human_confirmation_before_seedance": form.get("require-seedance-confirmation") == "on",
            "auto_reflect_visual_plan": form.get("auto-reflect-visual-plan") == "on",
            "force_animation_style": form.get("force-animation-style") == "on",
        },
        "seedance": {
            "resolution": form.get("seedance-resolution", "720p"),
            "ratio": form.get("seedance-ratio", "16:9"),
        },
        "prompt_modules": {
            "full_script_context": form.get("prompt-module-full-script-context") == "on",
            "visual_element_plan": form.get("prompt-module-visual-element-plan") == "on",
            "holistic_guidance": form.get("prompt-module-holistic-guidance") == "on",
            "should_reference": form.get("prompt-module-should-reference") == "on",
            "should_exclude": form.get("prompt-module-should-exclude") == "on",
        },
        "evaluation": {
            "auto_submit_eval": form.get("auto-submit-eval") == "on",
        },
    }


def _visual_element_rows_from_form(form) -> list[dict]:
    tokens = form.getlist("element-token")
    ids = form.getlist("element-id")
    names = form.getlist("element-name")
    types = form.getlist("element-type")
    statuses = form.getlist("element-status")
    introduced = form.getlist("element-introduced")
    notes = form.getlist("element-notes")
    reasons = form.getlist("element-reason")
    rows = []
    for index, _token in enumerate(tokens):
        rows.append(
            {
                "element_id": _form_list_value(ids, index),
                "name": _form_list_value(names, index),
                "type": _form_list_value(types, index),
                "status": _form_list_value(statuses, index),
                "introduced_at": _form_list_value(introduced, index),
                "notes": _form_list_value(notes, index),
                "reason": _form_list_value(reasons, index),
            }
        )
    return rows


def _reference_rows_from_form(form) -> list[dict]:
    tokens = form.getlist("reference-token")
    ids = form.getlist("reference-id")
    keep_values = set(str(value) for value in form.getlist("reference-keep"))
    guidance = form.getlist("reference-guidance")
    rows = []
    for index, token in enumerate(tokens):
        token_value = str(token)
        rows.append(
            {
                "reference_id": _form_list_value(ids, index),
                "reference_guidance": _form_list_value(guidance, index),
                "keep": token_value in keep_values,
            }
        )
    return rows


def _produced_memory_rows_from_form(form) -> list[dict]:
    tokens = form.getlist("memory-token")
    asset_ids = form.getlist("memory-asset-id")
    keep_values = set(str(value) for value in form.getlist("memory-keep"))
    descriptions = form.getlist("memory-holistic-description")
    rows = []
    for index, token in enumerate(tokens):
        token_value = str(token)
        rows.append(
            {
                "asset_id": _form_list_value(asset_ids, index),
                "holistic_description": _form_list_value(descriptions, index),
                "keep": token_value in keep_values,
            }
        )
    return rows


def _form_list_value(values, index: int) -> str:
    return str(values[index]) if index < len(values) else ""


app = create_app()
