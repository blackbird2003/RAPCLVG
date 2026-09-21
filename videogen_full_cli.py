#!/usr/bin/env python3
"""Run the full video generation pipeline from a story JSON file."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from pipeline.runtime.domain import (
    STEP_LABELS,
    STEP_SEQUENCE,
    default_settings,
    normalize_story,
    validate_settings,
)
from pipeline.runtime.json_store import read_json
from pipeline.runtime.project_store import ProjectStore
from pipeline.runtime.runner import create_runner


DEFAULT_WORKSPACE = Path(".runtime/videogen_ui")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a long video with the full pipeline. "
            "Real paid API submission is enabled by default."
        )
    )
    parser.add_argument(
        "story_json",
        nargs="?",
        type=Path,
        help="Story JSON to import. Omit when resuming with --project-id.",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        help="Partial or complete project settings JSON. Values override the Full defaults.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.getenv("VIDEOGEN_WORKSPACE", DEFAULT_WORKSPACE)),
        help="Project workspace (default: .runtime/videogen_ui).",
    )
    parser.add_argument("--name", help="Optional project name for a newly imported story.")
    parser.add_argument("--project-id", help="Resume an existing project instead of importing a story.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use the fake backend; no LLM, VLM, Seedance, or GPU work is performed.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Automatically approve Seedance when settings require human confirmation.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="Progress display refresh interval (default: 1.0).",
    )
    args = parser.parse_args(argv)
    if bool(args.story_json) == bool(args.project_id):
        parser.error("provide exactly one of story_json or --project-id")
    if args.name and args.project_id:
        parser.error("--name is only valid when importing story_json")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return args


def full_settings(settings_path: Path | None = None) -> dict[str, Any]:
    settings = default_settings()
    if settings_path is not None:
        path = settings_path.expanduser().resolve()
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Settings file must contain a JSON object: {path}")
        settings = _deep_merge(settings, payload)
    return validate_settings(settings)


def main(argv: list[str] | None = None, *, stream: TextIO = sys.stdout) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    store = ProjectStore(workspace)

    try:
        if args.project_id:
            project_id = str(args.project_id)
            bundle = store.get_project(project_id)
            if args.settings:
                settings_update = full_settings(args.settings)
                bundle = store.save_project(project_id, settings_update=settings_update)
        else:
            story_path = args.story_json.expanduser().resolve()
            story = _load_story(story_path)
            settings = full_settings(args.settings)
            import_payload = _materialize_story_settings(story, story_path.parent, settings)
            bundle = store.import_story(import_payload, name=args.name)
            project_id = bundle["project"]["project_id"]
            bundle = store.save_project(project_id, settings_update=settings)
    except Exception as exc:
        _print(stream, f"Error: {type(exc).__name__}: {exc}")
        return 2

    project = bundle["project"]
    settings = bundle["settings"]
    _configure_runtime_environment()
    _print_header(stream, project, bundle, workspace, args.dry_run)
    _print_settings(stream, settings, args.settings)
    if not args.dry_run and not (os.getenv("SEEDANCE_API_KEY") or os.getenv("ARK_API_KEY")):
        _print(
            stream,
            "Error: SEEDANCE_API_KEY or ARK_API_KEY is required. "
            "Set it in the environment or provide ~/.seedance_api_key.",
        )
        return 2

    runner = create_runner(store, "fake" if args.dry_run else "real")
    reporter = ProgressReporter(
        store,
        project_id,
        stream=stream,
        poll_seconds=args.poll_seconds,
    )
    reporter.start()
    exit_code = 0
    try:
        while True:
            bundle = runner.run_all(
                project_id,
                apply_review_delay=not args.dry_run,
            )
            run_state = bundle["project"].get("run_state") or {}
            if run_state.get("status") != "paused_for_confirmation":
                break
            shot_id = str(run_state.get("shot_id") or "")
            _print(stream, f"\nSeedance submission is waiting for confirmation at Shot {shot_id}.")
            if not _approve_seedance(args.yes, stream):
                _print(stream, f"Paused. Resume later with: python videogen_full_cli.py --project-id {project_id}")
                exit_code = 3
                break
            _print(stream, "Confirmed. Submitting the Seedance generation step.")
            runner.run_step(project_id, shot_id, "seedance_generation")
    except KeyboardInterrupt:
        _print(stream, "\nInterrupt requested. Saving the current project state...")
        try:
            runner.interrupt_project(project_id)
        except Exception:
            pass
        exit_code = 130
    except Exception as exc:
        _print(stream, f"\nPipeline failed: {type(exc).__name__}: {exc}")
        exit_code = 1
    finally:
        reporter.stop()

    try:
        final_bundle = store.get_project(project_id)
        if exit_code == 0 and final_bundle["project"].get("status") != "completed":
            exit_code = 1
        _print_results(stream, store, project_id, final_bundle, exit_code)
    except Exception as exc:
        _print(stream, f"Could not read final project state: {type(exc).__name__}: {exc}")
        return exit_code or 1
    return exit_code


class ProgressReporter:
    def __init__(
        self,
        store: ProjectStore,
        project_id: str,
        *,
        stream: TextIO,
        poll_seconds: float,
    ) -> None:
        self.store = store
        self.project_id = project_id
        self.stream = stream
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="videogen-cli-progress", daemon=True)
        self._step_statuses: dict[tuple[str, str], str] = {}
        self._shot_statuses: dict[str, str] = {}
        self._run_state_signature: tuple[Any, ...] | None = None
        self._print_lock = threading.Lock()

    def start(self) -> None:
        self._scan()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.poll_seconds * 2))
        self._scan()

    def _run(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            self._scan()

    def _scan(self) -> None:
        try:
            project = read_json(self.store.project_dir(self.project_id) / "project.json", {})
            shots = self.store.list_shots(self.project_id)
        except Exception:
            return
        total = len(shots)
        for index, shot in enumerate(shots, start=1):
            shot_id = str(shot["shot_id"])
            shot_status = str((shot.get("state") or {}).get("status") or "draft")
            previous_shot_status = self._shot_statuses.get(shot_id)
            self._shot_statuses[shot_id] = shot_status
            if shot_status == "completed" and previous_shot_status != "completed":
                self._emit(f"Shot {index:02d}/{total:02d} completed.")
            elif shot_status in {"failed", "interrupted"} and shot_status != previous_shot_status:
                self._emit(f"Shot {index:02d}/{total:02d} {shot_status}.")

            for step in STEP_SEQUENCE:
                status = str((shot.get("steps") or {}).get(step, {}).get("status") or "draft")
                key = (shot_id, step)
                previous = self._step_statuses.get(key)
                self._step_statuses[key] = status
                if status == previous or status == "draft":
                    continue
                display = _display_step_status(step, status)
                elapsed = _step_elapsed((shot.get("steps") or {}).get(step, {}))
                suffix = f" ({elapsed})" if elapsed and status.endswith("_completed") else ""
                self._emit(f"Shot {index:02d}/{total:02d} | {STEP_LABELS[step]} | {display}{suffix}")

        run_state = project.get("run_state") if isinstance(project, dict) else {}
        if not isinstance(run_state, dict):
            return
        signature = (
            run_state.get("status"),
            run_state.get("shot_id"),
            run_state.get("from_step"),
            run_state.get("to_step"),
            run_state.get("attempt"),
            run_state.get("deadline_at"),
        )
        if signature == self._run_state_signature:
            return
        self._run_state_signature = signature
        if run_state.get("status") == "waiting_next_step":
            from_label = STEP_LABELS.get(str(run_state.get("from_step")), str(run_state.get("from_step")))
            to_label = STEP_LABELS.get(str(run_state.get("to_step")), str(run_state.get("to_step")))
            delay = run_state.get("delay_seconds")
            self._emit(f"Review pause: {from_label} -> {to_label} ({delay}s).")
        elif run_state.get("status") == "waiting_retry":
            step = STEP_LABELS.get(str(run_state.get("step")), str(run_state.get("step")))
            self._emit(
                f"Retry wait: {step}, attempt {run_state.get('attempt')}/{run_state.get('total_attempts')} "
                f"after {run_state.get('delay_seconds')}s."
            )

    def _emit(self, message: str) -> None:
        with self._print_lock:
            _print(self.stream, f"[{time.strftime('%H:%M:%S')}] {message}")


def _load_story(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Story file must contain a JSON object: {path}")
    return payload


def _materialize_story_settings(
    story: dict[str, Any],
    story_dir: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    payload = deepcopy(story)
    for scene in payload.get("scenes") or []:
        references_by_shot = scene.get("predefined_references")
        if not isinstance(references_by_shot, list):
            continue
        for references in references_by_shot:
            if not isinstance(references, list):
                continue
            for reference in references:
                if not isinstance(reference, dict):
                    continue
                key = "image_path" if reference.get("image_path") else "path"
                raw_path = str(reference.get(key) or "").strip()
                if raw_path and not Path(raw_path).expanduser().is_absolute():
                    reference[key] = str((story_dir / raw_path).resolve())

    normalized = normalize_story(
        payload,
        default_duration=settings["generation"]["default_duration_seconds"],
        default_non_cut_mode=settings["generation"]["default_non_cut_mode"],
    )
    offset = 0
    for scene in payload["scenes"]:
        count = len(scene["video_prompts"])
        shots = normalized.shots[offset : offset + count]
        offset += count
        scene["cut"] = [shot.is_cut for shot in shots]
        scene["durations"] = [shot.duration_seconds for shot in shots]
        scene["generation_modes"] = [shot.generation_mode for shot in shots]
        scene["predefined_references"] = [
            [dict(reference) for reference in shot.predefined_references]
            for shot in shots
        ]
    return payload


def _configure_runtime_environment() -> None:
    os.environ.setdefault("VIDEOGEN_REFERENCE_VIDEO_PUBLISHER", "cloudflare_tunnel")
    if os.getenv("SEEDANCE_API_KEY") or os.getenv("ARK_API_KEY"):
        return
    key_path = Path.home() / ".seedance_api_key"
    if key_path.is_file():
        value = key_path.read_text(encoding="utf-8").strip()
        if value:
            os.environ["SEEDANCE_API_KEY"] = value


def _print_header(
    stream: TextIO,
    project: dict[str, Any],
    bundle: dict[str, Any],
    workspace: Path,
    dry_run: bool,
) -> None:
    _print(stream, "\nVideogen Full Pipeline")
    _print(stream, "=" * 72)
    _print(stream, f"Project:   {project.get('name')}")
    _print(stream, f"Project ID:{' ' if project.get('project_id') else ''}{project.get('project_id')}")
    _print(stream, f"Shots:     {len(bundle.get('shots') or [])}")
    _print(stream, f"Backend:   {'fake dry-run' if dry_run else 'real paid APIs'}")
    _print(stream, f"Workspace: {workspace}")
    _print(stream, "=" * 72)


def _print_settings(stream: TextIO, settings: dict[str, Any], source: Path | None) -> None:
    generation = settings["generation"]
    memory = settings["visual_element_memory"]
    seedance = settings["seedance"]
    _print(stream, f"Settings:  {source.expanduser().resolve() if source else 'built-in Full defaults'}")
    _print(
        stream,
        (
            f"Generation: duration={generation['default_duration_seconds']}, "
            f"non-cut={generation['default_non_cut_mode']}, audio={generation['audio']}, "
            f"review_delay={generation['auto_run_step_review_delay_seconds']}s"
        ),
    )
    _print(
        stream,
        (
            f"Retrieval: mode={memory['selection_mode']}, sink={memory['sink_frame_count']}, "
            f"max_history={memory['max_retrieved_frames']}"
        ),
    )
    _print(
        stream,
        f"Seedance:  model={seedance['model']}, resolution={seedance['resolution']}, ratio={seedance['ratio']}",
    )
    _print(stream, f"Prompt modules: {', '.join(key for key, enabled in settings['prompt_modules'].items() if enabled)}")
    _print(stream, "")


def _approve_seedance(auto_yes: bool, stream: TextIO) -> bool:
    if auto_yes:
        return True
    if not sys.stdin.isatty():
        return False
    _print(stream, "Type 'yes' to submit this paid Seedance task, or press Enter to pause.")
    return input("> ").strip().lower() in {"y", "yes"}


def _print_results(
    stream: TextIO,
    store: ProjectStore,
    project_id: str,
    bundle: dict[str, Any],
    exit_code: int,
) -> None:
    project = bundle["project"]
    project_dir = store.project_dir(project_id).resolve()
    final_asset_id = project.get("current_final_video_asset_id")
    final_path = None
    if final_asset_id:
        try:
            final_path = store.asset_path(project_id, str(final_asset_id))
        except Exception:
            final_path = None
    _print(stream, "\nResults")
    _print(stream, "-" * 72)
    _print(stream, f"Status:        {project.get('status')}")
    _print(stream, f"Completed:     {project.get('completed_prefix', 0)}/{project.get('shot_count', 0)} shots")
    _print(stream, f"Project path:  {project_dir}")
    _print(stream, f"Project state: {project_dir / 'project.json'}")
    if final_path:
        _print(stream, f"Final video:   {final_path}")
    else:
        _print(stream, "Final video:   not available")
    _print(stream, f"Web UI:        http://localhost:{os.getenv('VIDEOGEN_PORT', '7880')}/projects/{project_id}")
    if exit_code:
        failure = _failure_summary(bundle)
        if failure:
            _print(stream, f"Failure:       {failure}")


def _failure_summary(bundle: dict[str, Any]) -> str:
    for shot in bundle.get("shots") or []:
        status = str((shot.get("state") or {}).get("status") or "")
        if status not in {"failed", "interrupted"} and not status.endswith("_failed"):
            continue
        error = (shot.get("attempt") or {}).get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("type")
        else:
            message = error
        return f"Shot {shot.get('shot_id')} ({status}): {message or 'see Attempt details'}"
    return ""


def _display_step_status(step: str, status: str) -> str:
    prefix = f"{step}_"
    value = status[len(prefix) :] if status.startswith(prefix) else status
    return value.replace("_", " ").upper()


def _step_elapsed(step: dict[str, Any]) -> str:
    started = _parse_time(step.get("started_at"))
    updated = _parse_time(step.get("updated_at"))
    if started is None or updated is None or updated < started:
        return ""
    seconds = int((updated - started).total_seconds())
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _deep_merge(target: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(target)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _print(stream: TextIO, message: str) -> None:
    print(message, file=stream, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
