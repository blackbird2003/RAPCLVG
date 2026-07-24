#!/usr/bin/env python3
"""Submit a completed videogen_notebook project to the A6000 eval queue.

4090 side responsibility is intentionally small:
  1. validate a local videogen_notebook project is complete;
  2. export a self-contained EntityBench-style submission package;
  3. rsync it to A6000 and atomically enqueue one job JSON.

The A6000 worker owns queueing, execution, retries, status, logs, and reports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_REMOTE_HOST = "lzg@10.130.128.150"
DEFAULT_QUEUE_ROOT = "/data3/lzg/wxh/world_model_projects/EntityBench/eval_queue"
DEFAULT_REMOTE_ENTITYBENCH_ROOT = "/home/lzg/wxh/world_model_projects/EntityBench"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def slugify(value: str, max_len: int = 80) -> str:
    value = re.sub(r"\s+", "_", value.strip())
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return (value or "project")[:max_len]


def run(cmd: List[str], *, dry_run: bool = False) -> None:
    print("+ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def infer_tier(episode_id: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    lowered = episode_id.lower()
    if lowered.startswith("hard"):
        return "hard"
    if lowered.startswith(("mid", "medium")):
        return "medium"
    return "easy"


def source_script_from_project(project_dir: Path) -> Dict[str, Any]:
    story_path = project_dir / "story.json"
    if not story_path.exists():
        raise FileNotFoundError(f"Cannot find {story_path}; cannot build EntityBench eval script.")

    story = load_json(story_path)
    source = story.get("source")
    if not isinstance(source, dict):
        raise ValueError("story.json does not contain a source object; cannot build EntityBench eval script.")

    missing = [key for key in ["scenes", "entity_schedule", "entity_descriptions"] if key not in source]
    if not missing:
        return source
    raise ValueError(
        "story.json.source is missing fields required by EntityBench evaluation: "
        + ", ".join(missing)
    )


def load_asset_manifest(project_dir: Path) -> Dict[str, Dict[str, Any]]:
    manifest_path = project_dir / "assets" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing asset manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    assets = manifest.get("assets", manifest)
    if not isinstance(assets, dict):
        raise ValueError(f"Invalid asset manifest: {manifest_path}")
    return assets


def completed_shots(project_dir: Path) -> List[Dict[str, Any]]:
    shots = []
    for shot_json in sorted((project_dir / "shots").glob("*/shot.json")):
        shot = load_json(shot_json)
        state = shot.get("state", {})
        if state.get("status") != "completed":
            raise RuntimeError(f"Shot is not completed: {shot_json} state={state.get('status')}")
        shots.append(shot)
    shots.sort(key=lambda item: int(item.get("order_index", 0)))
    return shots


def raw_shot_video(project_dir: Path, assets: Dict[str, Dict[str, Any]], shot: Dict[str, Any]) -> Path:
    shot_id = str(shot["shot_id"])
    attempt_id = str(shot.get("state", {}).get("current_attempt_id") or "")
    exact = []
    fallback = []
    for asset_id, entry in assets.items():
        source = entry.get("source") or {}
        if entry.get("kind") != "video" or source.get("shot_id") != shot_id:
            continue
        if not str(source.get("type", "")).startswith("seedance"):
            continue
        bucket = exact if source.get("attempt_id") == attempt_id else fallback
        bucket.append((asset_id, entry))
    matches = exact or fallback
    if not matches:
        raise FileNotFoundError(f"Missing raw Seedance video asset for shot {shot_id}")
    matches.sort(key=lambda item: item[0])
    path = project_dir / matches[-1][1]["path"]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def full_video(project_dir: Path, assets: Dict[str, Dict[str, Any]], total_shots: int) -> Path:
    project = load_json(project_dir / "project.json")
    asset_id = project.get("current_final_video_asset_id")
    if asset_id and asset_id in assets:
        path = project_dir / assets[asset_id]["path"]
        if path.exists():
            return path

    candidates = sorted(
        (project_dir / "final").glob(f"prefix_{total_shots:04d}_*.mp4"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"Missing completed full video under {project_dir / 'final'}")
    return candidates[-1]


def validate_project_complete(project_dir: Path) -> Dict[str, Any]:
    project_path = project_dir / "project.json"
    if not project_path.exists():
        raise FileNotFoundError(f"Missing project.json: {project_path}")
    project = load_json(project_path)
    shot_count = int(project.get("shot_count") or 0)
    completed_prefix = int(project.get("completed_prefix") or 0)
    if project.get("status") != "completed" or completed_prefix != shot_count:
        raise RuntimeError(
            f"Project is not complete: status={project.get('status')} "
            f"completed_prefix={completed_prefix}/{shot_count}"
        )
    return project


def export_submission(
    *,
    project_dir: Path,
    stage_dir: Path,
    episode_id: str,
    method_name: str,
    tier: str,
    script: Dict[str, Any],
) -> Dict[str, Any]:
    project = validate_project_complete(project_dir)
    assets = load_asset_manifest(project_dir)
    shots = completed_shots(project_dir)
    if len(shots) != int(project.get("shot_count") or len(shots)):
        raise RuntimeError(f"Found {len(shots)} completed shot files, expected {project.get('shot_count')}")

    script = dict(script)
    script["story_name"] = episode_id

    scripts_dir = stage_dir / "scripts"
    results_dir = stage_dir / "results" / episode_id
    metadata_dir = stage_dir / "metadata"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    write_json(scripts_dir / f"{episode_id}.json", script)

    exported_shots = []
    for shot in shots:
        src = raw_shot_video(project_dir, assets, shot)
        scene_num = int(shot.get("scene_num") or 1)
        shot_num = int(shot.get("shot_num") or len(exported_shots) + 1)
        dst = results_dir / f"{scene_num:03d}_{shot_num:03d}.mp4"
        shutil.copy2(src, dst)
        exported_shots.append(
            {
                "shot_id": shot.get("shot_id"),
                "order_index": shot.get("order_index"),
                "scene_num": scene_num,
                "shot_num": shot_num,
                "source_video": str(src),
                "exported_video": str(dst.relative_to(stage_dir)),
            }
        )

    full_src = full_video(project_dir, assets, len(shots))
    shutil.copy2(full_src, results_dir / "full_concatenated.mp4")

    split = {"easy": [], "medium": [], "hard": []}
    split[tier].append(episode_id)
    write_json(metadata_dir / "split.json", split)

    export_manifest = {
        "exported_at": utc_now(),
        "project_dir": str(project_dir),
        "project_id": project.get("project_id"),
        "project_name": project.get("name"),
        "episode_id": episode_id,
        "tier": tier,
        "method_name": method_name,
        "shot_count": len(shots),
        "full_video": str((results_dir / "full_concatenated.mp4").relative_to(stage_dir)),
        "shots": exported_shots,
    }
    write_json(metadata_dir / "export_manifest.json", export_manifest)
    return export_manifest


def remote_path_join(*parts: str) -> str:
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def enqueue_remote_job(args: argparse.Namespace, job: Dict[str, Any], stage_dir: Path) -> None:
    remote_submission_dir = job["submission_dir"]
    queue_root = args.remote_queue_root.rstrip("/")
    remote_jobs = f"{queue_root}/jobs"
    remote_status = f"{queue_root}/status"
    remote_logs = f"{queue_root}/logs"
    remote_reports = f"{queue_root}/reports"
    remote_tmp_job = f"{remote_jobs}/{job['job_id']}.json.tmp"
    remote_job = f"{remote_jobs}/{job['job_id']}.json"

    run(
        [
            "ssh",
            args.remote_host,
            "mkdir",
            "-p",
            remote_submission_dir,
            remote_jobs,
            remote_status,
            remote_logs,
            remote_reports,
            f"{queue_root}/lock",
            f"{queue_root}/archive",
        ],
        dry_run=args.dry_run,
    )
    run(["rsync", "-a", "--delete", f"{stage_dir}/", f"{args.remote_host}:{remote_submission_dir}/"], dry_run=args.dry_run)

    with tempfile.TemporaryDirectory(prefix="videogen_eval_job_") as tmp:
        local_job = Path(tmp) / f"{job['job_id']}.json"
        write_json(local_job, job)
        run(["rsync", "-a", str(local_job), f"{args.remote_host}:{remote_tmp_job}"], dry_run=args.dry_run)
    run(["ssh", args.remote_host, "mv", remote_tmp_job, remote_job], dry_run=args.dry_run)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True, help="Completed videogen_notebook project directory")
    parser.add_argument("--episode-id", help="Episode id used by EntityBench. Default: sanitized project name.")
    parser.add_argument("--method-name", help="Method name used in report file. Default: sanitized project name.")
    parser.add_argument("--tier", choices=["easy", "medium", "hard"], help="EntityBench tier. Default inferred from episode id.")
    parser.add_argument("--pillars", default="1,2,3", help="EntityBench pillars, e.g. 1,2,3 or 2,3")
    parser.add_argument("--transition-boundary-source", choices=["shot", "full"], default="full")
    parser.add_argument("--llm-concurrency", type=int, default=5)
    parser.add_argument("--remote-host", default=os.environ.get("STORYMEM_EVAL_REMOTE_HOST", DEFAULT_REMOTE_HOST))
    parser.add_argument("--remote-queue-root", default=os.environ.get("STORYMEM_EVAL_QUEUE_ROOT", DEFAULT_QUEUE_ROOT))
    parser.add_argument("--remote-entitybench-root", default=os.environ.get("STORYMEM_REMOTE_ENTITYBENCH_ROOT", DEFAULT_REMOTE_ENTITYBENCH_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def submission_failure(project_dir: Path, exc: Exception) -> Dict[str, Any]:
    return {
        "submitted": False,
        "state": "failed",
        "failed_at": utc_now(),
        "project_dir": str(project_dir),
        "error": repr(exc),
        "note": (
            "Evaluation was not submitted. The 4090 submitter only reads EntityBench "
            "metadata from story.json.source; if required fields are absent, no A6000 "
            "queue job is created."
        ),
    }


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    project_dir = args.project_dir.resolve()
    project = validate_project_complete(project_dir)
    project_name = str(project.get("name") or project.get("project_id") or project_dir.name)
    episode_id = slugify(args.episode_id or project_name)
    method_name = slugify(args.method_name or project_name)
    tier = infer_tier(episode_id, args.tier)
    job_id = slugify(f"eval_{timestamp()}_{episode_id}_{method_name}", max_len=140)
    remote_submission_dir = f"{args.remote_queue_root.rstrip()}/submissions/{job_id}"

    script = source_script_from_project(project_dir)

    with tempfile.TemporaryDirectory(prefix="videogen_eval_submission_") as tmp:
        stage_dir = Path(tmp)
        export_manifest = export_submission(
            project_dir=project_dir,
            stage_dir=stage_dir,
            episode_id=episode_id,
            method_name=method_name,
            tier=tier,
            script=script,
        )
        job = {
            "job_id": job_id,
            "source": "videogen_notebook_4090",
            "created_at": utc_now(),
            "project_name": project_name,
            "project_id": project.get("project_id"),
            "episode_id": episode_id,
            "tier": tier,
            "method_name": method_name,
            "pillars": args.pillars,
            "transition_boundary_source": args.transition_boundary_source,
            "llm_concurrency": args.llm_concurrency,
            "remote_entitybench_root": args.remote_entitybench_root,
            "submission_dir": remote_submission_dir,
            "scripts_dir": "scripts",
            "results_dir": "results",
            "split_json": "metadata/split.json",
            "export_manifest": "metadata/export_manifest.json",
            "full_video_name": "full_concatenated.mp4",
        }
        write_json(stage_dir / "metadata" / "job.json", job)
        enqueue_remote_job(args, job, stage_dir)

    submitted = {
        "submitted": not args.dry_run,
        "state": "submitted" if not args.dry_run else "dry_run",
        "dry_run": args.dry_run,
        "job_id": job_id,
        "submitted_at": utc_now(),
        "remote_host": args.remote_host,
        "remote_queue_root": args.remote_queue_root,
        "remote_submission_dir": remote_submission_dir,
        "remote_job_file": f"{args.remote_queue_root.rstrip()}/jobs/{job_id}.json",
        "note": "A6000 manages evaluation status, execution logs, retries, and reports.",
        "export_manifest": export_manifest,
    }
    if not args.dry_run:
        write_json(project_dir / "eval_submitted.json", submitted)

    print(json.dumps(submitted, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        project_arg = None
        try:
            argv = sys.argv[1:]
            if "--project-dir" in argv:
                project_arg = Path(argv[argv.index("--project-dir") + 1]).resolve()
            elif any(arg.startswith("--project-dir=") for arg in argv):
                value = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--project-dir="))
                project_arg = Path(value).resolve()
            if project_arg is not None:
                write_json(project_arg / "eval_submitted.json", submission_failure(project_arg, exc))
        except Exception:
            pass
        print(f"[submit-error] {exc}", file=sys.stderr)
        raise
