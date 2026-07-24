#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storymem_web.runtime import git_info, source_fingerprint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize StoryMem runtime state")
    parser.add_argument("--project", help="Project ID prefix or case-insensitive name fragment")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--web-port", type=int, default=int(os.getenv("STORYMEM_WEB_PORT", "7860")))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def default_workspace() -> Path:
    explicit = os.getenv("STORYMEM_WEB_WORKSPACE")
    if explicit:
        return Path(explicit).expanduser().resolve()
    data = os.getenv("STORYMEM_DATA")
    if data:
        return (Path(data).expanduser() / "web").resolve()
    return (ROOT / ".runtime" / "web").resolve()


def collect_status(args: argparse.Namespace) -> Dict[str, Any]:
    workspace = (args.workspace or default_workspace()).expanduser().resolve()
    current_git = git_info(ROOT)
    current_git["changes"] = _git_changes()
    health = _web_health(args.web_port)
    current_source = source_fingerprint(ROOT)
    running_source = health.get("source_fingerprint") if health.get("reachable") else None
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "workspace": str(workspace),
        "git": current_git,
        "source_fingerprint": current_source,
        "web": {
            **health,
            "version_match": running_source == current_source if running_source else None,
            **_pid_status(workspace),
            "log": str(workspace / "server.log"),
        },
        "secrets": {
            "seedance": bool(os.getenv("SEEDANCE_API_KEY") or (Path.home() / ".seedance_api_key").is_file()),
            "deepseek": bool(os.getenv("DEEPSEEK_API_KEY") or (Path.home() / ".deepseek_api_key").is_file()),
        },
        "gpu": _gpu_status(),
        "database": _database_status(workspace, args.project, args.limit),
    }


def _web_health(port: int) -> Dict[str, Any]:
    url = f"http://127.0.0.1:{port}/api/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
        return {"reachable": False, "url": url, "error": str(exc)}
    return {"reachable": True, "url": url, **payload}


def _pid_status(workspace: Path) -> Dict[str, Any]:
    pid_file = workspace / "storymem-web.pid"
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return {"pid_file": str(pid_file), "pid_file_pid": None, "pid_alive": False}
    try:
        os.kill(pid, 0)
        alive = True
    except (OSError, ProcessLookupError):
        alive = False
    return {"pid_file": str(pid_file), "pid_file_pid": pid, "pid_alive": alive}


def _database_status(workspace: Path, selector: Optional[str], limit: int) -> Dict[str, Any]:
    db_path = workspace / "storymem_web.sqlite3"
    if not db_path.is_file():
        return {"available": False, "path": str(db_path), "projects": []}
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
        connection.row_factory = sqlite3.Row
        projects = connection.execute(
            """
            SELECT p.project_id, p.name, p.status, p.active_shot_id, p.updated_at,
                   COUNT(s.shot_id) AS shot_count,
                   SUM(CASE WHEN s.state = 'completed' THEN 1 ELSE 0 END) AS completed_count
            FROM projects p
            LEFT JOIN shots s ON s.project_id = p.project_id
            GROUP BY p.project_id
            ORDER BY p.updated_at DESC
            LIMIT ?
            """,
            (max(1, limit),),
        ).fetchall()
        schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        active_jobs = [dict(row) for row in connection.execute(
            "SELECT * FROM jobs WHERE status IN ('queued', 'running') ORDER BY created_at DESC"
        ).fetchall()]
        selected = _select_project(connection, selector) if selector else None
        detail = _project_detail(connection, selected) if selected else None
        connection.close()
    except sqlite3.Error as exc:
        return {"available": False, "path": str(db_path), "error": str(exc), "projects": []}
    return {
        "available": True,
        "path": str(db_path),
        "schema_version": schema_version,
        "active_jobs": active_jobs,
        "projects": [_project_summary(row) for row in projects],
        "selected_project": detail,
        "selector": selector,
        "selector_matched": detail is not None if selector else None,
    }


def _select_project(connection: sqlite3.Connection, selector: str) -> Optional[sqlite3.Row]:
    value = selector.strip()
    return connection.execute(
        """
        SELECT * FROM projects
        WHERE project_id = ? OR project_id LIKE ? OR lower(name) LIKE lower(?)
        ORDER BY CASE WHEN project_id = ? THEN 0 ELSE 1 END, updated_at DESC
        LIMIT 1
        """,
        (value, f"{value}%", f"%{value}%", value),
    ).fetchone()


def _project_detail(connection: sqlite3.Connection, project: sqlite3.Row) -> Dict[str, Any]:
    shot_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(shots)").fetchall()
    }
    if "generation_mode" in shot_columns:
        generation_mode = "s.generation_mode"
    elif "last_frame_only" in shot_columns:
        generation_mode = (
            "CASE WHEN s.last_frame_only = 1 THEN 'last_frame_only' ELSE 'default' END"
        )
    else:
        generation_mode = "'default'"
    duration_seconds = (
        "s.duration_seconds" if "duration_seconds" in shot_columns else "8"
    )
    memory_sink = "s.memory_sink" if "memory_sink" in shot_columns else "1"
    memory_retrieve = (
        "s.memory_retrieve"
        if "memory_retrieve" in shot_columns
        else ("s.is_cut" if "is_cut" in shot_columns else "0")
    )
    memory_recent = "s.memory_recent" if "memory_recent" in shot_columns else "1"
    shots = connection.execute(
        f"""
        SELECT s.order_index, s.scene_num, s.shot_num, s.state,
               {generation_mode} AS generation_mode,
               {duration_seconds} AS duration_seconds,
               {memory_sink} AS memory_sink,
               {memory_retrieve} AS memory_retrieve,
               {memory_recent} AS memory_recent,
               s.current_attempt_id, a.status AS attempt_status, a.task_id,
               a.error_json, a.usage_json, a.updated_at AS attempt_updated_at
        FROM shots s
        LEFT JOIN attempts a ON a.attempt_id = s.current_attempt_id
        WHERE s.project_id = ?
        ORDER BY s.order_index
        """,
        (project["project_id"],),
    ).fetchall()
    return {
        "project_id": project["project_id"],
        "name": project["name"],
        "status": project["status"],
        "active_shot_id": project["active_shot_id"],
        "updated_at": project["updated_at"],
        "shots": [_shot_summary(row) for row in shots],
    }


def _project_summary(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "project_id": row["project_id"],
        "name": row["name"],
        "status": row["status"],
        "completed": int(row["completed_count"] or 0),
        "shots": int(row["shot_count"] or 0),
        "updated_at": row["updated_at"],
    }


def _shot_summary(row: sqlite3.Row) -> Dict[str, Any]:
    error = _json(row["error_json"])
    usage = _json(row["usage_json"])
    message = str(error.get("message") or "")
    return {
        "index": int(row["order_index"]) + 1,
        "scene": int(row["scene_num"]),
        "shot": int(row["shot_num"]),
        "state": row["state"],
        "generation_mode": row["generation_mode"],
        "duration_seconds": int(row["duration_seconds"]),
        "memory_policy": {
            "sink": bool(row["memory_sink"]),
            "retrieve": bool(row["memory_retrieve"]),
            "recent": bool(row["memory_recent"]),
        },
        "attempt_status": row["attempt_status"],
        "task_id": row["task_id"],
        "error_type": error.get("type"),
        "error": message[:240],
        "tokens": usage.get("total_tokens"),
        "updated_at": row["attempt_updated_at"],
    }


def _gpu_status() -> Dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc), "devices": []}
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip(), "devices": []}
    devices = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        devices.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
                "utilization_percent": int(parts[4]),
            }
        )
    return {"available": True, "devices": devices}


def _git_changes() -> List[str]:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return result.stdout.splitlines()[:20]


def _json(value: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def print_human(status: Dict[str, Any]) -> None:
    git = status["git"]
    print(f"StoryMem {git['git_branch']}@{git['git_commit_short']} source={status['source_fingerprint']}")
    print(f"Workspace: {status['workspace']}")
    web = status["web"]
    if web["reachable"]:
        match = "current" if web["version_match"] else "STALE"
        print(
            f"Web: up pid={web.get('pid')} port={web['url'].split(':')[2].split('/')[0]} "
            f"source={web.get('source_fingerprint')} ({match})"
        )
    else:
        print(f"Web: down/unhealthy ({web.get('error', 'unknown error')})")
    print(f"Secrets: Seedance={'yes' if status['secrets']['seedance'] else 'no'} "
          f"DeepSeek={'yes' if status['secrets']['deepseek'] else 'no'}")
    if status["gpu"]["available"]:
        for device in status["gpu"]["devices"]:
            print(
                f"GPU{device['index']}: {device['memory_used_mib']}/{device['memory_total_mib']} MiB "
                f"util={device['utilization_percent']}% {device['name']}"
            )
    database = status["database"]
    if not database["available"]:
        print(f"Database: unavailable ({database.get('error', database['path'])})")
        return
    print(f"Database: schema={database['schema_version']} active_jobs={len(database['active_jobs'])}")
    selected = database.get("selected_project")
    if selected:
        print(f"Project: {selected['name']} [{selected['project_id'][:8]}] {selected['status']}")
        for shot in selected["shots"]:
            suffix = f" error={shot['error_type']}: {shot['error']}" if shot["error_type"] else ""
            task = f" task={shot['task_id']}" if shot["task_id"] else ""
            print(
                f"  {shot['index']:02d} S{shot['scene']}/Shot{shot['shot']} {shot['state']} "
                f"mode={shot['generation_mode']} duration={shot['duration_seconds']}s "
                f"memory={int(shot['memory_policy']['sink'])}"
                f"{int(shot['memory_policy']['retrieve'])}"
                f"{int(shot['memory_policy']['recent'])}{task}{suffix}"
            )
    else:
        if database.get("selector") and not database.get("selector_matched"):
            print(f"Project selector did not match: {database['selector']}")
        for project in database["projects"]:
            print(
                f"  {project['project_id'][:8]} {project['status']:<11} "
                f"{project['completed']}/{project['shots']} {project['name']}"
            )
    if git["changes"]:
        print("Changes:")
        for line in git["changes"]:
            print(f"  {line}")


def main() -> None:
    args = parse_args()
    status = collect_status(args)
    if args.as_json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print_human(status)


if __name__ == "__main__":
    main()
