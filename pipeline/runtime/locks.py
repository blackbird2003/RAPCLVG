from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from .json_store import read_json, write_json_atomic
from .project_store import InvalidProjectError, _now


class GlobalRunLock:
    def __init__(self, locks_dir: Path) -> None:
        self.locks_dir = locks_dir

    def acquire(self, project_id: str) -> None:
        path = self._path(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = read_json(path, None)
        if isinstance(existing, dict):
            pid = existing.get("pid")
            if isinstance(pid, int) and _pid_exists(pid):
                raise InvalidProjectError(
                    f"This project is already running: {existing.get('project_id')}"
                )
            path.unlink(missing_ok=True)
        write_json_atomic(
            path,
            {
                "project_id": project_id,
                "pid": os.getpid(),
                "started_at": _now(),
                "heartbeat_at": _now(),
            },
        )

    def release(self, project_id: str) -> None:
        path = self._path(project_id)
        existing = read_json(path, None)
        if isinstance(existing, dict) and existing.get("project_id") == project_id:
            path.unlink(missing_ok=True)

    def read(self) -> Dict[str, Any] | None:
        existing = read_json(self.locks_dir / "global_run.lock", None)
        return existing if isinstance(existing, dict) else None

    def _path(self, project_id: str) -> Path:
        safe_project_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in project_id)
        return self.locks_dir / f"project_{safe_project_id}.lock"


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
