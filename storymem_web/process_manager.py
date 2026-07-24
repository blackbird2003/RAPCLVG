from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .repository import ProjectRepository


class ProcessManager:
    def __init__(self, repository: ProjectRepository, *, spawn_reaper: bool = True) -> None:
        self.repository = repository
        self.project_root = Path(__file__).resolve().parents[1]
        self.spawn_reaper = spawn_reaper
        self._processes: Dict[str, subprocess.Popen] = {}

    def start(self, project_id: str, mode: str, shot_id: Optional[str] = None) -> Dict[str, Any]:
        job = self.repository.create_job(project_id, mode, shot_id)
        launcher_log = self.repository.project_dir(project_id) / "worker.stdout.log"
        launcher_log.parent.mkdir(parents=True, exist_ok=True)
        worker_env = _worker_environment()
        try:
            with launcher_log.open("ab") as output:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "storymem_web.worker",
                        "--workspace",
                        str(self.repository.workspace),
                        "--job-id",
                        job["job_id"],
                    ],
                    cwd=self.project_root,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=worker_env,
                )
        except Exception:
            self.repository.update_job(job["job_id"], status="failed")
            raise
        updated = self.repository.update_job(job["job_id"], pid=process.pid)
        if self.spawn_reaper:
            self._processes[job["job_id"]] = process
            threading.Thread(
                target=self._reap,
                args=(job["job_id"], process),
                name=f"storymem-reaper-{job['job_id'][:8]}",
                daemon=True,
            ).start()
        return updated

    def interrupt(self, project_id: str) -> Optional[Dict[str, Any]]:
        job = self.repository.active_job(project_id)
        if job is None:
            return None
        return self.repository.request_job_cancel(job["job_id"])

    def reconcile(self) -> None:
        for job in self.repository.list_jobs(active_only=True):
            pid = job.get("pid")
            if pid and _process_exists(int(pid)):
                continue
            self.repository.update_job(job["job_id"], status="failed")
            self.repository.set_project_state(
                job["project_id"],
                "failed",
                active_shot_id=None,
                run_mode=None,
            )

    def _reap(self, job_id: str, process: subprocess.Popen) -> None:
        process.wait()
        self._processes.pop(job_id, None)
        try:
            job = self.repository.get_job(job_id)
        except Exception:
            return
        if job["status"] not in {"queued", "running"}:
            return
        self.repository.update_job(job_id, status="failed")
        self.repository.set_project_state(
            job["project_id"],
            "failed",
            active_shot_id=None,
            run_mode=None,
        )


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _worker_environment() -> Dict[str, str]:
    environment = os.environ.copy()
    thread_limit = environment.get("STORYMEM_WORKER_CPU_THREADS", "8")
    try:
        if int(thread_limit) < 1:
            raise ValueError
    except ValueError as exc:
        raise ValueError("STORYMEM_WORKER_CPU_THREADS must be a positive integer") from exc
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = thread_limit
    return environment
