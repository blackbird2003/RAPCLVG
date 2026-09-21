from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Callable, Dict

from pipeline.runtime.project_store import InvalidProjectError
from pipeline.runtime.runner import VideoGenRunner


class BackgroundJobManager:
    def __init__(self, runner: VideoGenRunner, *, max_workers: int | None = None) -> None:
        self.runner = runner
        self.max_workers = max(1, int(max_workers or os.getenv("VIDEOGEN_MAX_WORKERS", "30")))
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="videogen")
        self._lock = Lock()
        self._jobs: Dict[str, Future] = {}

    def start_run_all(self, project_id: str) -> None:
        self._start(project_id, lambda: self.runner.run_all(project_id, apply_review_delay=True))

    def start_run_shot(self, project_id: str, shot_id: str) -> None:
        self._start(project_id, lambda: self.runner.run_shot(project_id, shot_id, apply_review_delay=True))

    def start_run_step(self, project_id: str, shot_id: str, step: str) -> None:
        self._start(project_id, lambda: self.runner.run_step(project_id, shot_id, step))

    def start_reflect_visual_plan(self, project_id: str, shot_id: str) -> None:
        self._start(project_id, lambda: self.runner.reflect_visual_plan(project_id, shot_id))

    def start_rerun_assembly(self, project_id: str) -> None:
        self._start(project_id, lambda: self.runner.rerun_assembly(project_id))

    def stop_project(self, project_id: str) -> None:
        self.runner.interrupt_project(project_id)
        with self._lock:
            future = self._jobs.get(project_id)
            if future is not None and not future.running():
                future.cancel()

    def _start(self, project_id: str, fn: Callable[[], object]) -> None:
        with self._lock:
            self._prune_locked()
            existing = self._jobs.get(project_id)
            if existing is not None and not existing.done():
                raise InvalidProjectError("This project is already running")
            future = self.executor.submit(fn)
            self._jobs[project_id] = future
            future.add_done_callback(lambda completed: self._finish(project_id, completed))

    def _finish(self, project_id: str, future: Future) -> None:
        try:
            future.exception()
        except Exception:
            pass
        with self._lock:
            if self._jobs.get(project_id) is future:
                self._jobs.pop(project_id, None)

    def _prune_locked(self) -> None:
        for project_id, future in list(self._jobs.items()):
            if future.done():
                self._jobs.pop(project_id, None)
