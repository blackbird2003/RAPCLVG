from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Callable, List, Tuple


class GpuLockCancelled(RuntimeError):
    pass


@dataclass
class GpuLease:
    _lock: "KeyframeGpuLock"
    token: str
    project_id: str
    shot_id: str
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._lock.release(self.token)

    def __enter__(self) -> "GpuLease":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


class KeyframeGpuLock:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._owner: str | None = None
        self._queue: List[Tuple[str, str, str]] = []

    def acquire(
        self,
        project_id: str,
        shot_id: str,
        should_cancel: Callable[[], bool] | None = None,
        *,
        poll_interval: float = 0.25,
    ) -> GpuLease:
        token = uuid.uuid4().hex
        should_cancel = should_cancel or (lambda: False)
        with self._condition:
            self._queue.append((token, project_id, shot_id))
            try:
                while True:
                    if should_cancel():
                        raise GpuLockCancelled("Keyframe GPU wait was interrupted")
                    is_front = bool(self._queue) and self._queue[0][0] == token
                    if is_front and self._owner is None:
                        self._owner = token
                        self._queue.pop(0)
                        return GpuLease(self, token, project_id, shot_id)
                    self._condition.wait(timeout=poll_interval)
            except BaseException:
                self._queue = [item for item in self._queue if item[0] != token]
                self._condition.notify_all()
                raise

    def release(self, token: str) -> None:
        with self._condition:
            if self._owner == token:
                self._owner = None
                self._condition.notify_all()

    def snapshot(self) -> dict:
        with self._condition:
            return {
                "owner": self._owner,
                "queue": [
                    {"project_id": project_id, "shot_id": shot_id}
                    for _, project_id, shot_id in self._queue
                ],
            }
