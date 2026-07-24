import json
import logging
import os
import time
import urllib.request
from typing import Any, Callable, Dict, Iterable, Optional

import requests
from requests import RequestException


DEFAULT_API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = "doubao-seedance-2-0-260128"
TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class SeedanceError(RuntimeError):
    pass


class SeedanceCancelledError(SeedanceError):
    pass


class SeedanceClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = DEFAULT_API_BASE,
        model: str = DEFAULT_MODEL,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key or os.getenv("SEEDANCE_API_KEY") or os.getenv("ARK_API_KEY")
        if self.api_key:
            self.api_key = self.api_key.strip().lstrip("\ufeff")
        if not self.api_key:
            raise SeedanceError("SEEDANCE_API_KEY or ARK_API_KEY is required")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def create_task(
        self,
        content: Iterable[Dict[str, Any]],
        duration: int = 4,
        ratio: str = "16:9",
        resolution: str = "720p",
        generate_audio: bool = True,
        watermark: bool = False,
        return_last_frame: bool = True,
        execution_expires_after: int = 86400,
        callback_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "content": list(content),
            "resolution": resolution,
            "generate_audio": generate_audio,
            "ratio": ratio,
            "duration": duration,
            "watermark": watermark,
            "return_last_frame": return_last_frame,
            "execution_expires_after": execution_expires_after,
        }
        if callback_url:
            payload["callback_url"] = callback_url
        response = self._request_with_retries(
            "post",
            f"{self.api_base}/contents/generations/tasks",
            json=payload,
            retries=0,
        )
        return self._checked_json(response, "create_task")

    def get_task(self, task_id: str) -> Dict[str, Any]:
        response = self._request_with_retries(
            "get",
            f"{self.api_base}/contents/generations/tasks/{task_id}",
            retries=5,
        )
        return self._checked_json(response, "get_task")

    def wait_task(
        self,
        task_id: str,
        poll_interval: int = 10,
        max_wait_seconds: int = 1800,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        deadline = time.time() + max_wait_seconds
        last = None
        last_logged_status = None
        last_log_time = 0.0
        while time.time() < deadline:
            if should_cancel and should_cancel():
                raise SeedanceCancelledError(f"Waiting for Seedance task {task_id} was interrupted")
            last = self.get_task(task_id)
            status = str(_first_key(last, ("status", "state", "task_status")) or "").lower()
            now = time.time()
            if status != last_logged_status or now - last_log_time >= 60:
                logging.info("Seedance task %s status=%s", task_id, status or "unknown")
                last_logged_status = status
                last_log_time = now
            if status in {"succeeded", "success", "completed", "done"}:
                return last
            if status in {"failed", "error", "cancelled", "canceled"}:
                raise SeedanceError(f"Seedance task {task_id} failed: {json.dumps(last, ensure_ascii=False)}")
            _sleep_with_cancel(poll_interval, should_cancel, task_id)
        raise TimeoutError(f"Seedance task {task_id} timed out. Last response: {last}")

    def download_video(self, url: str, output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        last_error = None
        for attempt in range(1, 6):
            try:
                urllib.request.urlretrieve(url, output_path)
                return
            except Exception as exc:
                last_error = exc
                if attempt == 5:
                    break
                time.sleep(min(2 ** attempt, 30))
        raise SeedanceError(f"download_video failed after retries: {last_error}") from last_error

    def _request_with_retries(self, method: str, url: str, retries: int, **kwargs) -> requests.Response:
        last_error = None
        for attempt in range(1, retries + 2):
            try:
                response = requests.request(
                    method,
                    url,
                    headers=self._headers,
                    timeout=self.timeout,
                    **kwargs,
                )
                if response.status_code not in TRANSIENT_STATUS_CODES:
                    return response
                last_error = SeedanceError(f"HTTP {response.status_code}: {response.text[:500]}")
            except RequestException as exc:
                last_error = exc
            if attempt <= retries:
                time.sleep(min(2 ** attempt, 30))
        if isinstance(last_error, Exception):
            raise last_error
        raise SeedanceError("Request failed for an unknown reason")

    @staticmethod
    def _checked_json(response: requests.Response, action: str) -> Dict[str, Any]:
        try:
            data = response.json()
        except Exception as exc:
            raise SeedanceError(
                f"{action} returned non-JSON HTTP {response.status_code}: {response.text[:1000]}"
            ) from exc
        if response.status_code >= 400:
            raise SeedanceError(
                f"{action} returned HTTP {response.status_code}: {json.dumps(data, ensure_ascii=False)}"
            )
        return data


def extract_task_id(data: Dict[str, Any]) -> str:
    task_id = _first_key(data, ("id", "task_id", "taskId"))
    if task_id is None and isinstance(data.get("data"), dict):
        task_id = _first_key(data["data"], ("id", "task_id", "taskId"))
    if not task_id:
        raise SeedanceError(f"Could not find task id in response: {json.dumps(data, ensure_ascii=False)}")
    return str(task_id)


def extract_video_url(data: Any) -> str:
    urls = []
    _collect_video_urls(data, urls)
    if not urls:
        raise SeedanceError(f"Could not find generated video URL in response: {json.dumps(data, ensure_ascii=False)}")
    return urls[0]


def extract_last_frame_url(data: Any) -> str:
    if isinstance(data, dict):
        for key, item in data.items():
            if key.lower() in {"last_frame_url", "lastframeurl"}:
                if isinstance(item, str) and item.startswith("http"):
                    return item
            try:
                return extract_last_frame_url(item)
            except SeedanceError:
                continue
    elif isinstance(data, list):
        for item in data:
            try:
                return extract_last_frame_url(item)
            except SeedanceError:
                continue
    raise SeedanceError("Could not find original last_frame_url in Seedance task response")


def _sleep_with_cancel(
    seconds: float,
    should_cancel: Optional[Callable[[], bool]],
    task_id: str,
) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if should_cancel and should_cancel():
            raise SeedanceCancelledError(f"Waiting for Seedance task {task_id} was interrupted")
        time.sleep(min(0.5, max(0.0, deadline - time.time())))


def _first_key(data: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _collect_video_urls(value: Any, out: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and item.startswith("http"):
                lower = item.lower()
                if "video" in key.lower() or ".mp4" in lower:
                    out.append(item)
            else:
                _collect_video_urls(item, out)
    elif isinstance(value, list):
        for item in value:
            _collect_video_urls(item, out)
