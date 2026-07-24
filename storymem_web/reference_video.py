from __future__ import annotations

import atexit
import http.server
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, urlparse

import requests

try:
    import cv2
except ModuleNotFoundError:  # Loaded lazily in the video worker environment.
    cv2 = None  # type: ignore[assignment]


class ReferenceVideoPublisher(Protocol):
    def publish(self, path: str | Path) -> str: ...


def require_public_https_url(value: str) -> str:
    parsed = urlparse(str(value).strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError("Reference-video publisher must return a public HTTPS URL")
    return parsed.geturl()


def extract_video_tail(
    source: str | Path,
    output: str | Path,
    *,
    seconds: float = 1.0,
) -> str:
    _require_cv2()
    source_path = Path(source)
    output_path = Path(output)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open previous raw video: {source_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or frame_count < 1 or width < 1 or height < 1:
        raise RuntimeError(f"Invalid previous raw video metadata: {source_path}")

    tail_frames = min(frame_count, max(1, round(fps * seconds)))
    start_frame = frame_count - tail_frames
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    command = [
        _ffmpeg_executable(),
        "-y",
        "-v",
        "error",
        "-i",
        str(source_path),
        "-vf",
        f"trim=start_frame={start_frame}:end_frame={frame_count},setpts=PTS-STARTPTS",
        "-frames:v",
        str(tail_frames),
        "-r",
        f"{fps:g}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        import subprocess

        subprocess.run(command, check=True)
        _validate_tail(temporary, fps, tail_frames, width, height)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(output_path)


def _validate_tail(path: Path, fps: float, frames: int, width: int, height: int) -> None:
    _require_cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("Extracted Smooth reference clip is unreadable")
    try:
        actual = (
            float(capture.get(cv2.CAP_PROP_FPS)),
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()
    if abs(actual[0] - fps) > 0.01 or actual[1:] != (frames, width, height):
        raise RuntimeError(
            "Extracted Smooth reference clip changed source video metadata: "
            f"expected {(fps, frames, width, height)}, got {actual}"
        )


class TmpfilesPublisher:
    endpoint = "https://tmpfiles.org/api/v1/upload"

    def __init__(self, *, timeout: float = 60.0, retries: int = 10) -> None:
        self.timeout = timeout
        self.retries = retries

    def publish(self, path: str | Path) -> str:
        source = Path(path)
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with source.open("rb") as handle:
                    response = requests.post(
                        self.endpoint,
                        files={"file": (source.name, handle, "video/mp4")},
                        timeout=self.timeout,
                    )
                response.raise_for_status()
                return require_public_https_url(_tmpfiles_download_url(response.json()))
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 4))
        raise RuntimeError("Reference-video publication failed") from last_error


class CloudflareTunnelPublisher:
    def __init__(
        self,
        *,
        serve_dir: str | Path | None = None,
        cloudflared_bin: str | None = None,
        startup_timeout: float = 60.0,
    ) -> None:
        self.serve_dir = Path(serve_dir or os.getenv("STORYMEM_REFERENCE_VIDEO_SERVE_DIR") or Path(tempfile.gettempdir()) / "storymem_reference_video_tunnel")
        self.cloudflared_bin = cloudflared_bin or os.getenv("STORYMEM_CLOUDFLARED_BIN") or _cloudflared_executable()
        self.startup_timeout = startup_timeout
        self._lock = threading.Lock()
        self._server: http.server.ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._base_url: str | None = None

    def publish(self, path: str | Path) -> str:
        self._ensure_started()
        source = Path(path)
        self.serve_dir.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}{source.suffix or '.mp4'}"
        target = self.serve_dir / name
        shutil.copyfile(source, target)
        return require_public_https_url(f"{self._base_url}/{quote(name)}")

    def _ensure_started(self) -> None:
        with self._lock:
            if self._base_url and self._process and self._process.poll() is None:
                return
            self._start_local_server()
            self._start_tunnel()

    def _start_local_server(self) -> None:
        if self._server is not None:
            return
        self.serve_dir.mkdir(parents=True, exist_ok=True)
        serve_dir = self.serve_dir

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(serve_dir), **kwargs)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        atexit.register(self._shutdown)

    def _start_tunnel(self) -> None:
        assert self._server is not None
        port = int(self._server.server_address[1])
        self._process = subprocess.Popen(
            [
                self.cloudflared_bin,
                "tunnel",
                "--url",
                f"http://127.0.0.1:{port}",
                "--no-autoupdate",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        deadline = time.time() + self.startup_timeout
        lines: list[str] = []
        pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
        output: queue.Queue[str] = queue.Queue()
        assert self._process.stdout is not None

        def read_output() -> None:
            assert self._process is not None
            assert self._process.stdout is not None
            for item in self._process.stdout:
                output.put(item)

        threading.Thread(target=read_output, daemon=True).start()
        while time.time() < deadline:
            try:
                line = output.get(timeout=0.5)
                lines.append(line.rstrip())
                match = pattern.search(line)
                if match:
                    self._base_url = match.group(0)
                    return
            except queue.Empty:
                pass
            if self._process.poll() is not None:
                break
        tail = "\n".join(lines[-20:])
        raise RuntimeError(f"Cloudflare Tunnel did not produce a public URL. Recent output:\n{tail}")

    def _shutdown(self) -> None:
        process = self._process
        if process and process.poll() is None:
            process.terminate()
        server = self._server
        if server is not None:
            server.shutdown()


_CLOUDFLARE_TUNNEL_PUBLISHER: CloudflareTunnelPublisher | None = None


def configured_reference_video_publisher() -> ReferenceVideoPublisher:
    global _CLOUDFLARE_TUNNEL_PUBLISHER
    provider = os.getenv("STORYMEM_REFERENCE_VIDEO_PUBLISHER", "").strip().lower()
    if provider in {"cloudflare", "cloudflare_tunnel", "trycloudflare"}:
        if _CLOUDFLARE_TUNNEL_PUBLISHER is None:
            _CLOUDFLARE_TUNNEL_PUBLISHER = CloudflareTunnelPublisher(
                startup_timeout=float(os.getenv("STORYMEM_CLOUDFLARE_TUNNEL_TIMEOUT", "60")),
            )
        return _CLOUDFLARE_TUNNEL_PUBLISHER
    if provider == "tmpfiles":
        return TmpfilesPublisher(
            timeout=float(os.getenv("STORYMEM_REFERENCE_VIDEO_TIMEOUT", "60")),
            retries=int(os.getenv("STORYMEM_REFERENCE_VIDEO_RETRIES", "10")),
        )
    raise RuntimeError(
        "Smooth mode requires STORYMEM_REFERENCE_VIDEO_PUBLISHER=cloudflare_tunnel, tmpfiles, "
        "or an injected publisher"
    )


def _tmpfiles_download_url(payload: Any) -> str:
    raw = str(((payload or {}).get("data") or {}).get("url") or "").strip()
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or parsed.netloc != "tmpfiles.org":
        raise RuntimeError("Reference-video publisher returned an invalid URL")
    path = parsed.path if parsed.path.startswith("/dl/") else f"/dl{parsed.path}"
    return f"https://tmpfiles.org{path}"


def _require_cv2() -> Any:
    global cv2
    if cv2 is None:
        import importlib

        cv2 = importlib.import_module("cv2")
    return cv2


def _ffmpeg_executable() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _cloudflared_executable() -> str:
    for candidate in (
        os.getenv("STORYMEM_CLOUDFLARED_BIN"),
        shutil.which("cloudflared"),
        "/tmp/storymem_tools/cloudflared",
    ):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    target = Path(tempfile.gettempdir()) / "storymem_tools" / "cloudflared"
    target.parent.mkdir(parents=True, exist_ok=True)
    url = os.getenv(
        "STORYMEM_CLOUDFLARED_DOWNLOAD_URL",
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    )
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    target.write_bytes(response.content)
    target.chmod(0o755)
    return str(target)
