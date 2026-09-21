from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # Loaded lazily in the video worker environment.
    cv2 = None  # type: ignore[assignment]


RIFE_REPO_REVISION = "17d8c7a1005b37f4c97bfee04e316aaec7fdc536"
RIFE_MODEL_VERSION = "4.25"
RIFE_FLOWNET_SHA256 = "6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2"
SMOOTH_TIMESTEPS = (1.0 / 3.0, 2.0 / 3.0)


class SmoothTransitionError(RuntimeError):
    pass


class FrameInterpolator(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    def interpolate(
        self, first: np.ndarray, second: np.ndarray, timesteps: Sequence[float]
    ) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frames: int
    width: int
    height: int


def padded_size(height: int, width: int, *, scale: float = 1.0) -> tuple[int, int]:
    tile = max(128, int(128 / scale))
    return (
        ((height - 1) // tile + 1) * tile,
        ((width - 1) // tile + 1) * tile,
    )


def crop_to_shape(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    return frame[:height, :width].copy()


class PracticalRifeInterpolator:
    def __init__(self, dependency_dir: str | Path | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self.dependency_dir = Path(
            dependency_dir or root / ".runtime/deps/Practical-RIFE"
        ).resolve()
        self._model: Any = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "implementation": "Practical-RIFE",
            "repo_revision": RIFE_REPO_REVISION,
            "model_version": RIFE_MODEL_VERSION,
            "flownet_sha256": RIFE_FLOWNET_SHA256,
            "scale": 1.0,
        }

    def interpolate(
        self, first: np.ndarray, second: np.ndarray, timesteps: Sequence[float]
    ) -> list[np.ndarray]:
        if first.shape != second.shape:
            raise SmoothTransitionError("Smooth anchors have different shapes")
        import torch
        import torch.nn.functional as functional

        model = self._load_model(torch)
        height, width = first.shape[:2]

        def tensor(frame: np.ndarray) -> Any:
            rgb = frame[:, :, ::-1].copy()
            value = (
                torch.from_numpy(np.transpose(rgb, (2, 0, 1)))
                .unsqueeze(0)
                .float()
                .cuda(non_blocking=True)
                / 255.0
            )
            padded_height, padded_width = padded_size(height, width)
            return functional.pad(
                value, (0, padded_width - width, 0, padded_height - height)
            )

        first_tensor = tensor(first)
        second_tensor = tensor(second)
        frames = []
        with torch.no_grad():
            for timestep in timesteps:
                output = model.inference(
                    first_tensor, second_tensor, timestep=float(timestep), scale=1.0
                )
                array = (
                    (output[0].clamp(0, 1) * 255.0)
                    .byte()
                    .detach()
                    .cpu()
                    .numpy()
                )
                bgr = np.transpose(array, (1, 2, 0))[:, :, ::-1].copy()
                frames.append(crop_to_shape(bgr, height, width))
        return frames

    def _load_model(self, torch: Any) -> Any:
        global _RIFE_MODEL
        if self._model is not None:
            return self._model
        if _RIFE_MODEL is not None:
            self._model = _RIFE_MODEL
            return self._model
        train_log = self.dependency_dir / "train_log"
        flownet = train_log / "flownet.pkl"
        if not flownet.is_file() or _sha256(flownet) != RIFE_FLOWNET_SHA256:
            raise SmoothTransitionError("Validated Practical-RIFE 4.25 weights are unavailable")
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(self.dependency_dir), "rev-parse", "HEAD"],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SmoothTransitionError("Cannot verify Practical-RIFE revision") from exc
        if revision != RIFE_REPO_REVISION:
            raise SmoothTransitionError(f"Unexpected Practical-RIFE revision: {revision}")
        if not torch.cuda.is_available():
            raise SmoothTransitionError("CUDA is required for Smooth RIFE assembly")
        if str(self.dependency_dir) not in sys.path:
            sys.path.insert(0, str(self.dependency_dir))
        from train_log.RIFE_HDv3 import Model

        torch.backends.cudnn.benchmark = True
        torch.set_grad_enabled(False)
        model = Model()
        model.load_model(str(train_log), -1)
        model.eval()
        model.device()
        _RIFE_MODEL = model
        self._model = model
        return model


_RIFE_MODEL: Any = None


class SmoothVideoAssembler:
    def __init__(self, interpolator: FrameInterpolator | None = None) -> None:
        self.interpolator = interpolator or PracticalRifeInterpolator()

    def __call__(self, clips: Sequence[dict[str, Any]], output_path: str) -> str:
        _require_cv2()
        if not clips:
            raise SmoothTransitionError("No clips available for final assembly")
        metadata = [_video_metadata(Path(item["output_video"])) for item in clips]
        _validate_compatible(metadata)
        transitions: dict[int, list[np.ndarray]] = {}
        for index in range(1, len(clips)):
            if clips[index].get("generation_mode") == "smooth":
                transitions[index] = self._transition_frames(
                    clips[index - 1], clips[index], metadata[index - 1], metadata[index]
                )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
        expected_frames = sum(item.frames for item in metadata)
        writer: _RawVideoWriter | None = None
        written = 0
        try:
            writer = _RawVideoWriter(
                temporary,
                fps=_stable_output_fps(metadata),
                width=metadata[0].width,
                height=metadata[0].height,
            )
            for index, (clip, meta) in enumerate(zip(clips, metadata)):
                start = 1 if clip.get("generation_mode") == "smooth" else 0
                end = meta.frames - 1 if index + 1 in transitions else meta.frames
                for frame in _iter_frames(Path(clip["output_video"]), start, end):
                    writer.write(frame)
                    written += 1
                if index + 1 in transitions:
                    for frame in transitions[index + 1]:
                        writer.write(frame)
                        written += 1
            writer.close()
            if written != expected_frames:
                raise SmoothTransitionError(
                    f"Smooth assembly changed frame count: {written} != {expected_frames}"
                )
            result = _video_metadata(temporary)
            if result.frames != expected_frames:
                raise SmoothTransitionError(
                    f"Encoded Smooth assembly changed frame count: {result.frames} != {expected_frames}"
                )
            _mux_concatenated_audio(
                temporary,
                [Path(item["output_video"]) for item in clips],
            )
            temporary.replace(output)
        except Exception as exc:
            if writer is not None:
                writer.abort()
            temporary.unlink(missing_ok=True)
            if isinstance(exc, SmoothTransitionError):
                raise
            raise SmoothTransitionError(str(exc)) from exc
        return str(output)

    def _transition_frames(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
        previous_meta: VideoMetadata,
        current_meta: VideoMetadata,
    ) -> list[np.ndarray]:
        if previous_meta.frames < 2 or current_meta.frames < 2:
            raise SmoothTransitionError("Smooth boundary requires at least two frames per clip")
        directory = Path(current["attempt_dir"]) / "smooth_transition"
        metadata_path = directory / "metadata.json"
        frame_paths = [directory / "t_033.png", directory / "t_067.png"]
        signature = {
            "source_attempt_ids": [previous["attempt_id"], current["attempt_id"]],
            "source_sha256": [
                _sha256(Path(previous["output_video"])),
                _sha256(Path(current["output_video"])),
            ],
            "anchors": {"left": "A[-2]", "right": "B[1]"},
            "timesteps": list(SMOOTH_TIMESTEPS),
            "model": self.interpolator.metadata,
        }
        existing = _read_json(metadata_path)
        if existing.get("status") == "completed" and existing.get("signature") == signature:
            cached = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in frame_paths]
            if all(frame is not None for frame in cached):
                return cached  # type: ignore[return-value]

        directory.mkdir(parents=True, exist_ok=True)
        try:
            left = _read_frame(Path(previous["output_video"]), previous_meta.frames - 2)
            right = _read_frame(Path(current["output_video"]), 1)
            frames = self.interpolator.interpolate(left, right, SMOOTH_TIMESTEPS)
            if len(frames) != 2:
                raise SmoothTransitionError("RIFE must return exactly two transition frames")
            for path, frame in zip(frame_paths, frames):
                if frame.shape != left.shape or not cv2.imwrite(str(path), frame):
                    raise SmoothTransitionError("Failed to persist a Smooth transition frame")
            _write_json(
                metadata_path,
                {
                    "status": "completed",
                    "signature": signature,
                    "source": {
                        "previous_video": str(previous["output_video"]),
                        "current_video": str(current["output_video"]),
                    },
                    "video": {
                        "fps": current_meta.fps,
                        "width": current_meta.width,
                        "height": current_meta.height,
                    },
                    "artifacts": [str(path) for path in frame_paths],
                    "error": None,
                },
            )
            return frames
        except Exception as exc:
            _write_json(
                metadata_path,
                {
                    "status": "failed",
                    "signature": signature,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
            if isinstance(exc, SmoothTransitionError):
                raise
            raise SmoothTransitionError(str(exc)) from exc


class _RawVideoWriter:
    def __init__(self, path: Path, *, fps: float, width: int, height: int) -> None:
        command = [
            _ffmpeg_executable(),
            "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise SmoothTransitionError("Smooth encoder input is unavailable")
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.wait() != 0:
            raise SmoothTransitionError("Failed to encode Smooth final assembly")

    def abort(self) -> None:
        if self.process.poll() is None:
            if self.process.stdin is not None:
                self.process.stdin.close()
            self.process.terminate()
            self.process.wait()


def _video_metadata(path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SmoothTransitionError(f"Cannot open raw video: {path}")
    try:
        result = VideoMetadata(
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()
    if result.fps <= 0 or result.frames < 1 or result.width < 1 or result.height < 1:
        raise SmoothTransitionError(f"Invalid raw video metadata: {path}")
    return result


def _validate_compatible(items: Sequence[VideoMetadata]) -> None:
    # first = items[0]
    # for item in items[1:]:
    #     if (
    #         abs(item.fps - first.fps) > 0.01
    #         or item.width != first.width
    #         or item.height != first.height
    #     ):
    #         raise SmoothTransitionError("Raw videos have incompatible frame rates or resolution")
    first = items[0]
    for item in items[1:]:
        if abs(item.fps - first.fps) > 0.01:
            # raise SmoothTransitionError("Raw videos have incompatible frame rates")
            # Do not raise yet; log a warning instead.
            logging.warning("Raw videos have incompatible frame rates: %.2f vs %.2f", item.fps, first.fps)
        if item.width != first.width or item.height != first.height:
            raise SmoothTransitionError("Raw videos have incompatible resolution")


def _stable_output_fps(items: Sequence[VideoMetadata]) -> float:
    if not items:
        raise SmoothTransitionError("No video metadata available")
    first = items[0].fps
    rounded = round(first)
    if rounded > 0 and abs(first - rounded) < 0.25:
        return float(rounded)
    return first



def _read_frame(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SmoothTransitionError(f"Cannot open raw video: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise SmoothTransitionError(f"Cannot read frame {index} from {path}")
    return frame


def _iter_frames(path: Path, start: int, end: int):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SmoothTransitionError(f"Cannot open raw video: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        for index in range(start, end):
            ok, frame = capture.read()
            if not ok:
                raise SmoothTransitionError(f"Cannot read frame {index} from {path}")
            yield frame
    finally:
        capture.release()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _require_cv2() -> Any:
    global cv2
    if cv2 is None:
        import importlib

        cv2 = importlib.import_module("cv2")
    return cv2


def _ffmpeg_executable() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        return ffmpeg
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _mux_concatenated_audio(video_path: Path, source_videos: Sequence[Path]) -> None:
    if not source_videos or not any(_video_has_audio_stream(path) for path in source_videos):
        return
    list_path = video_path.with_suffix(video_path.suffix + ".audio.concat.txt")
    muxed_path = video_path.with_suffix(video_path.suffix + ".muxed.mp4")
    _write_concat_list(source_videos, list_path)
    command = [
        _ffmpeg_executable(),
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        str(muxed_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    list_path.unlink(missing_ok=True)
    if result.returncode != 0:
        muxed_path.unlink(missing_ok=True)
        raise SmoothTransitionError(
            f"Failed to mux Smooth final audio: {(result.stderr or result.stdout).strip()}"
        )
    video_path.unlink(missing_ok=True)
    muxed_path.replace(video_path)


def _video_has_audio_stream(path: Path) -> bool:
    command = [_ffmpeg_executable(), "-hide_banner", "-i", str(path)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    stream_info = (result.stderr or "") + "\n" + (result.stdout or "")
    has_audio = "Audio:" in stream_info
    if not has_audio and result.returncode not in (0, 1):
        logging.warning("Could not inspect audio stream for %s", path)
    return has_audio


def _write_concat_list(paths: Sequence[Path], list_path: Path) -> None:
    lines = []
    for path in paths:
        escaped = str(path.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'\n")
    list_path.write_text("".join(lines), encoding="utf-8")
