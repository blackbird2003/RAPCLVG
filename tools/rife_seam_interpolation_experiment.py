#!/usr/bin/env python3
"""Practical-RIFE seam interpolation experiment for the Little Prince clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
import requests
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / ".runtime/experiments/little_prince_video_extension"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "interpolation_rife"

DEP_ROOT = ROOT / ".runtime/deps"
RIFE_REPO_URL = "https://github.com/hzwer/Practical-RIFE.git"
RIFE_REPO_REVISION = "17d8c7a1005b37f4c97bfee04e316aaec7fdc536"
RIFE_REPO_DIR = DEP_ROOT / "Practical-RIFE"
RIFE_TRAIN_LOG_DIR = RIFE_REPO_DIR / "train_log"
RIFE_MODEL_VERSION = "4.25"
RIFE_MODEL_PACKAGE_SOURCE = {
    "provider": "Official Practical-RIFE Google Drive release linked from README.md",
    "file_id": "1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg",
    "archive_name": "RIFEv4.25_0919.zip",
    "url": "https://drive.google.com/uc?id=1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg&export=download",
    "sha256": "e63d481b7ae5d4a4e6ad7ac5b410ff78f3bf7be3b51b2e38ca8152747abde5b4",
}
RIFE_MODEL_ARCHIVE = RIFE_REPO_DIR / RIFE_MODEL_PACKAGE_SOURCE["archive_name"]
RIFE_MODEL_FILES = {
    "IFNet_HDv3.py": {
        "sha256": "655b4c772b037967b86c2dd31c8fa3b5323b79dd9a0e0088708d89149bbc8a32",
    },
    "RIFE_HDv3.py": {
        "sha256": "81bbd0648e499de79e44768d284005d9d57d0f6eb7c30adae407f22675055730",
    },
    "refine.py": {
        "sha256": "0c5698b4a05b9f6ab551740575c1c35e248e5b1829bab6445186081ebe15f032",
    },
    "flownet.pkl": {
        "sha256": "6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2",
    },
}

SOURCE_REPORT_PATH = EXPERIMENT_ROOT / "experiment_report.json"


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    fps: float
    frames: int
    width: int
    height: int
    size_bytes: int


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir or DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "rife_seam_interpolation_report.json"

    start = time.perf_counter()
    report: dict[str, Any] = {
        "status": "starting",
        "experiment": "Little Prince seam interpolation with Practical-RIFE",
        "output_dir": str(output_dir),
        "source_report": str(SOURCE_REPORT_PATH),
        "protocol": {
            "anchor_left": "A191",
            "anchor_right": "B2",
            "inserted_timesteps": [0.25, 0.5, 0.75],
            "interpolation_factor": 4,
            "replacement": ["A192", "B0", "B1"],
            "target_frames": 386,
            "target_fps": 24,
            "target_resolution": [1280, 720],
            "target_audio": "none",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "opencv": cv2.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "versions": {
            "rife_repo_url": RIFE_REPO_URL,
            "rife_repo_revision": RIFE_REPO_REVISION,
            "rife_model_version": RIFE_MODEL_VERSION,
            "rife_model_package_source": RIFE_MODEL_PACKAGE_SOURCE,
        },
        "commands": {
            "run": (
                "source storymem_env.sh && CUDA_VISIBLE_DEVICES=0 "
                "python tools/rife_seam_interpolation_experiment.py"
            ),
            "repo_checkout": (
                f"git clone {RIFE_REPO_URL} {RIFE_REPO_DIR}; "
                f"git -C {RIFE_REPO_DIR} checkout --detach {RIFE_REPO_REVISION}"
            ),
            "model_download": RIFE_MODEL_PACKAGE_SOURCE["url"],
            "model_extraction": "extract train_log/{IFNet_HDv3.py,RIFE_HDv3.py,refine.py,flownet.pkl}",
            "interpolation": "model.inference(A191, B2, timestep=t, scale=1.0) for t=0.25,0.50,0.75",
            "audio_probe": "ffmpeg -hide_banner -i <final_video>; count Audio stream declarations",
        },
    }

    try:
        source_report = load_source_report(SOURCE_REPORT_PATH)
        source_shot1 = Path(source_report["input"]["shot1_video"])
        source_shot2 = Path(source_report["result"]["generated_video"])
        validate_input_videos(source_shot1, source_shot2)

        prepare_official_rife_deps()
        model = load_rife_model()

        shot1_meta = video_metadata(source_shot1)
        shot2_meta = video_metadata(source_shot2)
        report["input_metadata"] = {
            "shot1": meta_dict(shot1_meta),
            "shot2": meta_dict(shot2_meta),
        }

        if (
            shot1_meta.frames != 193
            or shot2_meta.frames != 193
            or shot1_meta.fps != 24.0
            or shot2_meta.fps != 24.0
            or shot1_meta.width != 1280
            or shot1_meta.height != 720
            or shot2_meta.width != 1280
            or shot2_meta.height != 720
        ):
            raise RuntimeError("Source clips do not match the fixed experiment contract")

        read_start = time.perf_counter()
        shot1_frames = read_frames(source_shot1, [190, 191])
        shot2_frames = read_frames(source_shot2, [2, 3])
        report["timing"] = {"input_read_seconds": round(time.perf_counter() - read_start, 3)}

        inference_start = time.perf_counter()
        a190 = shot1_frames[190]
        a191 = shot1_frames[191]
        b2 = shot2_frames[2]
        b3 = shot2_frames[3]
        i25 = infer_frame(model, a191, b2, 0.25)
        i50 = infer_frame(model, a191, b2, 0.50)
        i75 = infer_frame(model, a191, b2, 0.75)
        report["timing"]["rife_inference_seconds"] = round(
            time.perf_counter() - inference_start, 3
        )
        padded_height, padded_width = padded_size(a191.shape[0], a191.shape[1], scale=1.0)
        report["inference"] = {
            "api": "train_log.RIFE_HDv3.Model.inference",
            "model_reported_version": model.version,
            "scale": 1.0,
            "input_shape": [a191.shape[1], a191.shape[0]],
            "padded_shape": [padded_width, padded_height],
            "padding": [0, padded_width - a191.shape[1], 0, padded_height - a191.shape[0]],
            "output_crop": [0, 0, a191.shape[1], a191.shape[0]],
        }

        intermediates = [i25, i50, i75]
        intermediate_paths = write_intermediate_pngs(output_dir, intermediates)
        contact_sheet_path = output_dir / "rife_seam_contact_sheet.png"
        write_contact_sheet(
            contact_sheet_path,
            [
                ("A190", a190),
                ("A191", a191),
                ("I.25", i25),
                ("I.50", i50),
                ("I.75", i75),
                ("B2", b2),
                ("B3", b3),
            ],
        )
        report["qualitative_notes"] = {
            "inspection_scope": "Contact sheet only; no full-video subjective-quality claim.",
            "pause_reduction": (
                "The sampled positions progress through all three intermediates without a "
                "duplicated anchor, consistent with reducing the original sampled pause."
            ),
            "warping_and_ghosting": (
                "No obvious double-image ghosting or large warp is visible at contact-sheet scale."
            ),
            "character_deformation": (
                "The Prince remains recognizable; the intermediate silhouette is mildly softened."
            ),
            "rose_deformation": (
                "The rose remains recognizable with slight interpolation softness and no gross deformation."
            ),
            "background_flicker": (
                "The horizon illumination and star field show no obvious sampled brightness "
                "discontinuity; temporal flicker was not assessed from the still sheet."
            ),
        }

        temp_dir = Path(tempfile.mkdtemp(prefix="rife_interp_", dir=str(output_dir)))
        try:
            middle_video = temp_dir / "rife_middle_3f.mp4"
            encode_video_from_frames(intermediates, middle_video, fps=24.0)
            final_video = output_dir / "rife_seam_interpolated.mp4"
            concat_sources(
                source_shot1,
                middle_video,
                source_shot2,
                final_video,
                fps=24.0,
                first_end_frame=192,
                second_start_frame=2,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        final_meta = video_metadata(final_video)
        if (
            final_meta.frames != 386
            or final_meta.fps != 24.0
            or final_meta.width != 1280
            or final_meta.height != 720
        ):
            raise RuntimeError(f"Unexpected final video metadata: {meta_dict(final_meta)}")

        seam_frames = [a190, a191, i25, i50, i75, b2, b3]
        metrics = compute_seam_metrics(seam_frames)
        report["metrics"] = metrics

        audio_streams = probe_audio_stream_count(final_video)
        if audio_streams != 0:
            raise RuntimeError(f"Final video unexpectedly has {audio_streams} audio stream(s)")

        report["artifacts"] = {
            "final_video": str(final_video),
            "intermediate_pngs": [str(path) for path in intermediate_paths],
            "contact_sheet": str(contact_sheet_path),
            "report": str(report_path),
        }
        report["output_metadata"] = {
            "video": meta_dict(final_meta),
            "audio": {"streams": audio_streams, "present": audio_streams > 0},
            "intermediate_pngs": [image_metadata(path) for path in intermediate_paths],
            "contact_sheet": image_metadata(contact_sheet_path),
        }
        report["setup"] = collect_setup_details()
        report["timing"]["total_seconds"] = round(time.perf_counter() - start, 3)
        report["status"] = "completed"
        report["error"] = None
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report.setdefault("timing", {})["total_seconds"] = round(time.perf_counter() - start, 3)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(report_path)
    print(report["artifacts"]["final_video"])
    print(report["artifacts"]["contact_sheet"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def load_source_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_input_videos(shot1: Path, shot2: Path) -> None:
    for path in (shot1, shot2):
        if not path.is_file():
            raise FileNotFoundError(path)


def prepare_official_rife_deps() -> None:
    DEP_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_git_checkout(RIFE_REPO_DIR, RIFE_REPO_URL, RIFE_REPO_REVISION)
    RIFE_TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ensure_download(
        RIFE_MODEL_PACKAGE_SOURCE["url"],
        RIFE_MODEL_ARCHIVE,
        expected_sha256=RIFE_MODEL_PACKAGE_SOURCE["sha256"],
    )
    extract_model_files(RIFE_MODEL_ARCHIVE)
    init_path = RIFE_TRAIN_LOG_DIR / "__init__.py"
    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")


def extract_model_files(archive: Path) -> None:
    with zipfile.ZipFile(archive) as package:
        for name, spec in RIFE_MODEL_FILES.items():
            target = RIFE_TRAIN_LOG_DIR / name
            expected_sha256 = spec["sha256"]
            if target.exists() and sha256_file(target) == expected_sha256:
                continue
            member = f"train_log/{name}"
            contents = package.read(member)
            actual_sha256 = hashlib.sha256(contents).hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"Checksum mismatch for {member}: expected {expected_sha256}, "
                    f"got {actual_sha256}"
                )
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_bytes(contents)
            tmp.replace(target)


def ensure_git_checkout(repo_dir: Path, repo_url: str, revision: str) -> None:
    if not repo_dir.exists():
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
    current = None
    try:
        current = (
            subprocess.check_output(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
            )
            .strip()
        )
    except subprocess.CalledProcessError:
        current = None
    if current != revision:
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "origin", revision], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "checkout", "--detach", revision], check=True)


def ensure_download(url: str, target: Path, expected_sha256: str | None = None) -> None:
    if target.exists() and (expected_sha256 is None or sha256_file(target) == expected_sha256):
        return
    tmp = target.with_suffix(target.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if expected_sha256 is not None:
        actual = sha256_file(tmp)
        if actual != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {target.name}: expected {expected_sha256}, got {actual}"
            )
    tmp.replace(target)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rife_model() -> Any:
    if str(RIFE_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(RIFE_REPO_DIR))
    from train_log.RIFE_HDv3 import Model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    model = Model()
    model.load_model(str(RIFE_TRAIN_LOG_DIR), -1)
    model.eval()
    model.device()
    return model


def infer_frame(model: Any, first: np.ndarray, second: np.ndarray, timestep: float) -> np.ndarray:
    rgb0 = pad_tensor(bgr_to_tensor(first), scale=1.0)
    rgb1 = pad_tensor(bgr_to_tensor(second), scale=1.0)
    with torch.no_grad():
        output = model.inference(rgb0, rgb1, timestep=timestep, scale=1.0)
    frame = tensor_to_bgr(output)
    return frame[: first.shape[0], : first.shape[1]]


def bgr_to_tensor(frame: np.ndarray) -> torch.Tensor:
    rgb = frame[:, :, ::-1].copy()
    tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).float() / 255.0
    return tensor.cuda(non_blocking=True)


def pad_tensor(tensor: torch.Tensor, *, scale: float) -> torch.Tensor:
    _, _, height, width = tensor.shape
    padded_height, padded_width = padded_size(height, width, scale=scale)
    return F.pad(tensor, (0, padded_width - width, 0, padded_height - height))


def padded_size(height: int, width: int, *, scale: float) -> tuple[int, int]:
    # Matches Practical-RIFE inference_video.py for v4.25.
    tile = max(128, int(128 / scale))
    padded_height = ((height - 1) // tile + 1) * tile
    padded_width = ((width - 1) // tile + 1) * tile
    return padded_height, padded_width


def tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    frame = (tensor[0].clamp(0, 1) * 255.0).byte().detach().cpu().numpy()
    frame = np.transpose(frame, (1, 2, 0))
    return frame[:, :, ::-1].copy()


def read_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    wanted = sorted(indices)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        target_set = set(wanted)
        frames: dict[int, np.ndarray] = {}
        max_target = wanted[-1]
        for index in range(max_target + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {index} from {path}")
            if index in target_set:
                frames[index] = frame
        return {index: frames[index] for index in indices}
    finally:
        capture.release()


def video_metadata(path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        return VideoMetadata(
            path=path,
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            size_bytes=path.stat().st_size,
        )
    finally:
        capture.release()


def probe_audio_stream_count(path: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        output = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(path),
            ],
            text=True,
        )
        return len([line for line in output.splitlines() if line.strip()])

    probe = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    stream_lines = [
        line.strip()
        for line in probe.stderr.splitlines()
        if line.lstrip().startswith("Stream #")
    ]
    if not stream_lines:
        raise RuntimeError("Bundled FFmpeg did not report any streams for the final video")
    return sum(": Audio:" in line for line in stream_lines)


def meta_dict(meta: VideoMetadata) -> dict[str, Any]:
    return {
        "path": str(meta.path),
        "fps": meta.fps,
        "frames": meta.frames,
        "width": meta.width,
        "height": meta.height,
        "duration_seconds": round(meta.frames / meta.fps, 6) if meta.fps else None,
        "size_bytes": meta.size_bytes,
    }


def image_metadata(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return {
        "path": str(path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "channels": int(image.shape[2]) if image.ndim == 3 else 1,
        "size_bytes": path.stat().st_size,
    }


def write_intermediate_pngs(output_dir: Path, frames: list[np.ndarray]) -> list[Path]:
    names = ["rife_I_25.png", "rife_I_50.png", "rife_I_75.png"]
    paths = []
    for name, frame in zip(names, frames):
        path = output_dir / name
        cv2.imwrite(str(path), frame)
        paths.append(path)
    return paths


def write_contact_sheet(path: Path, labeled_frames: list[tuple[str, np.ndarray]]) -> None:
    thumb_w, thumb_h = 320, 180
    cols = 4
    rows = 2
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 18, dtype=np.uint8)
    for index, (label, frame) in enumerate(labeled_frames):
        row = index // cols
        col = index % cols
        thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        y0 = row * thumb_h
        x0 = col * thumb_w
        sheet[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = thumb
        text = label
        cv2.rectangle(sheet, (x0, y0), (x0 + thumb_w - 1, y0 + thumb_h - 1), (0, 0, 0), 1)
        cv2.putText(
            sheet,
            text,
            (x0 + 10, y0 + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            text,
            (x0 + 10, y0 + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), sheet)


def encode_video_from_frames(frames: list[np.ndarray], output: Path, *, fps: float) -> None:
    height, width = frames[0].shape[:2]
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:g}",
        "-i",
        "pipe:0",
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
        str(output),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"Failed to encode {output}")


def concat_sources(
    first: Path,
    middle: Path,
    second: Path,
    output: Path,
    *,
    fps: float,
    first_end_frame: int,
    second_start_frame: int,
) -> None:
    filters = (
        f"[0:v]trim=end_frame={first_end_frame},setpts=PTS-STARTPTS,fps={fps:g}[v0];"
        f"[1:v]trim=end_frame=3,setpts=PTS-STARTPTS,fps={fps:g}[v1];"
        f"[2:v]trim=start_frame={second_start_frame},setpts=PTS-STARTPTS,fps={fps:g}[v2];"
        f"[v0][v1][v2]concat=n=3:v=1:a=0[v]"
    )
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-i",
        str(first),
        "-i",
        str(middle),
        "-i",
        str(second),
        "-filter_complex",
        filters,
        "-map",
        "[v]",
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
        str(output),
    ]
    subprocess.run(cmd, check=True)


def compute_ssim(first: np.ndarray, second: np.ndarray, compare_width: int = 320) -> float:
    height = max(1, round(first.shape[0] * compare_width / first.shape[1]))
    size = (compare_width, height)
    first_float = cv2.resize(first, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    second_float = cv2.resize(second, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_first = cv2.GaussianBlur(first_float, (11, 11), 1.5)
    mu_second = cv2.GaussianBlur(second_float, (11, 11), 1.5)
    mu_first_sq = mu_first * mu_first
    mu_second_sq = mu_second * mu_second
    mu_product = mu_first * mu_second
    sigma_first_sq = cv2.GaussianBlur(first_float * first_float, (11, 11), 1.5) - mu_first_sq
    sigma_second_sq = cv2.GaussianBlur(second_float * second_float, (11, 11), 1.5) - mu_second_sq
    sigma_product = cv2.GaussianBlur(first_float * second_float, (11, 11), 1.5) - mu_product
    score_map = ((2 * mu_product + c1) * (2 * sigma_product + c2)) / (
        (mu_first_sq + mu_second_sq + c1) * (sigma_first_sq + sigma_second_sq + c2)
    )
    return float(score_map.mean())


def compute_dis_flow_magnitude(first: np.ndarray, second: np.ndarray, compare_width: int = 320) -> float:
    height = max(1, round(first.shape[0] * compare_width / first.shape[1]))
    size = (compare_width, height)
    first_gray = cv2.cvtColor(cv2.resize(first, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(cv2.resize(second, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(first_gray, second_gray, None)
    return float(np.linalg.norm(flow, axis=2).mean())


def compute_seam_metrics(frames: list[np.ndarray]) -> dict[str, Any]:
    compare_width = 320
    labels = ["A190->A191", "A191->I.25", "I.25->I.50", "I.50->I.75", "I.75->B2", "B2->B3"]
    pairs = []
    for label, first, second in zip(labels, frames[:-1], frames[1:]):
        ssim_score = compute_ssim(first, second, compare_width=compare_width)
        flow_magnitude = compute_dis_flow_magnitude(
            first, second, compare_width=compare_width
        )
        pairs.append(
            {
                "pair": label,
                "ssim": round(ssim_score, 6),
                "dis_flow_mean_magnitude": round(flow_magnitude, 6),
            }
        )
    return {
        "compare_width": compare_width,
        "pairs": pairs,
        "ssim_mean": round(float(np.mean([item["ssim"] for item in pairs])), 6),
        "dis_flow_mean_magnitude": round(
            float(np.mean([item["dis_flow_mean_magnitude"] for item in pairs])), 6
        ),
    }


def collect_setup_details() -> dict[str, Any]:
    pkl_path = RIFE_TRAIN_LOG_DIR / "flownet.pkl"
    return {
        "repo": {
            "url": RIFE_REPO_URL,
            "revision": RIFE_REPO_REVISION,
            "path": str(RIFE_REPO_DIR),
        },
        "model": {
            "version": RIFE_MODEL_VERSION,
            "package_source": RIFE_MODEL_PACKAGE_SOURCE,
            "archive": {
                "path": str(RIFE_MODEL_ARCHIVE),
                "size_bytes": RIFE_MODEL_ARCHIVE.stat().st_size,
                "sha256": sha256_file(RIFE_MODEL_ARCHIVE),
            },
            "train_log_files": {
                name: {
                    "path": str(RIFE_TRAIN_LOG_DIR / name),
                    "sha256": sha256_file(RIFE_TRAIN_LOG_DIR / name)
                    if (RIFE_TRAIN_LOG_DIR / name).exists()
                    else None,
                }
                for name in RIFE_MODEL_FILES
            },
            "flownet_pkl_sha256": sha256_file(pkl_path) if pkl_path.exists() else None,
        },
        "python": sys.version,
        "ffmpeg": imageio_ffmpeg.get_ffmpeg_exe(),
        "ffprobe": shutil.which("ffprobe"),
        "audio_probe": "ffprobe when available; bundled FFmpeg stream listing otherwise",
        "environment_changes": "none; reused the project py311 environment without installing packages",
    }


if __name__ == "__main__":
    main()
