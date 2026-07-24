#!/usr/bin/env python3
"""FFmpeg seam interpolation experiment for the Little Prince extension seam."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / ".runtime/experiments/little_prince_video_extension/interpolation_ffmpeg"
DEFAULT_REPORT = ROOT / ".runtime/experiments/little_prince_video_extension/experiment_report.json"

FIRST_SEAM_INDEX = 191
SECOND_SEAM_INDEX = 2
FIRST_TAIL_END = 191
SECOND_HEAD_START = 3
FPS = 24.0
SEAM_INTERP_FPS = 96
OUTPUT_FRAME_COUNT = 386
COMPARE_WIDTH = 320
TARGET_SIZE = (1280, 720)


@dataclass(frozen=True)
class FrameMetric:
    left_label: str
    right_label: str
    ssim: float
    dis_flow_mean: float


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir or DEFAULT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    first_path, second_path = resolve_inputs(args)
    started = time.time()
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_version = run_capture([ffmpeg_exe, "-hide_banner", "-version"]).stdout
    filter_help = run_capture([ffmpeg_exe, "-hide_banner", "-h", "filter=minterpolate"]).stdout
    support = parse_minterpolate_support(filter_help)
    minterpolate_filter = build_minterpolate_filter(support)

    first_meta = video_metadata(first_path)
    second_meta = video_metadata(second_path)
    validate_inputs(first_meta, second_meta)

    with tempfile.TemporaryDirectory(prefix="ffmpeg_seam_", dir=output_dir) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        seam_frames_dir = temp_dir / "seam_frames"
        seam_frames_dir.mkdir(parents=True, exist_ok=True)
        seam_source = temp_dir / "seam_source.mp4"
        seam_frame_pattern = seam_frames_dir / "seam_%02d.png"
        output_video = output_dir / "ffmpeg_seam_interpolation.mp4"
        report_path = output_dir / "ffmpeg_seam_interpolation_report.json"
        contact_sheet_path = output_dir / "ffmpeg_seam_interpolation_contact_sheet.png"
        interpolated_paths = [
            output_dir / "ffmpeg_seam_interpolation_I25.png",
            output_dir / "ffmpeg_seam_interpolation_I50.png",
            output_dir / "ffmpeg_seam_interpolation_I75.png",
        ]

        seam_commands = {
            "make_seam_source": [
                ffmpeg_exe,
                "-hide_banner",
                "-y",
                "-v",
                "error",
                "-i",
                str(first_path),
                "-i",
                str(second_path),
                "-filter_complex",
                (
                    f"[0:v]trim=start_frame={FIRST_SEAM_INDEX}:end_frame={FIRST_SEAM_INDEX + 1},"
                    f"setpts=PTS-STARTPTS,fps={int(FPS)}[left];"
                    f"[1:v]trim=start_frame={SECOND_SEAM_INDEX}:end_frame={SECOND_SEAM_INDEX + 1},"
                    f"setpts=PTS-STARTPTS,fps={int(FPS)}[middle];"
                    f"[1:v]trim=start_frame={SECOND_SEAM_INDEX + 1}:end_frame={SECOND_SEAM_INDEX + 2},"
                    f"setpts=PTS-STARTPTS,fps={int(FPS)}[right];"
                    f"[left][middle][right]concat=n=3:v=1:a=0,setsar=1[seam]"
                ),
                "-map",
                "[seam]",
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
                "-frames:v",
                "3",
                str(seam_source),
            ],
            "interpolate_seam": [
                ffmpeg_exe,
                "-hide_banner",
                "-y",
                "-v",
                "error",
                "-i",
                str(seam_source),
                "-vf",
                f"minterpolate={minterpolate_filter}",
                "-an",
                "-vsync",
                "0",
                "-frames:v",
                "5",
                "-c:v",
                "png",
                str(seam_frame_pattern),
            ],
            "compose_final": [
                ffmpeg_exe,
                "-hide_banner",
                "-y",
                "-v",
                "error",
                "-i",
                str(first_path),
                "-framerate",
                str(int(FPS)),
                "-start_number",
                "1",
                "-i",
                str(seam_frames_dir / "seam_%02d.png"),
                "-i",
                str(second_path),
                "-filter_complex",
                (
                    f"[0:v]trim=end_frame={FIRST_TAIL_END},setpts=PTS-STARTPTS,fps={int(FPS)}[first];"
                    f"[1:v]setpts=PTS-STARTPTS,fps={int(FPS)}[seam];"
                    f"[2:v]trim=start_frame={SECOND_HEAD_START},setpts=PTS-STARTPTS,fps={int(FPS)}[second];"
                    f"[first][seam][second]concat=n=3:v=1:a=0,setsar=1[v]"
                ),
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
                str(output_video),
            ],
        }

        run(seam_commands["make_seam_source"])
        run(seam_commands["interpolate_seam"])
        seam_pngs = sorted(seam_frames_dir.glob("seam_*.png"))
        if len(seam_pngs) != 5:
            raise RuntimeError(f"Expected 5 seam frames, found {len(seam_pngs)}")

        seam_images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in seam_pngs]
        if any(image is None for image in seam_images):
            raise RuntimeError("Could not read one or more interpolated seam PNGs")
        if seam_images[0].shape[1::-1] != TARGET_SIZE:
            raise RuntimeError(f"Unexpected seam frame size: {seam_images[0].shape[1::-1]}")

        for index, src_path in enumerate(interpolated_paths, start=1):
            shutil.copy2(seam_pngs[index], src_path)

        run(seam_commands["compose_final"])

        output_meta = video_metadata(output_video)
        if output_meta["frames"] != OUTPUT_FRAME_COUNT:
            raise RuntimeError(
                f"Unexpected output frame count {output_meta['frames']} != {OUTPUT_FRAME_COUNT}"
            )
        contact_sheet_meta = image_metadata(contact_sheet_path)
        interpolated_meta = [image_metadata(path) for path in interpolated_paths]

        sampled_labels = [
            ("A190", first_path, 190),
            ("A191", first_path, 191),
            ("I25", interpolated_paths[0], 0),
            ("I50", interpolated_paths[1], 0),
            ("I75", interpolated_paths[2], 0),
            ("B2", second_path, 2),
            ("B3", second_path, 3),
        ]
        sampled_frames = [
            load_frame(source_path, frame_index) for _, source_path, frame_index in sampled_labels
        ]
        metrics = compute_metrics(sampled_labels, sampled_frames)
        contact_sheet = make_contact_sheet(sampled_labels, sampled_frames)
        cv2.imwrite(str(contact_sheet_path), contact_sheet)

        report = {
            "experiment": "ffmpeg seam interpolation",
            "protocol": {
                "left_anchor": "A191",
                "right_anchor": "B2",
                "replaced_frames": ["A192", "B0", "B1"],
                "interpolated_t": [0.25, 0.5, 0.75],
                "frame_count": OUTPUT_FRAME_COUNT,
                "fps": FPS,
                "resolution": {"width": TARGET_SIZE[0], "height": TARGET_SIZE[1]},
                "audio": "none",
            },
            "inputs": {
                "first": str(first_path),
                "second": str(second_path),
                "first_metadata": first_meta,
                "second_metadata": second_meta,
                "source_report": str(DEFAULT_REPORT),
            },
            "versions": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "opencv": cv2.__version__,
                "imageio_ffmpeg": getattr(imageio_ffmpeg, "__version__", "unknown"),
                "ffmpeg": summarize_version(ffmpeg_version),
                "minterpolate_support": support,
            },
            "commands": seam_commands,
            "outputs": {
                "video": str(output_video),
                "interpolated_pngs": [str(path) for path in interpolated_paths],
                "contact_sheet": str(contact_sheet_path),
            },
            "output_metadata": {
                "video": output_meta,
                "contact_sheet": contact_sheet_meta,
                "interpolated_pngs": interpolated_meta,
            },
            "frame_metadata": {
                "sampled": [
                    {
                        "label": label,
                        "source": str(source_path),
                        "frame_index": frame_index,
                    }
                    for label, source_path, frame_index in sampled_labels
                ]
            },
            "metrics": {
                "ssim": {
                    "compare_width": COMPARE_WIDTH,
                    "pairs": [
                        {
                            "transition": f"{metric.left_label}->{metric.right_label}",
                            "value": round(metric.ssim, 6),
                        }
                        for metric in metrics
                    ],
                    "mean": round(float(np.mean([metric.ssim for metric in metrics])), 6),
                },
                "dis_flow": {
                    "pairs": [
                        {
                            "transition": f"{metric.left_label}->{metric.right_label}",
                            "mean_magnitude": round(metric.dis_flow_mean, 6),
                        }
                        for metric in metrics
                    ],
                    "mean": round(float(np.mean([metric.dis_flow_mean for metric in metrics])), 6),
                },
            },
            "qualitative_notes": [],
            "errors": [],
            "runtime": {
                "elapsed_seconds": round(time.time() - started, 3),
            },
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(output_video)
    print(report_path)
    print(contact_sheet_path)
    for path in interpolated_paths:
        print(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", help="Path to the shot 1 video.")
    parser.add_argument("--second", help="Path to the shot 2 extension video.")
    parser.add_argument("--output-dir", help="Runtime output directory.")
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.first and args.second:
        first = Path(args.first)
        second = Path(args.second)
    elif args.first or args.second:
        raise ValueError("Pass --first and --second together, or neither.")
    else:
        report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
        first = Path(report["input"]["shot1_video"])
        second = Path(report["result"]["generated_video"])
    for path in (first, second):
        if not path.is_file():
            raise FileNotFoundError(path)
    return first, second


def validate_inputs(first_meta: dict[str, Any], second_meta: dict[str, Any]) -> None:
    expected = {"fps": FPS, "frames": 193, "width": 1280, "height": 720}
    for name, meta in (("first", first_meta), ("second", second_meta)):
        for key, value in expected.items():
            if key == "fps":
                if not math.isclose(float(meta[key]), float(value), rel_tol=0.0, abs_tol=1e-6):
                    raise ValueError(f"{name} video {key} mismatch: {meta[key]} != {value}")
            elif int(meta[key]) != int(value):
                raise ValueError(f"{name} video {key} mismatch: {meta[key]} != {value}")


def video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        return {
            "path": str(path),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fourcc": int(capture.get(cv2.CAP_PROP_FOURCC)),
        }
    finally:
        capture.release()


def image_metadata(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot open image: {path}")
    return {
        "path": str(path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "channels": int(image.shape[2]) if image.ndim == 3 else 1,
        "size_bytes": int(path.stat().st_size),
    }


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = run_capture(command)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + "\nstdout:\n"
            + result.stdout
            + "\nstderr:\n"
            + result.stderr
        )
    return result


def run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def parse_minterpolate_support(help_text: str) -> dict[str, bool]:
    return {
        "mi_mode": "mi_mode" in help_text,
        "mc_mode": "mc_mode" in help_text,
        "me_mode": "me_mode" in help_text,
        "me": " me                <int>" in help_text or "\n   me                <int>" in help_text,
        "vsbmc": "vsbmc" in help_text,
    }


def build_minterpolate_filter(support: dict[str, bool]) -> str:
    options = [f"fps={SEAM_INTERP_FPS}"]
    if support.get("mi_mode"):
        options.append("mi_mode=mci")
    if support.get("mc_mode"):
        options.append("mc_mode=aobmc")
    if support.get("me_mode"):
        options.append("me_mode=bidir")
    if support.get("me"):
        options.append("me=epzs")
    if support.get("vsbmc"):
        options.append("vsbmc=1")
    return ":".join(options)


def summarize_version(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    summary = {"first_lines": lines[:4]}
    for line in lines[:20]:
        if line.startswith("ffmpeg version"):
            summary["version_line"] = line
            break
    return summary


def load_frame(path: Path, frame_index: int) -> np.ndarray:
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        if frame_index != 0:
            raise ValueError(f"Image inputs only support frame_index=0: {path}")
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read image: {path}")
        return frame
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index} from {path}")
        return frame
    finally:
        capture.release()


def compute_metrics(
    labels: list[tuple[str, Path, int]],
    frames: list[np.ndarray],
) -> list[FrameMetric]:
    metrics: list[FrameMetric] = []
    for (left_label, _, _), (right_label, _, _), left, right in zip(
        labels[:-1], labels[1:], frames[:-1], frames[1:]
    ):
        metrics.append(
            FrameMetric(
                left_label=left_label,
                right_label=right_label,
                ssim=ssim(left, right, COMPARE_WIDTH),
                dis_flow_mean=mean_dis_flow(left, right, COMPARE_WIDTH),
            )
        )
    return metrics


def ssim(first: np.ndarray, second: np.ndarray, compare_width: int) -> float:
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


def mean_dis_flow(first: np.ndarray, second: np.ndarray, compare_width: int) -> float:
    first_gray = prepare_flow_frame(first, compare_width)
    second_gray = prepare_flow_frame(second, compare_width)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(
        first_gray, second_gray, None
    )
    magnitude = np.linalg.norm(flow, axis=2)
    return float(magnitude.mean())


def prepare_flow_frame(frame: np.ndarray, compare_width: int) -> np.ndarray:
    height = max(1, round(frame.shape[0] * compare_width / frame.shape[1]))
    resized = cv2.resize(frame, (compare_width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def make_contact_sheet(
    labels: list[tuple[str, Path, int]], frames: list[np.ndarray]
) -> np.ndarray:
    cell_width = 340
    image_height = 180
    label_height = 42
    columns = 4
    rows = math.ceil(len(frames) / columns)
    sheet = np.full(
        (rows * (image_height + label_height), columns * cell_width, 3),
        18,
        dtype=np.uint8,
    )
    for index, ((label, _, _), frame) in enumerate(zip(labels, frames)):
        row = index // columns
        col = index % columns
        resized = cv2.resize(frame, (cell_width, image_height), interpolation=cv2.INTER_AREA)
        x = col * cell_width
        y = row * (image_height + label_height)
        sheet[y : y + image_height, x : x + cell_width] = resized
        overlay = sheet[y + image_height : y + image_height + label_height, x : x + cell_width]
        overlay[:] = (30, 30, 30)
        text = label
        scale = 0.65
        thickness = 1
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        while text_size[0] > cell_width - 16 and scale > 0.35:
            scale -= 0.05
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        text_x = 10
        text_y = label_height // 2 + text_size[1] // 2
        cv2.putText(
            overlay,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (245, 245, 245),
            thickness,
            cv2.LINE_AA,
        )
        cv2.rectangle(sheet, (x, y), (x + cell_width - 1, y + image_height + label_height - 1), (55, 55, 55), 1)
    return sheet


if __name__ == "__main__":
    main()
