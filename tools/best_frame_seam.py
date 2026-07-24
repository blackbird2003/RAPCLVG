#!/usr/bin/env python3
"""Find the most similar edge-frame pair and join two videos at that seam."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / ".runtime/experiments/little_prince_video_extension"


@dataclass(frozen=True)
class FrameSample:
    index: int
    image: np.ndarray


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir or DEFAULT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    first_path, second_path = resolve_inputs(args, output_dir)

    first_frames, first_meta = read_edge_frames(first_path, args.window, tail=True)
    second_frames, second_meta = read_edge_frames(second_path, args.window, tail=False)
    matrix = similarity_matrix(first_frames, second_frames, args.compare_width)
    best_row, best_col = np.unravel_index(np.argmax(matrix), matrix.shape)
    best_first = first_frames[int(best_row)]
    best_second = second_frames[int(best_col)]

    output_video = output_dir / args.output_name
    concatenate_at_frames(
        first_path,
        second_path,
        output_video,
        first_end_frame=best_first.index + 1,
        second_start_frame=best_second.index,
        fps=float(first_meta["fps"]),
    )

    report = {
        "metric": "SSIM (mean over BGR channels)",
        "window": args.window,
        "inputs": {
            "first": str(first_path),
            "second": str(second_path),
            "first_metadata": first_meta,
            "second_metadata": second_meta,
        },
        "matrix": {
            "row_frame_indices": [frame.index for frame in first_frames],
            "column_frame_indices": [frame.index for frame in second_frames],
            "scores": [[round(float(value), 6) for value in row] for row in matrix],
        },
        "best_match": {
            "first_frame_index": best_first.index,
            "first_tail_offset": best_first.index - int(first_meta["frames"]),
            "second_frame_index": best_second.index,
            "score": round(float(matrix[best_row, best_col]), 6),
            "first_frames_removed": int(first_meta["frames"]) - best_first.index - 1,
            "second_frames_removed": best_second.index,
        },
        "output_video": str(output_video),
    }
    report_path = output_dir / "best_frame_seam_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(output_dir / "best_frame_seam_matrix.csv", first_frames, second_frames, matrix)
    write_pair_preview(
        output_dir / "best_frame_seam_pair.jpg",
        best_first,
        best_second,
        float(matrix[best_row, best_col]),
    )

    print_matrix(first_frames, second_frames, matrix)
    print(
        f"Best seam: shot1 frame {best_first.index} -> shot2 frame {best_second.index} "
        f"(SSIM={matrix[best_row, best_col]:.6f})"
    )
    print(output_video)
    print(report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the last N frames of video 1 with the first N frames of video 2."
    )
    parser.add_argument("--first", help="First video. Defaults to shot1 from experiment report.")
    parser.add_argument("--second", help="Second video. Defaults to generated shot2.")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-name", default="shot1_shot2_best_ssim.mp4")
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--compare-width", type=int, default=320)
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path]:
    if args.first and args.second:
        paths = Path(args.first), Path(args.second)
    elif args.first or args.second:
        raise ValueError("Pass both --first and --second, or neither")
    else:
        report_path = output_dir / "experiment_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        paths = Path(report["input"]["shot1_video"]), Path(
            report["result"]["generated_video"]
        )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def read_edge_frames(
    path: Path, count: int, *, tail: bool
) -> tuple[list[FrameSample], dict[str, Any]]:
    if count <= 0:
        raise ValueError("--window must be positive")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        start = max(0, total - count) if tail else 0
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames: list[FrameSample] = []
        for index in range(start, min(total, start + count)):
            ok, image = capture.read()
            if not ok:
                break
            frames.append(FrameSample(index=index, image=image))
        if len(frames) != min(count, total):
            raise RuntimeError(f"Could not read {count} edge frames from {path}")
        metadata = {
            "fps": fps,
            "frames": total,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        return frames, metadata
    finally:
        capture.release()


def similarity_matrix(
    first: list[FrameSample], second: list[FrameSample], compare_width: int
) -> np.ndarray:
    return np.array(
        [
            [ssim(frame_a.image, frame_b.image, compare_width) for frame_b in second]
            for frame_a in first
        ],
        dtype=np.float64,
    )


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


def concatenate_at_frames(
    first: Path,
    second: Path,
    output: Path,
    *,
    first_end_frame: int,
    second_start_frame: int,
    fps: float,
) -> None:
    filters = (
        f"[0:v]trim=end_frame={first_end_frame},setpts=PTS-STARTPTS,fps={fps:g}[v0];"
        f"[1:v]trim=start_frame={second_start_frame},setpts=PTS-STARTPTS,"
        f"fps={fps:g}[v1];[v0][v1]concat=n=2:v=1:a=0[v]"
    )
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(first),
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
            "-y",
            str(output),
        ],
        check=True,
    )


def write_csv(
    path: Path,
    first: list[FrameSample],
    second: list[FrameSample],
    matrix: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["shot1_frame\\shot2_frame", *[frame.index for frame in second]])
        for frame, row in zip(first, matrix):
            writer.writerow([frame.index, *[f"{value:.6f}" for value in row]])


def write_pair_preview(
    path: Path, first: FrameSample, second: FrameSample, score: float
) -> None:
    height = 360
    width = round(first.image.shape[1] * height / first.image.shape[0])
    left = cv2.resize(first.image, (width, height), interpolation=cv2.INTER_AREA)
    right = cv2.resize(second.image, (width, height), interpolation=cv2.INTER_AREA)
    preview = np.concatenate([left, right], axis=1)
    labels = (
        f"shot1 frame {first.index}",
        f"shot2 frame {second.index} | SSIM {score:.4f}",
    )
    for x, label in ((12, labels[0]), (width + 12, labels[1])):
        cv2.putText(preview, label, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(preview, label, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.imwrite(str(path), preview)


def print_matrix(
    first: list[FrameSample], second: list[FrameSample], matrix: np.ndarray
) -> None:
    header = "shot1\\shot2 " + " ".join(f"{frame.index:>8}" for frame in second)
    print(header)
    for frame, row in zip(first, matrix):
        print(f"{frame.index:>11} " + " ".join(f"{value:8.5f}" for value in row))


if __name__ == "__main__":
    main()
