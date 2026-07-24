#!/usr/bin/env python3
"""Measure retained input frames and find a local splice around one second."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from best_frame_seam import (
    FrameSample,
    concatenate_at_frames,
    read_edge_frames,
    similarity_matrix,
    ssim,
    write_csv,
    write_pair_preview,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / ".runtime/experiments/little_prince_video_extension"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir or DEFAULT_DIR)
    first, second, input_video = resolve_inputs(args, output_dir)
    result = analyze_and_splice(
        first,
        second,
        input_video,
        output_dir,
        retention_frames=args.retention_frames,
        tail_candidates=args.tail_candidates,
        boundary_frame=args.boundary_frame,
        boundary_radius=args.boundary_radius,
        compare_width=args.compare_width,
    )
    match = result["best_match"]
    print(
        f"Input retention mean SSIM: {result['input_retention']['mean_ssim']:.6f}"
    )
    print(
        f"Best seam: shot1 frame {match['first_frame_index']} -> "
        f"shot2 frame {match['second_frame_index']} (SSIM={match['score']:.6f})"
    )
    print(result["output_video"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first")
    parser.add_argument("--second")
    parser.add_argument("--input-video", help="The exact tail clip uploaded to Seedance.")
    parser.add_argument("--output-dir")
    parser.add_argument("--retention-frames", type=int, default=24)
    parser.add_argument("--tail-candidates", type=int, default=12)
    parser.add_argument("--boundary-frame", type=int, default=24)
    parser.add_argument("--boundary-radius", type=int, default=6)
    parser.add_argument("--compare-width", type=int, default=320)
    return parser.parse_args()


def resolve_inputs(
    args: argparse.Namespace, output_dir: Path
) -> tuple[Path, Path, Path]:
    if args.first and args.second:
        first = Path(args.first)
        second = Path(args.second)
        input_video = Path(args.input_video) if args.input_video else first
    elif args.first or args.second or args.input_video:
        raise ValueError("Pass --first and --second together")
    else:
        report = json.loads(
            (output_dir / "experiment_report_with_input.json").read_text(encoding="utf-8")
        )
        first = Path(report["input"]["shot1_video"])
        second = Path(report["result"]["generated_video"])
        input_video = Path(report["input"]["tail_video"])
    for path in (first, second, input_video):
        if not path.is_file():
            raise FileNotFoundError(path)
    return first, second, input_video


def analyze_and_splice(
    first_path: Path,
    second_path: Path,
    input_video_path: Path,
    output_dir: Path,
    *,
    retention_frames: int = 24,
    tail_candidates: int = 12,
    boundary_frame: int = 24,
    boundary_radius: int = 6,
    compare_width: int = 320,
) -> dict[str, Any]:
    input_frames, _ = read_frame_range(input_video_path, 0, retention_frames)
    generated_prefix, _ = read_frame_range(second_path, 0, retention_frames)
    if len(input_frames) != retention_frames or len(generated_prefix) != retention_frames:
        raise ValueError("Videos are too short for the input-retention comparison")
    retention_scores = [
        ssim(source.image, generated.image, compare_width)
        for source, generated in zip(input_frames, generated_prefix)
    ]

    first_candidates, first_meta = read_edge_frames(
        first_path, tail_candidates, tail=True
    )
    second_start = max(0, boundary_frame - boundary_radius)
    second_count = boundary_radius * 2
    second_candidates, _ = read_frame_range(second_path, second_start, second_count)
    if len(second_candidates) != second_count:
        raise ValueError("Generated video is too short for the boundary search")
    matrix = similarity_matrix(first_candidates, second_candidates, compare_width)
    best_row, best_col = np.unravel_index(np.argmax(matrix), matrix.shape)
    best_first = first_candidates[int(best_row)]
    best_second = second_candidates[int(best_col)]

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_label = f"{len(first_candidates)}x{len(second_candidates)}"
    output_video = output_dir / f"shot1_shot2_with_input_{matrix_label}_ssim.mp4"
    concatenate_at_frames(
        first_path,
        second_path,
        output_video,
        first_end_frame=best_first.index + 1,
        second_start_frame=best_second.index,
        fps=float(first_meta["fps"]),
    )
    write_csv(
        output_dir / f"retained_overlap_{matrix_label}_matrix.csv",
        first_candidates,
        second_candidates,
        matrix,
    )
    write_pair_preview(
        output_dir / f"retained_overlap_{matrix_label}_best_pair.jpg",
        best_first,
        best_second,
        float(matrix[best_row, best_col]),
    )

    report = {
        "metric": "SSIM (mean over BGR channels)",
        "input_retention": {
            "reference_video": str(input_video_path),
            "generated_video": str(second_path),
            "frames": retention_frames,
            "frame_scores": [round(score, 6) for score in retention_scores],
            "mean_ssim": round(float(np.mean(retention_scores)), 6),
            "minimum_ssim": round(float(np.min(retention_scores)), 6),
            "maximum_ssim": round(float(np.max(retention_scores)), 6),
        },
        "candidate_search": {
            "first_frame_indices": [frame.index for frame in first_candidates],
            "second_frame_indices": [frame.index for frame in second_candidates],
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
    report_path = output_dir / f"retained_overlap_{matrix_label}_seam_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def read_frame_range(
    path: Path, start: int, count: int
) -> tuple[list[FrameSample], dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for index in range(start, min(total, start + count)):
            ok, image = capture.read()
            if not ok:
                break
            frames.append(FrameSample(index=index, image=image))
        return frames, {"fps": fps, "frames": total}
    finally:
        capture.release()


if __name__ == "__main__":
    main()
