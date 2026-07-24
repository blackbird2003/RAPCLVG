#!/usr/bin/env python3
"""Select a video seam using appearance and optical-flow continuity."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from best_frame_seam import (
    FrameSample,
    concatenate_at_frames,
    read_edge_frames,
    ssim,
    write_pair_preview,
)
from retained_overlap_seam import read_frame_range


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / ".runtime/experiments/little_prince_video_extension"


@dataclass(frozen=True)
class MotionRepresentation:
    global_flow: np.ndarray
    local_grid: np.ndarray


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir or DEFAULT_DIR)
    first_path, second_path = resolve_inputs(args, output_dir)
    result = analyze_delta_seam(
        first_path,
        second_path,
        output_dir,
        window=args.window,
        compare_width=args.compare_width,
        grid_rows=args.grid_rows,
        grid_columns=args.grid_columns,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )
    best = result["best_match"]
    print(
        f"Best seam: shot1 frame {best['first_frame_index']} -> "
        f"shot2 frame {best['second_frame_index']} (cost={best['cost']:.6f})"
    )
    print(
        "Raw terms: "
        f"appearance={best['raw']['appearance']:.6f}, "
        f"prev_cross={best['raw']['motion_prev_cross']:.6f}, "
        f"cross_next={best['raw']['motion_cross_next']:.6f}"
    )
    print(result["output_video"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first")
    parser.add_argument("--second")
    parser.add_argument("--output-dir")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--compare-width", type=int, default=320)
    parser.add_argument("--grid-rows", type=int, default=6)
    parser.add_argument("--grid-columns", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.3)
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path]:
    if args.first and args.second:
        paths = Path(args.first), Path(args.second)
    elif args.first or args.second:
        raise ValueError("Pass --first and --second together")
    else:
        report = json.loads(
            (output_dir / "experiment_report.json").read_text(encoding="utf-8")
        )
        paths = Path(report["input"]["shot1_video"]), Path(
            report["result"]["generated_video"]
        )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def analyze_delta_seam(
    first_path: Path,
    second_path: Path,
    output_dir: Path,
    *,
    window: int = 12,
    compare_width: int = 320,
    grid_rows: int = 6,
    grid_columns: int = 8,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
) -> dict[str, Any]:
    if min(alpha, beta, gamma) < 0 or alpha + beta + gamma <= 0:
        raise ValueError("alpha, beta, and gamma must be non-negative with a positive sum")
    weight_sum = alpha + beta + gamma
    alpha, beta, gamma = alpha / weight_sum, beta / weight_sum, gamma / weight_sum

    first_candidates, first_meta = read_edge_frames(first_path, window, tail=True)
    second_candidates, _ = read_frame_range(second_path, 0, window)
    first_context, _ = read_frame_range(
        first_path, first_candidates[0].index - 1, window + 1
    )
    second_context, _ = read_frame_range(second_path, 0, window + 1)
    if len(first_context) != window + 1 or len(second_context) != window + 1:
        raise ValueError("Videos do not contain enough context frames")

    prepared_first = [prepare_frame(frame, compare_width) for frame in first_context]
    prepared_second = [prepare_frame(frame, compare_width) for frame in second_context]
    flow_engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    previous_motion = [
        represent_motion(
            calculate_flow(flow_engine, prepared_first[index], prepared_first[index + 1]),
            grid_rows,
            grid_columns,
        )
        for index in range(window)
    ]
    next_motion = [
        represent_motion(
            calculate_flow(flow_engine, prepared_second[index], prepared_second[index + 1]),
            grid_rows,
            grid_columns,
        )
        for index in range(window)
    ]

    candidates: list[dict[str, Any]] = []
    for row, first_frame in enumerate(first_candidates):
        first_prepared = prepared_first[row + 1]
        for column, second_frame in enumerate(second_candidates):
            cross_motion = represent_motion(
                calculate_flow(flow_engine, first_prepared, prepared_second[column]),
                grid_rows,
                grid_columns,
            )
            left = motion_difference(previous_motion[row], cross_motion)
            right = motion_difference(cross_motion, next_motion[column])
            candidates.append(
                {
                    "row": row,
                    "column": column,
                    "first_frame_index": first_frame.index,
                    "second_frame_index": second_frame.index,
                    "appearance": 1.0
                    - ssim(first_frame.image, second_frame.image, compare_width),
                    "motion_prev_cross": left["combined"],
                    "motion_cross_next": right["combined"],
                    "camera_prev_cross": left["camera"],
                    "local_prev_cross": left["local"],
                    "camera_cross_next": right["camera"],
                    "local_cross_next": right["local"],
                }
            )

    add_normalized_costs(candidates, alpha, beta, gamma)
    best = min(candidates, key=lambda candidate: candidate["cost"])
    best_first = first_candidates[int(best["row"])]
    best_second = second_candidates[int(best["column"])]

    output_dir.mkdir(parents=True, exist_ok=True)
    label = f"{window}x{window}"
    output_video = output_dir / f"shot1_shot2_delta_{label}.mp4"
    concatenate_at_frames(
        first_path,
        second_path,
        output_video,
        first_end_frame=best_first.index + 1,
        second_start_frame=best_second.index,
        fps=float(first_meta["fps"]),
    )
    write_pair_preview(
        output_dir / f"delta_{label}_best_pair.jpg",
        best_first,
        best_second,
        1.0 - float(best["appearance"]),
    )
    csv_path = output_dir / f"delta_{label}_candidates.csv"
    write_candidates_csv(csv_path, candidates)

    report = {
        "method": {
            "appearance": "1 - SSIM",
            "flow": "OpenCV DIS medium preset at reduced resolution",
            "camera": "RANSAC affine global-flow difference / frame diagonal",
            "local": "6x8 median residual-flow grid difference / frame diagonal",
            "motion_difference": "0.5 * camera + 0.5 * local",
            "normalization": "independent min-max normalization over all candidates",
        },
        "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
        "inputs": {"first": str(first_path), "second": str(second_path)},
        "candidate_frames": {
            "first": [frame.index for frame in first_candidates],
            "second": [frame.index for frame in second_candidates],
        },
        "best_match": report_candidate(best, first_meta),
        "ranked_candidates": [
            report_candidate(candidate, first_meta)
            for candidate in sorted(candidates, key=lambda item: item["cost"])
        ],
        "output_video": str(output_video),
        "candidates_csv": str(csv_path),
    }
    report_path = output_dir / f"delta_{label}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def prepare_frame(frame: FrameSample, width: int) -> np.ndarray:
    height = max(1, round(frame.image.shape[0] * width / frame.image.shape[1]))
    resized = cv2.resize(frame.image, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def calculate_flow(
    engine: cv2.DISOpticalFlow, first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    return engine.calc(first, second, None)


def represent_motion(
    flow: np.ndarray, grid_rows: int, grid_columns: int
) -> MotionRepresentation:
    height, width = flow.shape[:2]
    step = max(4, min(height, width) // 24)
    ys, xs = np.mgrid[step // 2 : height : step, step // 2 : width : step]
    source = np.column_stack((xs.ravel(), ys.ravel())).astype(np.float32)
    sampled = flow[ys, xs].reshape(-1, 2)
    target = source + sampled
    matrix, _ = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=1.5
    )
    if matrix is None:
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    all_y, all_x = np.mgrid[0:height, 0:width].astype(np.float32)
    global_x = matrix[0, 0] * all_x + matrix[0, 1] * all_y + matrix[0, 2]
    global_y = matrix[1, 0] * all_x + matrix[1, 1] * all_y + matrix[1, 2]
    global_flow = np.stack((global_x - all_x, global_y - all_y), axis=-1)
    residual = flow - global_flow
    local_grid = grid_median(residual, grid_rows, grid_columns)
    return MotionRepresentation(global_flow=global_flow, local_grid=local_grid)


def grid_median(flow: np.ndarray, rows: int, columns: int) -> np.ndarray:
    row_chunks = np.array_split(flow, rows, axis=0)
    values = []
    for row in row_chunks:
        for cell in np.array_split(row, columns, axis=1):
            values.append(np.median(cell.reshape(-1, 2), axis=0))
    return np.asarray(values, dtype=np.float32).reshape(rows, columns, 2)


def motion_difference(
    first: MotionRepresentation, second: MotionRepresentation
) -> dict[str, float]:
    height, width = first.global_flow.shape[:2]
    diagonal = float(np.hypot(width, height))
    camera = float(
        np.linalg.norm(first.global_flow - second.global_flow, axis=2).mean()
        / diagonal
    )
    local = float(
        np.linalg.norm(first.local_grid - second.local_grid, axis=2).mean()
        / diagonal
    )
    return {"camera": camera, "local": local, "combined": 0.5 * (camera + local)}


def add_normalized_costs(
    candidates: list[dict[str, Any]], alpha: float, beta: float, gamma: float
) -> None:
    terms = ("appearance", "motion_prev_cross", "motion_cross_next")
    normalized = {term: min_max([item[term] for item in candidates]) for term in terms}
    for index, candidate in enumerate(candidates):
        candidate["normalized"] = {
            term: float(normalized[term][index]) for term in terms
        }
        candidate["weighted"] = {
            "appearance": alpha * candidate["normalized"]["appearance"],
            "motion_prev_cross": beta
            * candidate["normalized"]["motion_prev_cross"],
            "motion_cross_next": gamma
            * candidate["normalized"]["motion_cross_next"],
        }
        candidate["cost"] = sum(candidate["weighted"].values())


def min_max(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    span = float(array.max() - array.min())
    return np.zeros_like(array) if span < 1e-12 else (array - array.min()) / span


def report_candidate(candidate: dict[str, Any], first_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "first_frame_index": int(candidate["first_frame_index"]),
        "first_tail_offset": int(candidate["first_frame_index"])
        - int(first_meta["frames"]),
        "second_frame_index": int(candidate["second_frame_index"]),
        "cost": round(float(candidate["cost"]), 8),
        "raw": {
            "appearance": round(float(candidate["appearance"]), 8),
            "motion_prev_cross": round(float(candidate["motion_prev_cross"]), 8),
            "motion_cross_next": round(float(candidate["motion_cross_next"]), 8),
        },
        "motion_components": {
            key: round(float(candidate[key]), 8)
            for key in (
                "camera_prev_cross",
                "local_prev_cross",
                "camera_cross_next",
                "local_cross_next",
            )
        },
        "normalized": {
            key: round(float(value), 8)
            for key, value in candidate["normalized"].items()
        },
        "weighted": {
            key: round(float(value), 8)
            for key, value in candidate["weighted"].items()
        },
    }


def write_candidates_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    fields = [
        "first_frame_index",
        "second_frame_index",
        "appearance",
        "motion_prev_cross",
        "motion_cross_next",
        "camera_prev_cross",
        "local_prev_cross",
        "camera_cross_next",
        "local_cross_next",
        "normalized_appearance",
        "normalized_motion_prev_cross",
        "normalized_motion_cross_next",
        "weighted_appearance",
        "weighted_motion_prev_cross",
        "weighted_motion_cross_next",
        "cost",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            row = {key: candidate[key] for key in fields[:9]}
            for term in ("appearance", "motion_prev_cross", "motion_cross_next"):
                row[f"normalized_{term}"] = candidate["normalized"][term]
                row[f"weighted_{term}"] = candidate["weighted"][term]
            row["cost"] = candidate["cost"]
            writer.writerow(row)


if __name__ == "__main__":
    main()
