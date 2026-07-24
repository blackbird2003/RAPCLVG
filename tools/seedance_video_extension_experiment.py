#!/usr/bin/env python3
"""One-off Seedance video-extension experiment for Little Prince shots 1 -> 2."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seedance_client import SeedanceClient, extract_task_id, extract_video_url


DEFAULT_PROJECT_ID = "9d90272eae614307b5f74d1704fb5d4e"
WITH_VIDEO_PRICE_CNY_PER_M_TOKENS = 28.0
NO_VIDEO_PRICE_CNY_PER_M_TOKENS = 46.0


def main() -> None:
    args = parse_args()
    root = ROOT
    db_path = root / ".runtime/web/storymem_web.sqlite3"
    output_dir = Path(args.output_dir or root / ".runtime/experiments/little_prince_video_extension")
    output_dir.mkdir(parents=True, exist_ok=True)

    shot1, shot2 = load_shots(db_path, args.project_id)
    shot1_video = Path(shot1["output_video"])
    if not shot1_video.is_file():
        raise FileNotFoundError(f"Shot 1 video not found: {shot1_video}")

    tail_video = output_dir / "shot1_tail_1s.mp4"
    suffix = "_with_input" if args.include_input_video else ""
    generated_video = output_dir / f"shot2_video_extension{suffix}.mp4"
    direct_video = output_dir / f"shot1_shot2{suffix}_direct.mp4"
    trimmed_video = output_dir / f"shot1_minus6_shot2_plus1{suffix}.mp4"
    report_path = output_dir / f"experiment_report{suffix}.json"
    prompt = extension_prompt(
        shot2["video_prompt"], include_input_video=args.include_input_video
    )

    extract_tail(shot1_video, tail_video, args.tail_seconds)
    report: dict[str, Any] = {
        "project_id": args.project_id,
        "experiment": "shot1 tail video -> shot2 video extension",
        "include_input_video": args.include_input_video,
        "prompt": prompt,
        "input": {
            "shot1_video": str(shot1_video),
            "tail_video": str(tail_video),
            "tail_seconds": args.tail_seconds,
            "tail_metadata": video_metadata(tail_video),
        },
        "baseline": {
            "mode": shot2["generation_mode"],
            "task_id": shot2["task_id"],
            "usage": json.loads(shot2["usage_json"] or "{}"),
        },
        "status": "prepared",
    }
    write_json(report_path, report)
    if args.prepare_only:
        print(report_path)
        return
    if not args.video_url and not args.task_id:
        raise ValueError(
            "Seedance requires reference_video to be a public web URL; "
            "pass --video-url or --task-id"
        )

    client = SeedanceClient(api_key=load_api_key())
    task_id = args.task_id
    try:
        if not task_id:
            created = client.create_task(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "video_url",
                        "video_url": {"url": args.video_url},
                        "role": "reference_video",
                    },
                ],
                duration=shot2["duration_seconds"],
                ratio="16:9",
                resolution="720p",
                generate_audio=False,
                watermark=False,
                return_last_frame=True,
            )
            task_id = extract_task_id(created)
            report.update({"status": "submitted", "task_id": task_id})
            write_json(report_path, report)
            print(f"Submitted Seedance task: {task_id}", flush=True)

        started = time.monotonic()
        result = client.wait_task(
            task_id,
            poll_interval=args.poll_interval,
            max_wait_seconds=args.max_wait_seconds,
        )
        client.download_video(extract_video_url(result), str(generated_video))
        elapsed = round(time.monotonic() - started, 3)
    except Exception as exc:
        report.update(
            {
                "status": "failed" if not task_id else "submitted",
                "task_id": task_id,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_json(report_path, report)
        raise

    concatenate(shot1_video, generated_video, direct_video)
    concatenate(
        shot1_video,
        generated_video,
        trimmed_video,
        first_end_trim_frames=6,
        second_start_trim_frames=1,
    )

    overlap_alignment = None
    if args.include_input_video:
        from retained_overlap_seam import analyze_and_splice

        overlap_alignment = analyze_and_splice(
            shot1_video,
            generated_video,
            tail_video,
            output_dir,
        )

    usage = dict(result.get("usage") or {})
    tokens = int(usage.get("total_tokens") or usage.get("completion_tokens") or 0)
    baseline_tokens = int(report["baseline"]["usage"].get("total_tokens") or 0)
    report.update(
        {
            "status": "completed",
            "elapsed_seconds": elapsed,
            "result": {
                "task_id": task_id,
                "model": result.get("model"),
                "duration_seconds": result.get("duration"),
                "resolution": result.get("resolution"),
                "usage": usage,
                "estimated_cny_with_video": round(
                    tokens / 1_000_000 * WITH_VIDEO_PRICE_CNY_PER_M_TOKENS, 6
                ),
                "generated_video": str(generated_video),
                "generated_metadata": video_metadata(generated_video),
            },
            "comparison": {
                "baseline_tokens": baseline_tokens,
                "experiment_tokens": tokens,
                "token_delta": tokens - baseline_tokens,
                "baseline_estimated_cny": round(
                    baseline_tokens / 1_000_000 * NO_VIDEO_PRICE_CNY_PER_M_TOKENS,
                    6,
                ),
            },
            "assemblies": {
                "direct": str(direct_video),
                "trim_previous_6_next_1": str(trimmed_video),
                "retained_overlap_aligned": (
                    overlap_alignment["output_video"] if overlap_alignment else None
                ),
            },
        }
    )
    write_json(report_path, report)
    print(report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--output-dir")
    parser.add_argument("--tail-seconds", type=float, default=1.0)
    parser.add_argument(
        "--video-url",
        help="Public URL for the prepared one-second tail video.",
    )
    parser.add_argument("--task-id", help="Resume polling an already submitted task.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--include-input-video",
        action="store_true",
        help="Ask Seedance to retain the complete input video before extending it.",
    )
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument("--max-wait-seconds", type=int, default=1800)
    return parser.parse_args()


def load_shots(db_path: Path, project_id: str) -> tuple[sqlite3.Row, sqlite3.Row]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT s.order_index, s.video_prompt, s.duration_seconds, s.generation_mode,
               a.task_id, a.output_video, a.usage_json
        FROM shots s
        LEFT JOIN attempts a ON a.attempt_id = s.current_attempt_id
        WHERE s.project_id = ? AND s.order_index IN (0, 1)
        ORDER BY s.order_index
        """,
        (project_id,),
    ).fetchall()
    connection.close()
    if len(rows) != 2 or not all(row["output_video"] for row in rows):
        raise RuntimeError("The selected project must have completed shots 1 and 2")
    return rows[0], rows[1]


def extension_prompt(shot_prompt: str, *, include_input_video: bool = False) -> str:
    if include_input_video:
        return (
            "输出视频的开头必须完整包含视频1的全部内容，保持原始帧序、播放速度、人物动作和"
            "镜头运动；不要省略、压缩、重排或只保留视频1的尾部画面。视频1播放完毕后立即向后"
            "延长，保持运动方向与速度连续，不要回退、重复、停顿或转场。后续内容："
            f"{shot_prompt} No subtitles, text overlays, watermark, fade-out, or ending transition."
        )
    return (
        "向后延长 视频1，从视频1最后时刻的画面、人物动作、物体运动和镜头运动自然继续。"
        "不要回退、重复或停顿，保持运动方向与速度连续。后续内容："
        f"{shot_prompt} No subtitles, text overlays, watermark, fade-out, or ending transition."
    )


def ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(arguments: list[str]) -> None:
    subprocess.run([ffmpeg(), "-v", "error", *arguments], check=True)


def extract_tail(source: Path, output: Path, seconds: float) -> None:
    run_ffmpeg(
        [
            "-sseof",
            f"-{seconds:g}",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            "-y",
            str(output),
        ]
    )


def concatenate(
    first: Path,
    second: Path,
    output: Path,
    *,
    first_end_trim_frames: int = 0,
    second_start_trim_frames: int = 0,
) -> None:
    first_meta = video_metadata(first)
    first_end = int(first_meta["frames"]) - first_end_trim_frames
    if first_end <= 0:
        raise ValueError("Cannot trim all frames from the first video")
    fps = float(first_meta["fps"])
    filters = (
        f"[0:v]trim=end_frame={first_end},setpts=PTS-STARTPTS,fps={fps:g}[v0];"
        f"[1:v]trim=start_frame={second_start_trim_frames},setpts=PTS-STARTPTS,"
        f"fps={fps:g}[v1];[v0][v1]concat=n=2:v=1:a=0[v]"
    )
    run_ffmpeg(
        [
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
            "-movflags",
            "+faststart",
            "-y",
            str(output),
        ]
    )


def video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return {
            "fps": fps,
            "frames": frames,
            "duration_seconds": round(frames / fps, 6) if fps else None,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "size_bytes": path.stat().st_size,
        }
    finally:
        capture.release()


def load_api_key() -> str | None:
    key = os.getenv("SEEDANCE_API_KEY") or os.getenv("ARK_API_KEY")
    path = Path.home() / ".seedance_api_key"
    if not key and path.is_file():
        key = path.read_text(encoding="utf-8-sig").strip()
    return key


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
