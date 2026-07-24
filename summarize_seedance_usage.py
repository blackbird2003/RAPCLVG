import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_NO_VIDEO_PRICE_CNY_PER_M_TOKENS = 46.0
DEFAULT_WITH_VIDEO_PRICE_CNY_PER_M_TOKENS = 28.0


def _load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _usage_from_record(record: dict) -> dict:
    result = record.get("result_response") or {}
    return result.get("usage") or {}


def _result_metadata(record: dict) -> dict:
    return record.get("result_response") or {}


def _price_for_record(record: dict, no_video_price: float, with_video_price: float) -> float:
    # Our current pipeline only sends text and images. Keep this branch for future video-reference runs.
    labels = record.get("reference_labels") or []
    has_video_reference = any(
        str(item.get("source_path", "")).lower().endswith((".mp4", ".mov", ".webm", ".avi"))
        for item in labels
        if isinstance(item, dict)
    )
    return with_video_price if has_video_reference else no_video_price


def summarize_path(path: Path, no_video_price: float, with_video_price: float) -> dict:
    log_path = path / "seedance_tasks.jsonl" if path.is_dir() else path
    if not log_path.exists():
        raise FileNotFoundError(f"Missing seedance task log: {log_path}")

    by_task = {}
    for record in _load_jsonl(log_path):
        if not record.get("result_response"):
            continue
        task_id = record.get("task_id")
        if not task_id:
            continue
        by_task[str(task_id)] = record

    totals = {
        "path": str(log_path),
        "tasks": len(by_task),
        "total_tokens": 0,
        "completion_tokens": 0,
        "estimated_cny": 0.0,
        "duration_seconds": 0,
        "by_resolution": defaultdict(int),
        "by_model": defaultdict(int),
    }

    for record in by_task.values():
        usage = _usage_from_record(record)
        meta = _result_metadata(record)
        total_tokens = int(usage.get("total_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        price = _price_for_record(record, no_video_price, with_video_price)

        totals["total_tokens"] += total_tokens
        totals["completion_tokens"] += completion_tokens
        totals["estimated_cny"] += total_tokens / 1_000_000 * price
        totals["duration_seconds"] += int(meta.get("duration") or 0)
        totals["by_resolution"][str(meta.get("resolution") or "unknown")] += 1
        totals["by_model"][str(meta.get("model") or "unknown")] += 1

    totals["by_resolution"] = dict(totals["by_resolution"])
    totals["by_model"] = dict(totals["by_model"])
    return totals


def _print_summary(summary: dict) -> None:
    print(f"Path: {summary['path']}")
    print(f"Tasks: {summary['tasks']}")
    print(f"Duration: {summary['duration_seconds']} s")
    print(f"Total tokens: {summary['total_tokens']:,}")
    print(f"Completion tokens: {summary['completion_tokens']:,}")
    print(f"Estimated cost: ¥{summary['estimated_cny']:.4f}")
    print(f"Resolution counts: {summary['by_resolution']}")
    print(f"Model counts: {summary['by_model']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Seedance task usage and estimate CNY cost.")
    parser.add_argument("paths", nargs="+", help="Result directories or seedance_tasks.jsonl files.")
    parser.add_argument("--no-video-price", type=float, default=DEFAULT_NO_VIDEO_PRICE_CNY_PER_M_TOKENS)
    parser.add_argument("--with-video-price", type=float, default=DEFAULT_WITH_VIDEO_PRICE_CNY_PER_M_TOKENS)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    summaries = [
        summarize_path(Path(path), args.no_video_price, args.with_video_price)
        for path in args.paths
    ]

    if args.json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return

    grand = {
        "tasks": 0,
        "duration_seconds": 0,
        "total_tokens": 0,
        "completion_tokens": 0,
        "estimated_cny": 0.0,
    }
    for idx, summary in enumerate(summaries):
        if idx:
            print()
        _print_summary(summary)
        for key in grand:
            grand[key] += summary[key]

    if len(summaries) > 1:
        print()
        print("Grand total")
        print(f"Tasks: {grand['tasks']}")
        print(f"Duration: {grand['duration_seconds']} s")
        print(f"Total tokens: {grand['total_tokens']:,}")
        print(f"Completion tokens: {grand['completion_tokens']:,}")
        print(f"Estimated cost: ¥{grand['estimated_cny']:.4f}")


if __name__ == "__main__":
    main()
