from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable


DEFAULT_NO_VIDEO_PRICE_CNY_PER_M_TOKENS = 46.0
DEFAULT_WITH_VIDEO_PRICE_CNY_PER_M_TOKENS = 28.0
VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".avi")


@dataclass(frozen=True)
class Pricing:
    no_video_cny_per_m_tokens: float = DEFAULT_NO_VIDEO_PRICE_CNY_PER_M_TOKENS
    with_video_cny_per_m_tokens: float = DEFAULT_WITH_VIDEO_PRICE_CNY_PER_M_TOKENS

    def to_dict(self) -> Dict[str, float]:
        return {
            "no_video_cny_per_m_tokens": self.no_video_cny_per_m_tokens,
            "with_video_cny_per_m_tokens": self.with_video_cny_per_m_tokens,
        }


def summarize_attempts(attempts: Iterable[Dict[str, Any]], pricing: Pricing | None = None) -> Dict[str, Any]:
    pricing = pricing or Pricing()
    rows = list(attempts)
    all_summary = _summarize(rows, pricing)
    current_summary = _summarize([row for row in rows if bool(row.get("is_current"))], pricing)
    return {
        "currency": "CNY",
        "pricing": pricing.to_dict(),
        "current_version": current_summary,
        "all_attempts": all_summary,
    }


def _summarize(rows: Iterable[Dict[str, Any]], pricing: Pricing) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "tasks": 0,
        "status_counts": {},
        "duration_seconds": 0,
        "total_tokens": 0,
        "completion_tokens": 0,
        "estimated_cny": 0.0,
        "usage_unknown": 0,
    }
    for row in rows:
        if not row.get("task_id"):
            continue
        summary["tasks"] += 1
        status = str(row.get("status") or "unknown")
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
        usage = row.get("usage") or {}
        total_tokens = _as_int(usage.get("total_tokens"))
        completion_tokens = _as_int(usage.get("completion_tokens"))
        duration = _as_int(usage.get("duration_seconds") or row.get("duration_seconds"))
        if total_tokens <= 0:
            summary["usage_unknown"] += 1
        summary["total_tokens"] += total_tokens
        summary["completion_tokens"] += completion_tokens
        summary["duration_seconds"] += duration
        price = (
            pricing.with_video_cny_per_m_tokens
            if _has_video_reference(row.get("input_snapshot") or {})
            else pricing.no_video_cny_per_m_tokens
        )
        summary["estimated_cny"] += total_tokens / 1_000_000 * price
    summary["estimated_cny"] = round(summary["estimated_cny"], 6)
    return summary


def _has_video_reference(input_snapshot: Dict[str, Any]) -> bool:
    for reference in input_snapshot.get("references") or []:
        path = str(reference.get("source_path") or "").lower()
        if os.path.splitext(path)[1] in VIDEO_SUFFIXES:
            return True
    return False


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
