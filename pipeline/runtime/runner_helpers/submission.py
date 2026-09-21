from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import requests

from ...agentic.visual_element_memory import (
    DEFAULT_VISUAL_ELEMENT_MODEL,
    VISUAL_ELEMENT_MAX_OUTPUT_TOKENS,
    _call_ark_chat,
    _extract_json_object,
)
from ...agentic.visual_plan_reflection import (
    VisualPlanReflectionRequest,
    apply_visual_plan_reflection,
    reflect_visual_plan_with_llm,
)
from ...generators.seedance_client import (
    SeedanceError,
    extract_last_frame_url,
    extract_task_id,
    extract_video_url,
)
from ...keyframes.settings import DEFAULT_PROFILE as DEFAULT_KEYFRAME_PROFILE
from ...models import RunConfig, ShotSpec
from ...prompting import (
    SMOOTH_CONTINUATION_INSTRUCTION,
    audio_item_with_metadata,
    compose_prompt,
    escape_prompt_line,
    image_item_with_metadata,
    story_prompt_outline,
    video_item_with_metadata,
)
from ..constants import (
    DEFAULT_ALGORITHM_STEP_MAX_ATTEMPTS,
    DEFAULT_STEP_RETRY_DELAY_SECONDS,
    FORCE_ANIMATION_PROMPT_PREFIX,
    MEDIA_PREFLIGHT_TIMEOUT_SECONDS,
    NAIVE_TOP_K_SELECTION_MODE,
    SINK_RECENT_KEYFRAME_PROFILE,
    SINK_RECENT_MEMORY_FIX,
    SINK_RECENT_MEMORY_MAX_SIZE,
    SINK_RECENT_MEMORY_SELECTION_MODE,
    STYLE_REFERENCE_GUIDANCE,
)
from ..contracts import (
    ReferenceSelectionContext,
    ReferenceSelectionResult,
    RunnerCancelledError,
    SeedanceGenerationContext,
    SeedanceGenerationResult,
    SeedancePromptCompositionResult,
    ShotExecutionContext,
    ShotExecutionResult,
    VideoPostprocessor,
    VideoPostprocessContext,
    VideoPostprocessResult,
    VisualMemoryPlanContext,
    VisualMemoryPlanResult,
    VisualMemoryPlanner,
)
from ..domain import (
    MAX_REFERENCE_IMAGES,
    MIN_SMOOTH_REFERENCE_SECONDS,
    SCHEMA_VERSION,
    STEP_SEQUENCE,
)
from ..gpu_lock import GpuLockCancelled, KeyframeGpuLock
from ..json_store import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic
from ..locks import GlobalRunLock
from ..project_store import InvalidProjectError, ProjectStore, _completed_prefix, _now

def _content_summary_for_storage(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for item in content:
        current = {key: value for key, value in item.items() if key not in {"image_url", "video_url", "audio_url"}}
        if item.get("type") == "text":
            current["text"] = item.get("text", "")
        if "image_url" in item:
            current["image_url"] = _image_url_summary_for_storage(item)
        if "video_url" in item:
            current["video_url"] = {"url": str(item["video_url"].get("url") or "")}
        if "audio_url" in item:
            current["audio_url"] = {"url": str(item["audio_url"].get("url") or "")}
        summary.append(current)
    return summary


_SEEDANCE_CONTENT_INDEX_RE = re.compile(r"content\[(\d+)\]")


def _seedance_rejected_content_index(exc: Exception) -> int | None:
    text = str(exc)
    data = _json_object_from_error_text(text)
    if isinstance(data, dict):
        error = data.get("error") if isinstance(data.get("error"), dict) else {}
        haystack = "\n".join(
            str(value or "")
            for value in (error.get("param"), error.get("message"), text)
        )
    else:
        haystack = text
    match = _SEEDANCE_CONTENT_INDEX_RE.search(haystack)
    if not match:
        return None
    return int(match.group(1))


def _json_object_from_error_text(text: str) -> Dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        data = json.loads(text[start:])
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _sensitive_image_drop_record(
    content_index: int,
    item: Dict[str, Any],
    exc: Exception,
) -> Dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return {
        "content_index": content_index,
        "type": item.get("type"),
        "role": item.get("role"),
        "local_source": metadata.get("source_path") or metadata.get("file") or "",
        "metadata": metadata,
        "content_summary": _content_summary_for_storage([item])[0],
        "seedance_error": str(exc),
    }


def _image_url_summary_for_storage(item: Dict[str, Any]) -> Dict[str, Any]:
    image_url = item.get("image_url") if isinstance(item.get("image_url"), dict) else {}
    url = str(image_url.get("url") or "")
    if not url.startswith("data:"):
        return {"url": url}
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    local_source = metadata.get("source_path") or metadata.get("file") or "[embedded image data]"
    result = {
        "url": str(local_source),
        "embedded_data_url": True,
    }
    if url.startswith("data:") and ";" in url[:80]:
        result["mime"] = url[5 : url.index(";")]
    return result


def _media_debug_for_submission(content: List[Dict[str, Any]]) -> Dict[str, Any]:
    reference_videos: List[Dict[str, Any]] = []
    reference_audio: List[Dict[str, Any]] = []
    for index, item in enumerate(content, start=1):
        if item.get("type") not in {"video_url", "audio_url"}:
            continue
        media_type = "video" if item.get("type") == "video_url" else "audio"
        media_url = item.get(f"{media_type}_url") if isinstance(item.get(f"{media_type}_url"), dict) else {}
        record = {
            "index": index,
            "role": item.get("role"),
            "url": str(media_url.get("url") or ""),
            "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
            "preflight": _preflight_media_url(str(media_url.get("url") or "")),
        }
        (reference_videos if media_type == "video" else reference_audio).append(record)
    return {"reference_videos": reference_videos, "reference_audio": reference_audio}


def _preflight_media_url(url: str) -> Dict[str, Any]:
    if not url.startswith(("http://", "https://")):
        return {"status": "skipped", "reason": "not_http_url"}
    try:
        response = requests.get(
            url,
            allow_redirects=True,
            headers={"Range": "bytes=0-63", "User-Agent": "videogen-debug/1.0"},
            stream=True,
            timeout=MEDIA_PREFLIGHT_TIMEOUT_SECONDS,
        )
        first = b""
        try:
            for chunk in response.iter_content(chunk_size=64):
                if chunk:
                    first = chunk[:64]
                    break
        finally:
            response.close()
        content_type = response.headers.get("content-type", "")
        first_ascii = "".join(chr(value) if 32 <= value < 127 else "." for value in first[:32])
        return {
            "status": "ok",
            "status_code": response.status_code,
            "final_url": response.url,
            "content_type": content_type,
            "content_length": response.headers.get("content-length"),
            "history": [
                {
                    "status_code": item.status_code,
                    "url": item.url,
                    "location": item.headers.get("location"),
                }
                for item in response.history
            ],
            "first_bytes_hex": first[:32].hex(),
            "first_bytes_ascii": first_ascii,
            "is_probable_mp4": "video/mp4" in content_type.lower() or b"ftyp" in first[:32],
        }
    except Exception as exc:
        return {
            "status": "error",
            "type": type(exc).__name__,
            "message": str(exc),
        }


def _redact_urls(value: Any) -> Any:
    return value
