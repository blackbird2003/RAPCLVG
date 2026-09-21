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
    BackendExecutionError,
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

def _prefix_video_assets(shots: List[Dict[str, Any]]) -> Dict[str, str]:
    assets: Dict[str, str] = {}
    for shot in sorted(shots, key=lambda item: item.get("order_index", 0)):
        if shot.get("state", {}).get("status") != "completed":
            break
        attempt = shot.get("attempt") or {}
        asset_id = attempt.get("outputs", {}).get("raw_video_asset_id")
        if asset_id:
            assets[shot["shot_id"]] = asset_id
    return assets


def _thumbnail_from_memory(produced_memory: List[Dict[str, Any]]) -> str | None:
    for memory in produced_memory:
        asset_id = memory.get("asset_id")
        if asset_id:
            return str(asset_id)
    return None


def _assembly_clips(
    store: ProjectStore,
    project_id: str,
    shots: List[Dict[str, Any]],
    assets: Dict[str, Any],
) -> List[Dict[str, Any]]:
    clips: List[Dict[str, Any]] = []
    for shot in shots:
        attempt = shot.get("attempt") or {}
        asset_id = attempt.get("outputs", {}).get("raw_video_asset_id")
        if not asset_id:
            raise BackendExecutionError(f"Completed shot {shot['shot_id']} has no raw video asset")
        path = store.asset_path(project_id, str(asset_id))
        asset = assets.get(str(asset_id), {})
        clips.append(
            {
                "shot_id": shot["shot_id"],
                "attempt_id": attempt.get("attempt_id"),
                "attempt_dir": str(store.attempt_dir(project_id, shot["shot_id"], str(attempt.get("attempt_id")))),
                "output_video": str(path),
                "raw_video_asset_id": str(asset_id),
                "generation_mode": shot.get("inputs", {}).get("generation_mode", "default"),
                "metadata": asset.get("metadata") or {},
            }
        )
    return clips


def _contains_placeholder_clip(clips: List[Dict[str, Any]]) -> bool:
    for clip in clips:
        metadata = clip.get("metadata") or {}
        if metadata.get("fake") or metadata.get("dry_run"):
            return True
        if not _looks_like_mp4(Path(str(clip.get("output_video") or ""))):
            return True
    return False


def _write_placeholder_final(path: Path, clips: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Video generation placeholder final assembly"]
    for clip in clips:
        lines.append(f"{clip['shot_id']} {clip['raw_video_asset_id']} {clip['output_video']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _concat_videos_ffmpeg(paths: List[Path], output_path: Path) -> None:
    if not paths:
        raise BackendExecutionError("No videos available for final assembly")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        shutil.copyfile(paths[0], output_path)
        return
    ffmpeg = _ffmpeg_executable()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        list_path = Path(handle.name)
        for path in paths:
            handle.write(f"file '{_ffmpeg_concat_escape(path)}'\n")
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        temporary.replace(output_path)
    except subprocess.CalledProcessError as exc:
        raise BackendExecutionError(f"Final video concatenation failed: {exc}") from exc
    finally:
        list_path.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def _ffmpeg_executable() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ModuleNotFoundError as exc:
        raise BackendExecutionError("ffmpeg executable is required for final assembly") from exc


def _ffmpeg_concat_escape(path: Path) -> str:
    return str(path).replace("'", "'\\''")


def _looks_like_mp4(path: Path) -> bool:
    try:
        header = path.read_bytes()[:32]
    except OSError:
        return False
    return b"ftyp" in header
