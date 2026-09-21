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

def _algorithm_step_max_attempts(bundle: Dict[str, Any]) -> int:
    raw = (
        bundle.get("settings", {})
        .get("generation", {})
        .get("algorithm_step_max_attempts", DEFAULT_ALGORITHM_STEP_MAX_ATTEMPTS)
    )
    try:
        attempts = int(raw)
    except (TypeError, ValueError):
        attempts = DEFAULT_ALGORITHM_STEP_MAX_ATTEMPTS
    return min(max(attempts, 1), 10)


def _reference_selection_disabled(bundle: Dict[str, Any]) -> bool:
    return _reference_selection_mode(bundle) == "none"


def _reference_selection_skips_visual_plan(bundle: Dict[str, Any]) -> bool:
    return _reference_selection_mode(bundle) in {NAIVE_TOP_K_SELECTION_MODE, SINK_RECENT_MEMORY_SELECTION_MODE}


def _reference_selection_naive(bundle: Dict[str, Any]) -> bool:
    return _reference_selection_mode(bundle) == NAIVE_TOP_K_SELECTION_MODE


def _reference_selection_sink_recent(bundle: Dict[str, Any]) -> bool:
    return _reference_selection_mode(bundle) == SINK_RECENT_MEMORY_SELECTION_MODE


def _reference_selection_mode(bundle: Dict[str, Any]) -> str:
    return str(
        bundle.get("settings", {})
        .get("visual_element_memory", {})
        .get("selection_mode", "greedy_coverage")
    )


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import json

        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
