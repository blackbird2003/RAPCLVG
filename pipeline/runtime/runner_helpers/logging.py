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


def _log(step: str, message: str) -> Dict[str, Any]:
    return {"time": _now(), "step": step, "message": message}


def _retry_attempt_record(attempt_index: int, exc: Exception) -> Dict[str, Any]:
    details = getattr(exc, "details", None)
    return {
        "attempt": attempt_index,
        "type": type(exc).__name__,
        "message": str(exc),
        "details": details if isinstance(details, dict) else None,
    }


def _attach_step_retry_details(exc: Exception, retry_attempts: List[Dict[str, Any]]) -> None:
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        details = {}
    details = {**details, "step_retry_attempts": retry_attempts}
    try:
        setattr(exc, "details", details)
    except Exception:
        pass


def _attach_keyframe_retry_details(exc: Exception, retry_attempts: List[Dict[str, Any]]) -> None:
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        details = {}
    details = {**details, "keyframe_maintaining_full_attempts": retry_attempts}
    try:
        setattr(exc, "details", details)
    except Exception:
        pass


def _step_retry_label(step: str) -> str:
    return {
        "visual_plan": "Visual Elements Plan",
        "reference_selection": "Historical Reference Selection",
        "seedance_generation": "Seedance Video Generation",
        "keyframe_maintaining": "Keyframe Maintaining",
    }.get(step, step.replace("_", " ").title())


def _terminal_log_step_for_step(step: str, status: str) -> str:
    if step == "visual_plan":
        return "planning_visual_elements"
    if step == "reference_selection":
        return "selecting_visual_references"
    if step == "seedance_prompt":
        return "composing_prompt"
    if step == "keyframe_maintaining":
        return "annotating_visual_memory"
    if step == "seedance_generation":
        return "submitting"
    return status


def _attempt_diagnostic_logs(details: Dict[str, Any], step: str) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    last_error = details.get("last_error")
    if last_error:
        logs.append(_log(step, f"Failure point: {last_error}"))
    for attempts_key in ("step_retry_attempts", "keyframe_maintaining_full_attempts"):
        attempts = details.get(attempts_key)
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            message = (
                f"Full step attempt {attempt.get('attempt', '?')} failed: "
                f"{attempt.get('type', 'Error')}: {attempt.get('message', '')}"
            )
            logs.append(_log(step, message))
    for label, attempts_key in (("LLM", "llm_attempts"), ("VLM", "vlm_attempts")):
        attempts = details.get(attempts_key)
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            attempt_number = attempt.get("attempt", "?")
            prompt = _attempt_prompt(attempt)
            if prompt:
                logs.append(_log(step, f"{label} attempt {attempt_number} submitted prompt:\n{prompt}"))
            raw_response = attempt.get("raw_response")
            if raw_response is not None:
                logs.append(_log(step, f"{label} attempt {attempt_number} raw response:\n{raw_response}"))
            parse_error = attempt.get("parse_error")
            if parse_error:
                logs.append(_log(step, f"{label} attempt {attempt_number} parse error:\n{parse_error}"))
    return logs


def _attempt_prompt(attempt: Dict[str, Any]) -> str:
    prompt = attempt.get("prompt")
    if isinstance(prompt, str):
        return prompt
    metadata = attempt.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("prompt"), str):
        return str(metadata["prompt"])
    return ""


def _short_name(prompt: str) -> str:
    clean = " ".join(prompt.split())
    return clean[:80] + ("..." if len(clean) > 80 else "")
