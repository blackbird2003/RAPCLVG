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
from ...agentic.language import detect_story_language, output_language_instruction
from ...agentic.visual_plan_reflection import (
    VisualPlanReflectionRequest,
    apply_visual_plan_reflection,
    reflect_visual_plan_with_llm,
)
from ...generators.seedance_client import (
    DEFAULT_MODEL as DEFAULT_SEEDANCE_MODEL,
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

def _dry_run_visual_status(shot: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "element_id": f"element-{shot['shot_id']}-001",
            "name": _short_name(shot["inputs"]["video_prompt"]),
            "type": "scene",
            "introduced_at": shot["shot_id"],
            "notes": "Dry-run placeholder; real LLM/VLM element maintenance is not invoked.",
            "status": "new",
            "reason": "Dry-run creates a deterministic element row to validate request construction.",
        }
    ]


def _dry_run_references(previous_shots: List[Dict[str, Any]], settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    limit = int(settings.get("visual_element_memory", {}).get("max_retrieved_frames", 4) or 0)
    if limit <= 0:
        return []
    references: List[Dict[str, Any]] = []
    for shot in reversed(previous_shots):
        if len(references) >= limit:
            break
        attempt = shot.get("attempt") or {}
        produced = attempt.get("produced_visual_memory") or []
        if not produced:
            continue
        memory = produced[0]
        references.append(
            {
                "reference_id": f"ref-{shot['shot_id']}-{attempt.get('attempt_id', 'unknown')}",
                "asset_id": memory.get("asset_id"),
                "source_shot_id": shot["shot_id"],
                "source_scene_num": shot.get("scene_num"),
                "source_shot_num": shot.get("shot_num"),
                "source_prompt": shot.get("inputs", {}).get("video_prompt", ""),
                "file": str(memory.get("asset_id") or ""),
                "roles": ["visual_element_memory"],
                "covered_elements": list(memory.get("visible_elements") or []),
                "conflict_elements": [],
                "score": 1.0,
                "reason": "Dry-run selected the previous produced memory placeholder.",
            }
        )
    references.reverse()
    return references


def _empty_visual_plan_result(selection_mode: str) -> VisualMemoryPlanResult:
    if selection_mode == NAIVE_TOP_K_SELECTION_MODE:
        reason = "Naive top-k ablation keeps the visual element collection empty."
    elif selection_mode == SINK_RECENT_MEMORY_SELECTION_MODE:
        reason = "Sink + Recent memory mode uses early sink and recent-window memory without visual element planning."
    else:
        reason = f"Reference selection mode {selection_mode} does not require visual element planning."
    return VisualMemoryPlanResult(
        visual_element_status=[],
        details={
            "skipped": True,
            "selection_mode": selection_mode,
            "reason": reason,
        },
        logs=[
            _log(
                "planning_visual_elements",
                f"Visual Elements Plan skipped because Reference selection is {selection_mode}.",
            )
        ],
    )


def _select_naive_top_k_references(context: ReferenceSelectionContext, store: ProjectStore) -> ReferenceSelectionResult:
    budget = _naive_reference_budget(context)
    limit = int(budget["effective_max_retrieved_frames"])
    prompt = str(context.input_snapshot.get("video_prompt") or context.shot.get("inputs", {}).get("video_prompt") or "")
    score_details: Dict[str, Any] = {
        "method": "clip_text_image",
        "model": "ViT-B/32",
        "image_embedding_cache": "assets/embeddings",
        "fallback": None,
        "cache_hits": 0,
        "cache_writes": 0,
        "score_failures": [],
    }
    try:
        text_embedding = _clip_text_embedding(prompt)
    except Exception as exc:
        text_embedding = None
        score_details["fallback"] = f"text embedding failed; all scores set to 0: {type(exc).__name__}: {exc}"
    candidates = _naive_reference_candidates(context, store, text_embedding, score_details)
    candidates.sort(
        key=lambda item: (
            -float(item.get("score") or 0.0),
            -int(item.get("_source_order_index") or 0),
            int(item.get("rank") or 0),
            str(item.get("asset_id") or ""),
        )
    )
    selected_raw = candidates[:limit] if limit > 0 else []
    for item in selected_raw:
        item.pop("_source_order_index", None)
    selected = _visual_reference_rows(
        store=store,
        project_id=context.project_id,
        shots=context.bundle["shots"],
        references=selected_raw,
    )
    for index, item in enumerate(selected, start=1):
        item["reference_index"] = index
    prompt_context = _visual_reference_prompt_context(
        context.bundle,
        context.shot,
        context.input_snapshot,
        [],
        selected,
    )
    return ReferenceSelectionResult(
        selected_references=selected,
        prompt_context=_prompt_for_generation_mode(
            prompt_context,
            str(context.input_snapshot.get("generation_mode") or "default"),
            default_last_frame_continuity=_uses_default_last_frame_continuity(context.shot, context.input_snapshot),
        ),
        details={
            "selection_mode": NAIVE_TOP_K_SELECTION_MODE,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "score": score_details,
            "reference_budget": budget,
            "skipped_visual_element_planning": True,
        },
        logs=[
            _log(
                "selecting_visual_references",
                f"naive top-k CLIP-scored {len(candidates)} historical keyframes and selected {len(selected)} references",
            ),
            _log(
                "selecting_visual_references",
                (
                    "static image budget: "
                    f"predefined={budget['predefined_reference_count']}, "
                    "sink=0, "
                    f"retrieved_limit={budget['effective_max_retrieved_frames']}"
                ),
            ),
        ],
    )


def _select_sink_recent_memory_references(context: ReferenceSelectionContext, store: ProjectStore) -> ReferenceSelectionResult:
    budget = _sink_recent_reference_budget(context)
    limit = int(budget["effective_max_memory_size"])
    candidates = _sink_recent_memory_candidates(context, store)
    paths = [str(item["source_path"]) for item in candidates]
    memory_bank = _sink_recent_default_memory_bank(paths, limit, SINK_RECENT_MEMORY_FIX) if limit > 0 else []
    role_by_path = _sink_recent_default_memory_roles(memory_bank, limit, SINK_RECENT_MEMORY_FIX)
    candidate_by_path = {str(item["source_path"]): item for item in candidates}
    selected_raw: List[Dict[str, Any]] = []
    for path in memory_bank:
        item = dict(candidate_by_path.get(str(path)) or {})
        if not item:
            continue
        roles = role_by_path.get(str(path), ["default_memory"])
        item["roles"] = roles
        if "early_sink_memory" in roles:
            item["reference_intent"] = "Early sink memory selected as a stable long-range anchor."
        elif "recent_window_memory" in roles:
            item["reference_intent"] = "Recent-window memory selected for local visual continuity."
        else:
            item["reference_intent"] = "Default memory reference."
        selected_raw.append(item)
    selected = _visual_reference_rows(
        store=store,
        project_id=context.project_id,
        shots=context.bundle["shots"],
        references=selected_raw,
    )
    for index, item in enumerate(selected, start=1):
        item["reference_index"] = index
    prompt_context = _visual_reference_prompt_context(
        context.bundle,
        context.shot,
        context.input_snapshot,
        [],
        selected,
    )
    return ReferenceSelectionResult(
        selected_references=selected,
        prompt_context=_prompt_for_generation_mode(
            prompt_context,
            str(context.input_snapshot.get("generation_mode") or "default"),
            default_last_frame_continuity=_uses_default_last_frame_continuity(context.shot, context.input_snapshot),
        ),
        details={
            "selection_mode": SINK_RECENT_MEMORY_SELECTION_MODE,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "sink_recent_memory": {
                "max_memory_size": SINK_RECENT_MEMORY_MAX_SIZE,
                "fix": SINK_RECENT_MEMORY_FIX,
                "keyframe_profile": SINK_RECENT_KEYFRAME_PROFILE,
            },
            "reference_budget": budget,
            "skipped_visual_element_planning": True,
        },
        logs=[
            _log(
                "selecting_visual_references",
                (
                    "Sink + Recent memory selected "
                    f"{len(selected)} of {len(candidates)} historical keyframes "
                    f"with max_memory_size={limit}, fix={min(SINK_RECENT_MEMORY_FIX, limit)}."
                ),
            ),
            _log(
                "selecting_visual_references",
                (
                    "static image budget: "
                    f"predefined={budget['predefined_reference_count']}, "
                    f"default_last_frame_reserved={budget['default_last_frame_reserved']}, "
                    f"sink_recent_memory_limit={budget['effective_max_memory_size']}"
                ),
            ),
        ],
    )


def _sink_recent_memory_candidates(context: ReferenceSelectionContext, store: ProjectStore) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for shot in context.previous_shots:
        if shot.get("state", {}).get("status") != "completed":
            continue
        attempt = shot.get("attempt") or {}
        attempt_id = str(attempt.get("attempt_id") or shot.get("state", {}).get("current_attempt_id") or "unknown")
        source_prompt = str(shot.get("inputs", {}).get("video_prompt") or "")
        memories = [
            memory for memory in (attempt.get("produced_visual_memory") or [])
            if isinstance(memory, dict) and memory.get("active_in_memory_pool") is not False and memory.get("asset_id")
        ]
        memories.sort(key=lambda item: int(item.get("rank") or 0))
        for memory in memories:
            asset_id = str(memory.get("asset_id") or "")
            try:
                source_path = store.asset_path(context.project_id, asset_id)
            except Exception:
                continue
            candidates.append(
                {
                    "reference_id": f"sink-recent-{shot['shot_id']}-{attempt_id}-{asset_id}",
                    "asset_id": asset_id,
                    "source_path": str(source_path),
                    "source_shot_id": shot["shot_id"],
                    "source_scene_num": shot.get("scene_num"),
                    "source_shot_num": shot.get("shot_num"),
                    "source_prompt": source_prompt,
                    "file": source_path.name,
                    "roles": ["default_memory"],
                    "rank": memory.get("rank"),
                    "score": None,
                    "visual_element_selection": {
                        "score": None,
                        "newly_covered_element_ids": [],
                        "already_covered_element_ids": [],
                        "should_reference": [],
                        "should_exclude": [],
                        "optional_or_uncertain": [],
                    },
                    "holistic_description": "",
                    "reference_guidance": "",
                    "_source_order_index": shot.get("order_index", 0),
                }
            )
    return candidates


def _sink_recent_default_memory_bank(all_memory: List[str], max_memory_size: int, fix: int) -> List[str]:
    memory_bank = list(all_memory)
    if len(memory_bank) <= max_memory_size:
        return memory_bank
    fixed = max(0, min(fix, max_memory_size))
    recent = max_memory_size - fixed
    return memory_bank[:fixed] + (memory_bank[-recent:] if recent > 0 else [])


def _sink_recent_default_memory_roles(memory_bank: List[str], max_memory_size: int, fix: int) -> Dict[str, List[str]]:
    if not memory_bank:
        return {}
    fixed = max(0, min(fix, max_memory_size, len(memory_bank)))
    roles: Dict[str, List[str]] = {}
    for path in memory_bank[:fixed]:
        roles.setdefault(path, []).append("early_sink_memory")
    for path in memory_bank[fixed:]:
        roles.setdefault(path, []).append("recent_window_memory")
    return roles


def _naive_reference_candidates(
    context: ReferenceSelectionContext,
    store: ProjectStore,
    text_embedding: List[float] | None,
    score_details: Dict[str, Any],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for shot in context.previous_shots:
        if shot.get("state", {}).get("status") != "completed":
            continue
        attempt = shot.get("attempt") or {}
        attempt_id = str(attempt.get("attempt_id") or shot.get("state", {}).get("current_attempt_id") or "unknown")
        source_prompt = str(shot.get("inputs", {}).get("video_prompt") or "")
        for memory in attempt.get("produced_visual_memory") or []:
            if not isinstance(memory, dict) or memory.get("active_in_memory_pool") is False:
                continue
            asset_id = str(memory.get("asset_id") or "")
            if not asset_id:
                continue
            try:
                source_path = store.asset_path(context.project_id, asset_id)
            except Exception:
                continue
            image_embedding = None
            if text_embedding is not None:
                image_embedding = _clip_image_embedding_for_asset(
                    store,
                    context.project_id,
                    asset_id,
                    score_details,
                )
            score = _cosine_similarity(text_embedding, image_embedding) if image_embedding is not None else 0.0
            selection = {
                "score": score,
                "newly_covered_element_ids": [],
                "already_covered_element_ids": [],
                "should_reference": [],
                "should_exclude": [],
                "optional_or_uncertain": [],
            }
            candidates.append(
                {
                    "reference_id": f"naive-{shot['shot_id']}-{attempt_id}-{asset_id}",
                    "asset_id": asset_id,
                    "source_path": str(source_path),
                    "source_shot_id": shot["shot_id"],
                    "source_scene_num": shot.get("scene_num"),
                    "source_shot_num": shot.get("shot_num"),
                    "source_prompt": source_prompt,
                    "file": source_path.name,
                    "roles": ["naive_top_k_memory"],
                    "rank": memory.get("rank"),
                    "score": selection["score"],
                    "reference_intent": "Naive top-k baseline selected this frame by CLIP text-image similarity.",
                    "visual_element_selection": selection,
                    "holistic_description": "",
                    "reference_guidance": "",
                    "_source_order_index": shot.get("order_index", 0),
                }
            )
    return candidates


def _clip_text_embedding(prompt: str) -> List[float]:
    import clip
    import torch

    model, device, dtype = _naive_clip_model()
    tokens = clip.tokenize([str(prompt or "")], truncate=True).to(device)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
    return [float(value) for value in embedding[0].detach().cpu().tolist()]


def _clip_image_embedding_for_asset(
    store: ProjectStore,
    project_id: str,
    asset_id: str,
    score_details: Dict[str, Any] | None = None,
) -> List[float] | None:
    project_dir = store.project_dir(project_id)
    image_path = store.asset_path(project_id, asset_id)
    cache_path = _clip_embedding_cache_path(project_dir, asset_id)
    cached = read_json(cache_path, {})
    image_stat = image_path.stat()
    if (
        isinstance(cached, dict)
        and cached.get("model") == "ViT-B/32"
        and cached.get("asset_id") == asset_id
        and cached.get("source_mtime_ns") == image_stat.st_mtime_ns
        and isinstance(cached.get("embedding"), list)
    ):
        if score_details is not None:
            score_details["cache_hits"] = int(score_details.get("cache_hits") or 0) + 1
        return [float(value) for value in cached["embedding"]]
    try:
        embedding = _clip_image_embedding(image_path)
    except Exception as exc:
        if score_details is not None:
            failures = score_details.setdefault("score_failures", [])
            if isinstance(failures, list):
                failures.append({"asset_id": asset_id, "type": type(exc).__name__, "message": str(exc)})
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        cache_path,
        {
            "schema_version": SCHEMA_VERSION,
            "asset_id": asset_id,
            "model": "ViT-B/32",
            "source_path": _relative(project_dir, image_path),
            "source_mtime_ns": image_stat.st_mtime_ns,
            "embedding": embedding,
            "created_at": _now(),
        },
    )
    if score_details is not None:
        score_details["cache_writes"] = int(score_details.get("cache_writes") or 0) + 1
    return embedding


def _clip_image_embedding(image_path: Path) -> List[float]:
    import numpy as np
    import torch
    from PIL import Image
    from ..keyframes.extract import _clip_preprocess_tensor

    model, device, dtype = _naive_clip_model()
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        array = np.asarray(image).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    tensor = _clip_preprocess_tensor(tensor).to(device, dtype=dtype)
    with torch.no_grad():
        embedding = model.encode_image(tensor)
        embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
    return [float(value) for value in embedding[0].detach().cpu().tolist()]


def _naive_clip_model():
    from ..keyframes.extract import _get_clip_model

    device = os.environ.get("VIDEOGEN_NAIVE_CLIP_DEVICE", "cpu")
    use_half = device != "cpu" and _env_flag("VIDEOGEN_NAIVE_CLIP_HALF", default=False)
    return _get_clip_model(device=device, use_half=use_half)


def _clip_embedding_cache_path(project_dir: Path, asset_id: str) -> Path:
    safe_asset_id = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in asset_id)
    return project_dir / "assets" / "embeddings" / f"{safe_asset_id}_clip_vit_b_32.json"


def _cosine_similarity(left: List[float] | None, right: List[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(float(a) * float(b) for a, b in zip(left, right)))


def _naive_produced_memory_from_keyframes(
    keyframe_paths: List[str],
    shot_id: str,
    attempt_id: str,
) -> List[Dict[str, Any]]:
    produced: List[Dict[str, Any]] = []
    for index, path in enumerate(keyframe_paths, start=1):
        produced.append(
            {
                "asset_id": Path(path).stem,
                "rank": index,
                "visible_elements": [],
                "elements": [],
                "annotation": "",
                "holistic_description": "",
                "active_in_memory_pool": True,
                "source_shot_id": shot_id,
                "attempt_id": attempt_id,
            }
        )
    return produced


def _fake_references(previous_shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    references = []
    for shot in previous_shots[-2:]:
        attempt_id = shot.get("state", {}).get("current_attempt_id")
        if not attempt_id:
            continue
        references.append(
            {
                "reference_id": f"ref-{shot['shot_id']}-{attempt_id}",
                "source_shot_id": shot["shot_id"],
                "roles": ["fake_historical_reference"],
                "covered_elements": [f"element-{shot['shot_id']}-001"],
                "conflict_elements": [],
                "score": 1.0,
                "reason": "Previous completed shot used as fake historical evidence.",
                "holistic_description": f"Fake holistic description for source shot {shot['shot_id']}.",
                "reference_guidance": f"Use source shot {shot['shot_id']} as a broad fake visual reference.",
            }
        )
    return references


def _shot_spec_from_bundle_shot(shot: Dict[str, Any]) -> ShotSpec:
    inputs = shot["inputs"]
    return ShotSpec(
        scene={},
        scene_num=int(shot["scene_num"]),
        shot_num=int(shot["shot_num"]),
        prompt=str(inputs["video_prompt"]),
        is_cut=bool(inputs["is_cut"]),
        duration_seconds=int(inputs["duration_seconds"]),
    )


def _story_script(bundle: Dict[str, Any]) -> Dict[str, Any]:
    story = bundle.get("story") or {}
    source = story.get("source")
    if isinstance(source, dict) and source.get("scenes"):
        return source
    scenes_by_num: Dict[int, List[str]] = {}
    for shot in bundle.get("shots", []):
        scenes_by_num.setdefault(int(shot["scene_num"]), []).append(
            str(shot.get("inputs", {}).get("video_prompt", ""))
        )
    return {
        "story_name": bundle.get("project", {}).get("name", ""),
        "scenes": [
            {"scene_num": scene_num, "video_prompts": prompts}
            for scene_num, prompts in sorted(scenes_by_num.items())
        ],
    }


def _story_script_through_shot(bundle: Dict[str, Any], shot: Dict[str, Any]) -> Dict[str, Any]:
    try:
        current_order = int(shot.get("order_index", 0))
    except (TypeError, ValueError):
        current_order = 0
    scenes_by_num: Dict[int, List[str]] = {}
    for item in bundle.get("shots", []):
        try:
            order_index = int(item.get("order_index", 0))
        except (TypeError, ValueError):
            order_index = 0
        if order_index > current_order:
            continue
        try:
            scene_num = int(item.get("scene_num", 1))
        except (TypeError, ValueError):
            scene_num = 1
        prompt = str(item.get("inputs", {}).get("video_prompt", "")).strip()
        if prompt:
            scenes_by_num.setdefault(scene_num, []).append(prompt)
    return {
        "story_name": bundle.get("project", {}).get("name", ""),
        "scenes": [
            {"scene_num": scene_num, "video_prompts": prompts}
            for scene_num, prompts in sorted(scenes_by_num.items())
        ],
    }


def _fake_produced_memory(shot: Dict[str, Any], attempt_id: str) -> List[Dict[str, Any]]:
    return [
        {
            "asset_id": f"img_{shot['shot_id']}_{attempt_id}_001",
            "rank": 1,
            "visible_elements": [f"element-{shot['shot_id']}-001"],
            "annotation": "Fake keyframe annotation.",
            "holistic_description": "Fake holistic keyframe description with composition and story context.",
            "active_in_memory_pool": True,
        }
    ]


def _placeholder_produced_memory(
    asset_id: str,
    visual_status: List[Dict[str, Any]],
    annotation: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "asset_id": asset_id,
            "rank": 1,
            "visible_elements": [item["element_id"] for item in visual_status],
            "annotation": annotation,
            "holistic_description": "",
            "active_in_memory_pool": True,
        }
    ]


def _seedance_content(
    project_id: str,
    store: ProjectStore,
    submitted_prompt: str,
    references: List[Dict[str, Any]],
    *,
    include_text: bool = True,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    if include_text:
        content.append({"type": "text", "text": submitted_prompt})
    for reference in references:
        asset_id = reference.get("asset_id")
        if asset_id:
            path = store.asset_path(project_id, str(asset_id))
        elif reference.get("source_path"):
            path = Path(str(reference["source_path"]))
        else:
            continue
        content.append(image_item_with_metadata(str(path), reference))
    return content


def _run_config_for_bundle(bundle: Dict[str, Any], output_dir: Path) -> RunConfig:
    settings = bundle.get("settings") or {}
    generation = settings.get("generation") or {}
    visual = settings.get("visual_element_memory") or {}
    scoring = visual.get("scoring") or {}
    seedance = settings.get("seedance") or {}
    return RunConfig(
        output_dir=str(output_dir),
        duration=int(generation.get("default_duration_seconds") or 8),
        ratio=str(seedance.get("ratio") or "16:9"),
        resolution=str(seedance.get("resolution") or "720p"),
        generate_audio=bool(generation.get("audio", True)),
        enhanced_text_prompt=True,
        seedance_model=str(seedance.get("model") or DEFAULT_SEEDANCE_MODEL),
        visual_element_memory=True,
        visual_element_selection_mode=str(visual.get("selection_mode") or "greedy_coverage"),
        visual_element_sink_frame_count=int(visual.get("sink_frame_count") or 0),
        visual_element_max_retrieved_frames=int(visual.get("max_retrieved_frames") or 4),
        visual_element_weight_character=float(scoring.get("character_weight", 3.0)),
        visual_element_weight_scene=float(scoring.get("scene_weight", 2.0)),
        visual_element_weight_object=float(scoring.get("object_weight", 1.5)),
        visual_element_weight_reference_uncovered=float(scoring.get("reference_uncovered_weight", 1.0)),
        visual_element_weight_reference_covered=float(scoring.get("reference_covered_weight", 0.2)),
        visual_element_weight_optional=float(scoring.get("optional_weight", 0.1)),
        visual_element_weight_exclude=float(scoring.get("exclude_weight", -0.1)),
        visual_element_weight_quality_full=float(scoring.get("quality_full_weight", 1.0)),
        visual_element_weight_quality_partial=float(scoring.get("quality_partial_weight", 0.2)),
        visual_element_weight_quality_weak=float(scoring.get("quality_weak_weight", 0.1)),
        smooth_reference_seconds=float(generation.get("smooth_reference_seconds", 2.0)),
    )


def _smooth_reference_seconds(context: ShotExecutionContext) -> float:
    settings = context.input_snapshot.get("project_settings")
    if not isinstance(settings, dict):
        settings = context.bundle.get("settings") if isinstance(context.bundle, dict) else {}
    generation = settings.get("generation") if isinstance(settings, dict) else {}
    try:
        value = float((generation or {}).get("smooth_reference_seconds", 2.0))
    except (TypeError, ValueError):
        value = 2.0
    return min(max(value, MIN_SMOOTH_REFERENCE_SECONDS), 10.0)


def _visual_memory_snapshot_from_shots(shots: List[Dict[str, Any]]) -> Dict[str, Any]:
    registry_by_id: Dict[str, Dict[str, Any]] = {}
    annotations: List[Dict[str, Any]] = []
    for shot in shots:
        if shot.get("state", {}).get("status") != "completed":
            continue
        attempt = shot.get("attempt") or {}
        for row in attempt.get("visual_element_status") or []:
            element_id = row.get("element_id") or row.get("id")
            if not element_id or element_id in registry_by_id:
                continue
            registry_by_id[str(element_id)] = {
                "id": str(element_id),
                "name": str(row.get("name") or element_id),
                "type": _normalize_visual_element_type(row.get("type")),
                "introduced_at": str(row.get("introduced_at") or row.get("source_shot_id") or shot["shot_id"]),
                "notes": str(row.get("notes") or ""),
            }
        for memory in attempt.get("produced_visual_memory") or []:
            annotation = memory.get("frame_annotation")
            if isinstance(annotation, dict):
                item = dict(annotation)
                item.setdefault("holistic_description", memory.get("holistic_description") or "")
                annotations.append(item)
    return {
        "registry": list(registry_by_id.values()),
        "annotations": annotations,
        "records": [],
    }


def _normalize_visual_element_type(value: object) -> str:
    raw = str(value or "object").strip()
    aliases = {
        "environment": "scene",
        "location": "scene",
        "style": "scene",
        "action": "object",
        "other": "object",
    }
    normalized = aliases.get(raw, raw)
    if normalized in {"character", "scene", "object"}:
        return normalized
    return "object"


def _visual_status_rows(report_or_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    plan = report_or_plan.get("visual_element_plan") or report_or_plan
    registry_by_id = {
        str(item.get("id")): item
        for item in (report_or_plan.get("visual_element_registry_after") or report_or_plan.get("registry_after") or [])
        if item.get("id")
    }
    inserted_by_id = {
        str(item.get("id")): item
        for item in (report_or_plan.get("visual_element_inserted") or [])
        if item.get("id")
    }
    element_by_id = {**registry_by_id, **inserted_by_id}
    rows: List[Dict[str, Any]] = []
    for status in ("should_reference", "should_exclude", "optional_or_uncertain"):
        for item in plan.get(status) or []:
            element_id = str(item.get("id") or "")
            metadata = element_by_id.get(element_id, {})
            rows.append(
                {
                    "element_id": item.get("id"),
                    "name": item.get("name"),
                    "type": item.get("type"),
                    "introduced_at": item.get("introduced_at") or metadata.get("introduced_at") or "",
                    "notes": item.get("notes") or metadata.get("notes") or "",
                    "status": status,
                    "reason": item.get("reason") or "",
                }
            )
    for item in plan.get("new_elements") or []:
        inserted = inserted_by_id.get(str(item.get("id"))) or {}
        rows.append(
            {
                "element_id": item.get("id"),
                "name": item.get("name"),
                "type": item.get("type"),
                "introduced_at": inserted.get("introduced_at") or item.get("introduced_at") or item.get("id"),
                "notes": inserted.get("notes") or item.get("notes") or item.get("reason") or "",
                "status": "new",
                "reason": item.get("reason") or "",
            }
        )
    return rows


def _visual_reference_rows(
    *,
    store: ProjectStore,
    project_id: str,
    shots: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    shot_id_by_scene_shot = {
        (int(shot["scene_num"]), int(shot["shot_num"])): shot["shot_id"]
        for shot in shots
    }
    for index, reference in enumerate(references, start=1):
        item = dict(reference)
        source_path = str(item.get("source_path") or "")
        selection = item.get("visual_element_selection") or {}
        roles = item.get("roles") or []
        source_scene = item.get("source_scene_num")
        source_shot = item.get("source_shot_num")
        source_shot_id = item.get("source_shot_id")
        if not source_shot_id and source_scene and source_shot:
            source_shot_id = shot_id_by_scene_shot.get((int(source_scene), int(source_shot)))
        row = {
            **item,
            "reference_id": item.get("reference_id") or f"ref-{index:03d}",
            "asset_id": item.get("asset_id") or _asset_id_for_path(store, project_id, source_path),
            "source_shot_id": source_shot_id,
            "roles": roles,
            "covered_elements": _covered_selection_element_ids(selection),
            "conflict_elements": _selection_element_ids(selection, "should_exclude"),
            "score": item.get("score"),
            "reason": item.get("reference_intent") or "",
            "holistic_description": str(item.get("holistic_description") or ""),
            "reference_guidance": str(item.get("reference_guidance") or ""),
        }
        rows.append(row)
    return rows


def _covered_selection_element_ids(selection: Dict[str, Any]) -> List[str]:
    values = [
        *(selection.get("newly_covered_element_ids") or []),
        *(selection.get("already_covered_element_ids") or []),
    ]
    return [str(item) for item in values if item]


def _selection_element_ids(selection: Dict[str, Any], key: str) -> List[str]:
    values = selection.get(key) or []
    result = []
    for item in values:
        if isinstance(item, dict):
            result.append(str(item.get("id") or item.get("name") or ""))
    return [item for item in result if item]


def _asset_id_for_path(store: ProjectStore, project_id: str, source_path: str) -> str | None:
    if not source_path:
        return None
    project_dir = store.project_dir(project_id).resolve()
    try:
        source = Path(source_path).resolve()
    except OSError:
        return None
    assets = store.get_project(project_id).get("assets", {})
    for asset_id, asset in assets.items():
        try:
            path = (project_dir / str(asset.get("path"))).resolve()
        except OSError:
            continue
        if path == source:
            return str(asset_id)
    return None


def _image_asset_record(
    project_dir: Path,
    target_path: Path,
    asset_id: str,
    *,
    source_type: str,
    shot_id: str,
    attempt_id: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "kind": "image",
        "path": _relative(project_dir, target_path),
        "created_at": _now(),
        "source": {
            "type": source_type,
            "shot_id": shot_id,
            "attempt_id": attempt_id,
        },
        "metadata": metadata,
    }


def _existing_memory_paths(
    store: ProjectStore,
    project_id: str,
    shots: List[Dict[str, Any]],
    current_shot_id: str,
) -> List[str]:
    paths: List[str] = []
    for shot in shots:
        if shot["shot_id"] == current_shot_id:
            break
        if shot.get("state", {}).get("status") != "completed":
            continue
        attempt = shot.get("attempt") or {}
        for memory in attempt.get("produced_visual_memory") or []:
            asset_id = memory.get("asset_id")
            if not asset_id:
                continue
            try:
                paths.append(str(store.asset_path(project_id, str(asset_id))))
            except Exception:
                continue
    return paths


def _sink_recent_existing_memory_paths(store: ProjectStore, context: VideoPostprocessContext) -> List[str]:
    selected_path = context.attempt_dir / "reference_selection" / "selected_references.json"
    selected = read_json(selected_path, None)
    if isinstance(selected, list):
        return _asset_paths_from_reference_rows(store, context.project_id, selected)
    candidates = _sink_recent_memory_candidates(
        ReferenceSelectionContext(
            project_id=context.project_id,
            bundle=context.bundle,
            shot=context.shot,
            previous_shots=[
                shot
                for shot in context.bundle["shots"]
                if int(shot.get("order_index") or 0) < int(context.shot.get("order_index") or 0)
            ],
            attempt_id=context.attempt_id,
            attempt_dir=context.attempt_dir,
            input_snapshot=dict(context.shot.get("inputs") or {}),
            visual_element_status=context.visual_element_status,
            visual_element_details={},
        ),
        store,
    )
    memory_bank = _sink_recent_default_memory_bank(
        [str(item["source_path"]) for item in candidates],
        SINK_RECENT_MEMORY_MAX_SIZE,
        SINK_RECENT_MEMORY_FIX,
    )
    return memory_bank


def _asset_paths_from_reference_rows(
    store: ProjectStore,
    project_id: str,
    references: List[Dict[str, Any]],
) -> List[str]:
    paths: List[str] = []
    for reference in references:
        if not isinstance(reference, dict):
            continue
        asset_id = reference.get("asset_id")
        if asset_id:
            try:
                paths.append(str(store.asset_path(project_id, str(asset_id))))
                continue
            except Exception:
                pass
        source_path = reference.get("source_path")
        if source_path and Path(str(source_path)).is_file():
            paths.append(str(Path(str(source_path))))
    return paths


def _produced_memory_from_annotations(
    annotations: List[Dict[str, Any]],
    shot_id: str,
    attempt_id: str,
) -> List[Dict[str, Any]]:
    produced = []
    for index, annotation in enumerate(annotations, start=1):
        asset_id = Path(str(annotation.get("frame_path") or f"img_{shot_id}_{attempt_id}_{index:03d}")).stem
        elements = annotation.get("elements") or []
        visible_ids = [str(item.get("id")) for item in elements if item.get("id")]
        visible_names = [str(item.get("name")) for item in elements if item.get("name")]
        produced.append(
            {
                "asset_id": asset_id,
                "rank": index,
                "visible_elements": visible_ids,
                "annotation": "Visible: " + ", ".join(visible_names) if visible_names else "No known visual elements visible.",
                "holistic_description": str(annotation.get("holistic_description") or ""),
                "active_in_memory_pool": True,
                "frame_annotation": annotation,
            }
        )
    return produced


def _uses_default_last_frame_continuity(shot: Dict[str, Any], input_snapshot: Dict[str, Any]) -> bool:
    if str(input_snapshot.get("generation_mode") or "default") != "default":
        return False
    if int(shot.get("order_index") or 0) <= 0:
        return False
    return not bool(input_snapshot.get("is_cut", shot.get("inputs", {}).get("is_cut", False)))


def _uses_smooth_reference_video(shot: Dict[str, Any], input_snapshot: Dict[str, Any]) -> bool:
    if str(input_snapshot.get("generation_mode") or "default") != "smooth":
        return False
    if int(shot.get("order_index") or 0) <= 0:
        return False
    return not bool(input_snapshot.get("is_cut", shot.get("inputs", {}).get("is_cut", False)))


def _prompt_for_generation_mode(
    prompt: str,
    generation_mode: str,
    *,
    default_last_frame_continuity: bool = False,
) -> str:
    if generation_mode == "smooth":
        return (
            f"{SMOOTH_CONTINUATION_INSTRUCTION}\n\n"
            "Input media numbering for this Smooth shot:\n"
            "- Video 1 is the tail segment of the preceding raw video. The current shot must extend Video 1 forward and continue naturally from the instant after its final frame.\n"
            "- Static reference images are numbered separately as Image 1, Image 2, ... after Video 1. "
            "When the prompt below says Image 1, it means the first static image reference, not Video 1.\n\n"
            f"{prompt}"
        )
    if generation_mode == "last_frame_only":
        return (
            "Use Image 1 as the exact first-frame continuity input from the previous Seedance output. "
            "Continue naturally from that image; do not restart or replay the previous shot.\n\n"
            f"{prompt}"
        )
    if generation_mode == "default" and default_last_frame_continuity:
        return (
            "The final reference image is the previous shot's ending frame. "
            "Use this final reference image as a first-frame continuity constraint for the new video; "
            "earlier static images are visual memory references.\n\n"
            f"{prompt}"
        )
    return prompt


def _add_reference_guidance(
    context: ReferenceSelectionContext,
    references: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    rows = [{**item, "reference_guidance": str(item.get("reference_guidance") or "")} for item in references]
    eligible = [item for item in rows if str(item.get("holistic_description") or "").strip()]
    if not eligible:
        _append_style_reference_guidance(rows)
        return rows, {"status": "skipped", "reason": "no selected references with holistic_description"}, []

    prompt = _reference_guidance_prompt(context, eligible)
    try:
        raw_text, metadata = _call_ark_chat(
            messages=[{"role": "user", "content": prompt}],
            model=DEFAULT_VISUAL_ELEMENT_MODEL,
            max_tokens=VISUAL_ELEMENT_MAX_OUTPUT_TOKENS,
            timeout_seconds=120,
        )
        parsed = _extract_json_object(raw_text)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("references"), list):
            raise ValueError("reference guidance output must contain a references list")
        guidance_by_id: Dict[str, str] = {}
        for item in parsed["references"]:
            if not isinstance(item, dict):
                continue
            reference_id = str(item.get("reference_id") or "")
            if reference_id:
                guidance_by_id[reference_id] = str(item.get("reference_guidance") or "").strip()
        for row in rows:
            if str(row.get("holistic_description") or "").strip():
                row["reference_guidance"] = guidance_by_id.get(str(row.get("reference_id") or ""), "")
            else:
                row["reference_guidance"] = ""
        _append_style_reference_guidance(rows)
        details = {
            "status": "completed",
            "prompt": prompt,
            "raw_response": raw_text,
            "metadata": metadata,
            "parsed": parsed,
        }
        return rows, details, [_log("selecting_visual_references", "reference guidance generated for selected references")]
    except Exception as exc:
        for row in rows:
            row["reference_guidance"] = ""
        _append_style_reference_guidance(rows)
        details = {
            "status": "failed",
            "prompt": prompt,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        return rows, details, [_log("selecting_visual_references", f"reference guidance failed without blocking selection: {exc}")]


def _append_style_reference_guidance(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        if _reference_covers_needed_elements(row):
            continue
        guidance = str(row.get("reference_guidance") or "").strip()
        if STYLE_REFERENCE_GUIDANCE in guidance:
            continue
        row["reference_guidance"] = f"{guidance}\n{STYLE_REFERENCE_GUIDANCE}".strip()


def _reference_covers_needed_elements(row: Dict[str, Any]) -> bool:
    if row.get("covered_elements"):
        return True
    selection = row.get("visual_element_selection") if isinstance(row.get("visual_element_selection"), dict) else {}
    return bool(selection.get("should_reference"))


def _reference_guidance_prompt(context: ReferenceSelectionContext, references: List[Dict[str, Any]]) -> str:
    story_rows = _shot_context_rows(context.bundle["shots"][: int(context.shot["order_index"]) + 1])
    output_language = detect_story_language(
        item.get("video_prompt") or item.get("prompt") or "" for item in story_rows
    )
    payload = {
        "story_so_far": story_rows,
        "current_shot": {
            "scene_num": context.shot.get("scene_num"),
            "shot_num": context.shot.get("shot_num"),
            "video_prompt": context.input_snapshot.get("video_prompt"),
        },
        "visual_element_status": context.visual_element_status,
        "selected_references": [
            {
                "reference_id": item.get("reference_id"),
                "source_scene_num": item.get("source_scene_num"),
                "source_shot_num": item.get("source_shot_num"),
                "holistic_description": item.get("holistic_description") or "",
                "visible_elements": item.get("covered_elements") or item.get("visible_elements") or [],
                "coverage": item.get("visual_element_selection", {}).get("should_reference", []),
                "exclude": item.get("visual_element_selection", {}).get("should_exclude", []),
                "optional_visible": item.get("visual_element_selection", {}).get("optional_or_uncertain", []),
                "score": item.get("score"),
            }
            for item in references
        ],
    }
    return (
        "You are the historical-reference explanation assistant for a long-form video generation project. The system has already selected "
        "reference frames using objective element-coverage rules. You must not add, remove, replace, or reorder frames; only add "
        "reference_guidance for each existing reference_id.\n"
        f"{output_language_instruction(output_language)} Write reference_guidance as concise text in that language explaining why the image is useful as an overall reference for the current shot and how "
        "to use its composition, mood, or story relationship during generation. Do not repeat objective coverage scores or introduce content "
        "not visible in the image. If holistic_description is empty, reference_guidance must be an empty string.\n"
        "Return only a JSON object in this format:\n"
        '{"references":[{"reference_id":"ref-001","reference_guidance":"..."}]}\n\n'
        "Input data:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _visual_reference_prompt_context(
    bundle: Dict[str, Any],
    shot: Dict[str, Any],
    input_snapshot: Dict[str, Any],
    visual_status: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
) -> str:
    full_script_enabled = _prompt_module_enabled(input_snapshot, bundle, "full_script_context")
    visual_plan_enabled = _prompt_module_enabled(input_snapshot, bundle, "visual_element_plan")
    holistic_guidance_enabled = _prompt_module_enabled(input_snapshot, bundle, "holistic_guidance")
    should_reference_enabled = _prompt_module_enabled(input_snapshot, bundle, "should_reference")
    should_exclude_enabled = _prompt_module_enabled(input_snapshot, bundle, "should_exclude")
    predefined_references = _predefined_references_from_input(input_snapshot)
    predefined_image_count = sum(
        1 for reference in predefined_references if reference["media_type"] == "image"
    )
    predefined_lines = _predefined_reference_prompt_lines(
        predefined_references,
        smooth_video_present=_uses_smooth_reference_video(shot, input_snapshot),
    )
    reference_lines: List[str] = []
    if references:
        for index, ref in enumerate(references, start=1):
            ref_index = int(ref.get("reference_index") or index) + predefined_image_count
            reference_lines.append(
                f"Reference image {ref_index} is from Scene {ref.get('source_scene_num')} / Shot {ref.get('source_shot_num')}:"
            )
            if "visual_sink_memory" in (ref.get("roles") or []):
                reference_lines.extend(
                    [
                        "This is an early reference anchor. Use it only to stabilize overall character identity, visual style, color tone, and long-range consistency.",
                        "Do not carry old scenes, old actions, or objects not required by the current shot into the new frame.",
                    ]
                )
                continue
            holistic = str(ref.get("holistic_description") or "").strip()
            guidance = str(ref.get("reference_guidance") or "").strip()
            if holistic_guidance_enabled and holistic:
                reference_lines.append(holistic)
            if holistic_guidance_enabled and guidance:
                reference_lines.append(guidance)
            selection = ref.get("visual_element_selection") or {}
            objective_lines: List[str] = []
            if should_reference_enabled:
                objective_lines.extend(
                    [
                        "Elements to reference:",
                        _reference_names(selection, "should_reference", ref.get("covered_elements")),
                    ]
                )
            if should_exclude_enabled:
                objective_lines.extend(
                    [
                        "Elements not to introduce:",
                        _reference_names(selection, "should_exclude", ref.get("conflict_elements")),
                    ]
                )
            if objective_lines:
                reference_lines.append("Objective element constraints from the visual plan:")
                reference_lines.extend(objective_lines)
    else:
        reference_lines.append("No historical reference images are provided for this shot.")

    scene_num = shot.get("scene_num")
    shot_num = shot.get("shot_num")
    prompt = str(input_snapshot.get("video_prompt") or shot.get("inputs", {}).get("video_prompt") or "")
    force_animation = _force_animation_enabled(input_snapshot, bundle)
    reference_video_instruction = (
        "The input media includes a reference video. Generate a continuation of that video: start immediately after its final frame and naturally continue its character actions, object motion, camera direction, and camera speed while preserving the overall visual style, color tone, and visual elements."
    )
    current_task_lines = [
        "[Current Shot Generation Task]",
        *([FORCE_ANIMATION_PROMPT_PREFIX] if force_animation else []),
        f"Generate only Scene {scene_num} / Shot {shot_num}:",
        escape_prompt_line(prompt),
        *([reference_video_instruction] if _uses_smooth_reference_video(shot, input_snapshot) else []),
    ]
    overall_constraint_lines = [
        "[Overall Constraints]",
        *([FORCE_ANIMATION_PROMPT_PREFIX] if force_animation else []),
        "The current shot generation task has the highest priority.",
    ]
    if full_script_enabled:
        overall_constraint_lines.append(
            "The preceding full script is provided only for context. Do not generate story content from other shots. For historical elements not fully described in this shot, consult their descriptions and reference images in preceding shots."
        )
    if visual_plan_enabled:
        overall_constraint_lines.extend(
            [
                "Reference only visual elements explicitly specified by the visual plan. Do not add elements from reference images that are unrelated to this shot.",
                "If a reference-image instruction conflicts with the current shot description, follow the current shot description and this shot's visual-element plan.",
            ]
        )
    else:
        overall_constraint_lines.append("If a reference-image instruction conflicts with the current shot description, follow the current shot description.")
    overall_constraint_lines.extend(
        [
            "Unless explicitly required by the script, do not add pull-backs, fade-outs, dimming, or other transitions at the end of the video.",
            "Maintain a stable shot scale through the end of the clip unless the script explicitly asks otherwise.",
            "Do NOT zoom out, pull the camera back, fade out, dim the image, or add an ending transition at the end of video, unless the script explicitly asks for it, as next shot may continue the same scene.",
        ]
    )
    prompt_sections: List[str] = ["\n".join(current_task_lines)]
    if full_script_enabled:
        prompt_sections.append(
            "\n".join(
                [
                    "[Preceding Full Script]",
                    "The following shot-level descriptions cover the current shot and all preceding shots. Use them only to understand prior events, relationships, scene changes, and which historical elements should not appear in the current shot.",
                    "Generate only the specified shot. Do not generate content from other shots or introduce characters, scenes, or objects not present in the preceding script or current shot.",
                    story_prompt_outline(_story_script_through_shot(bundle, shot)),
                ]
            )
        )
    if visual_plan_enabled:
        prompt_sections.append(
            "\n".join(
                [
                    "[Visual Element Plan for This Shot]",
                    "Historical elements to reference and keep consistent:",
                    _status_names(visual_status, "should_reference"),
                    "Historical elements not to introduce:",
                    _status_names(visual_status, "should_exclude"),
                    "Optional or uncertain visual elements:",
                    _status_names(visual_status, "optional_or_uncertain"),
                    "New visual elements introduced in this shot:",
                    _status_names(visual_status, "new"),
                ]
            )
        )
    if predefined_lines["image"]:
        prompt_sections.append(
            "\n".join(
                [
                    "[Predefined Reference Image Instructions]",
                    "The following reference images are predefined inputs for this shot, supplied by the user or script. Use each image according to its instructions; do not mechanically reproduce content not required by the current shot.",
                    *predefined_lines["image"],
                ]
            )
        )
    if predefined_lines["video"]:
        prompt_sections.append(
            "\n".join(
                [
                    "[Predefined Reference Video Instructions]",
                    "The following reference videos are predefined inputs for this shot. Use each video according to its instructions and the current shot task.",
                    *predefined_lines["video"],
                ]
            )
        )
    if predefined_lines["audio"]:
        prompt_sections.append(
            "\n".join(
                [
                    "[Predefined Reference Audio Instructions]",
                    "The following reference audio clips are predefined inputs for this shot. Use each audio clip according to its instructions and the current shot task.",
                    *predefined_lines["audio"],
                ]
            )
        )
    prompt_sections.append(
        "\n".join(
            [
                "[Historical Reference Image Instructions]",
                "The following images are keyframes extracted from previously generated shots. Use them only to preserve specified visual elements or visual style; they do not mean that the entire image should be reproduced.",
                *reference_lines,
            ]
        )
    )
    prompt_sections.append("\n".join(overall_constraint_lines))
    prompt_text = "\n\n".join(prompt_sections)
    return prompt_text


def _predefined_references_from_input(input_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    references = input_snapshot.get("predefined_references") if isinstance(input_snapshot, dict) else []
    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(references or [], start=1):
        if not isinstance(item, dict):
            continue
        asset_id = str(item.get("asset_id") or "").strip()
        media_type = str(item.get("media_type") or "image").strip().lower()
        if media_type not in {"image", "audio", "video"}:
            continue
        media_path = str(item.get("media_path") or item.get("image_path") or item.get("source_path") or "").strip()
        if not asset_id and not media_path:
            continue
        reference = {
            "id": str(item.get("id") or f"pref-{index:04d}"),
            "reference_id": str(item.get("id") or f"pref-{index:04d}"),
            "asset_id": asset_id,
            "media_type": media_type,
            "source_path": media_path,
            "media_path": media_path,
            "label": str(item.get("label") or f"Predefined {media_type} {index}").strip(),
            "guidance": str(item.get("guidance") or "").strip(),
            "roles": [f"predefined_{media_type}_reference"],
            "source_type": "predefined_reference",
        }
        if media_type == "image":
            reference["image_path"] = media_path
        normalized.append(reference)
    return normalized


def _predefined_non_image_references(input_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [reference for reference in _predefined_references_from_input(input_snapshot) if reference["media_type"] != "image"]


def _predefined_media_summary_items(input_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for reference in _predefined_non_image_references(input_snapshot):
        media_type = reference["media_type"]
        items.append(
            {
                "type": f"{media_type}_asset_pending_publication",
                "role": f"reference_{media_type}",
                "metadata": dict(reference),
            }
        )
    return items


def _predefined_reference_prompt_lines(
    references: List[Dict[str, Any]],
    *,
    smooth_video_present: bool,
) -> Dict[str, List[str]]:
    lines = {"image": [], "audio": [], "video": []}
    counts = {"image": 0, "audio": 0, "video": 1 if smooth_video_present else 0}
    for reference in references:
        media_type = str(reference.get("media_type") or "image")
        if media_type not in lines:
            continue
        counts[media_type] += 1
        label = str(reference.get("label") or f"Predefined {media_type} {counts[media_type]}").strip()
        guidance = str(reference.get("guidance") or "").strip()
        lines[media_type].append(f"Reference {media_type} {counts[media_type]}: {label}")
        if guidance:
            lines[media_type].append(guidance)
    return lines


def _combined_static_references(context: Any, selected_references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    predefined = [
        reference
        for reference in _predefined_references_from_input(context.input_snapshot)
        if reference["media_type"] == "image"
    ]
    if len(predefined) > MAX_REFERENCE_IMAGES:
        raise BackendExecutionError(
            f"Predefined references exceed Seedance's {MAX_REFERENCE_IMAGES}-image limit"
        )
    combined: List[Dict[str, Any]] = []
    for index, reference in enumerate(predefined, start=1):
        item = dict(reference)
        item["reference_index"] = index
        combined.append(item)
    offset = len(combined)
    for index, reference in enumerate(selected_references or [], start=1):
        item = dict(reference)
        item["reference_index"] = offset + index
        combined.append(item)
    if len(combined) > MAX_REFERENCE_IMAGES:
        raise BackendExecutionError(
            f"Static reference images exceed Seedance's {MAX_REFERENCE_IMAGES}-image limit"
        )
    return combined


def _effective_reference_budget(context: ReferenceSelectionContext) -> Dict[str, Any]:
    predefined_count = sum(
        1 for reference in _predefined_references_from_input(context.input_snapshot) if reference["media_type"] == "image"
    )
    if predefined_count > MAX_REFERENCE_IMAGES:
        raise BackendExecutionError(
            f"Predefined references exceed Seedance's {MAX_REFERENCE_IMAGES}-image limit"
        )
    memory_settings = (context.bundle.get("settings") or {}).get("visual_element_memory") or {}
    try:
        configured_sink = int(memory_settings.get("sink_frame_count", 0))
    except (TypeError, ValueError):
        configured_sink = 0
    try:
        configured_retrieved = int(memory_settings.get("max_retrieved_frames", 4))
    except (TypeError, ValueError):
        configured_retrieved = 4
    remaining = max(0, MAX_REFERENCE_IMAGES - predefined_count)
    effective_sink = min(max(configured_sink, 0), remaining)
    effective_retrieved = min(max(configured_retrieved, 0), max(0, remaining - effective_sink))
    return {
        "max_static_reference_images": MAX_REFERENCE_IMAGES,
        "predefined_reference_count": predefined_count,
        "configured_sink_frame_count": max(configured_sink, 0),
        "configured_max_retrieved_frames": max(configured_retrieved, 0),
        "remaining_slots_after_predefined": remaining,
        "effective_sink_frame_count": effective_sink,
        "effective_max_retrieved_frames": effective_retrieved,
    }


def _naive_reference_budget(context: ReferenceSelectionContext) -> Dict[str, Any]:
    budget = _effective_reference_budget(context)
    budget["effective_sink_frame_count"] = 0
    budget["effective_max_retrieved_frames"] = min(
        int(budget["configured_max_retrieved_frames"]),
        int(budget["remaining_slots_after_predefined"]),
    )
    return budget


def _sink_recent_reference_budget(context: ReferenceSelectionContext) -> Dict[str, Any]:
    predefined_count = sum(
        1 for reference in _predefined_references_from_input(context.input_snapshot) if reference["media_type"] == "image"
    )
    if predefined_count > MAX_REFERENCE_IMAGES:
        raise BackendExecutionError(
            f"Predefined references exceed Seedance's {MAX_REFERENCE_IMAGES}-image limit"
        )
    default_last_frame_reserved = 1 if _uses_default_last_frame_continuity(context.shot, context.input_snapshot) else 0
    remaining = max(0, MAX_REFERENCE_IMAGES - predefined_count - default_last_frame_reserved)
    effective_max_memory_size = min(SINK_RECENT_MEMORY_MAX_SIZE, remaining)
    return {
        "max_static_reference_images": MAX_REFERENCE_IMAGES,
        "predefined_reference_count": predefined_count,
        "default_last_frame_reserved": default_last_frame_reserved,
        "configured_sink_recent_max_memory_size": SINK_RECENT_MEMORY_MAX_SIZE,
        "configured_sink_recent_fix": SINK_RECENT_MEMORY_FIX,
        "remaining_slots_after_predefined_and_continuity": remaining,
        "effective_max_memory_size": effective_max_memory_size,
        "effective_sink_frame_count": min(SINK_RECENT_MEMORY_FIX, effective_max_memory_size),
        "effective_recent_frame_count": max(0, effective_max_memory_size - min(SINK_RECENT_MEMORY_FIX, effective_max_memory_size)),
    }


def _maybe_force_animation_prompt(prompt: str, input_snapshot: Dict[str, Any], bundle: Dict[str, Any] | None = None) -> str:
    if not _force_animation_enabled(input_snapshot, bundle):
        return prompt
    text = str(prompt or "").lstrip()
    if FORCE_ANIMATION_PROMPT_PREFIX in text:
        return str(prompt or "")
    return f"{FORCE_ANIMATION_PROMPT_PREFIX}\n\n{prompt}"


def _force_animation_enabled(input_snapshot: Dict[str, Any], bundle: Dict[str, Any] | None = None) -> bool:
    settings = input_snapshot.get("project_settings") if isinstance(input_snapshot, dict) else None
    if not isinstance(settings, dict) and isinstance(bundle, dict):
        settings = bundle.get("settings")
    generation = settings.get("generation") if isinstance(settings, dict) else {}
    return bool((generation or {}).get("force_animation_style", True))


def _prompt_module_enabled(input_snapshot: Dict[str, Any], bundle: Dict[str, Any] | None, key: str) -> bool:
    settings = input_snapshot.get("project_settings") if isinstance(input_snapshot, dict) else None
    if not isinstance(settings, dict) and isinstance(bundle, dict):
        settings = bundle.get("settings")
    modules = settings.get("prompt_modules") if isinstance(settings, dict) else {}
    return bool((modules or {}).get(key, True))


def _shot_context_rows(shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "scene_num": item.get("scene_num"),
            "shot_num": item.get("shot_num"),
            "video_prompt": item.get("inputs", {}).get("video_prompt", ""),
        }
        for item in shots
    ]


def _status_names(rows: List[Dict[str, Any]], status: str) -> str:
    names = [str(item.get("name") or item.get("element_id")) for item in rows if item.get("status") == status]
    return "\n".join(f"- {name}" for name in names) if names else "- none"


def _reference_names(selection: Dict[str, Any], key: str, fallback: Any = None) -> str:
    values = selection.get(key) if isinstance(selection, dict) else None
    fallback_ids = {str(item) for item in fallback} if key == "should_reference" and isinstance(fallback, list) else set()
    names: List[str] = []
    if isinstance(values, list):
        for item in values:
            if isinstance(item, dict):
                if fallback_ids and str(item.get("id") or "") not in fallback_ids:
                    continue
                name = item.get("name") or item.get("id")
            else:
                name = item
                if fallback_ids and str(name) not in fallback_ids:
                    continue
            if name:
                names.append(str(name))
    if not names and isinstance(fallback, list):
        names = [str(item) for item in fallback if item]
    return "\n".join(f"- {name}" for name in names) if names else "- none"


def _previous_completed_shot(context: ShotExecutionContext) -> Dict[str, Any]:
    if not context.previous_shots:
        raise BackendExecutionError("This generation mode requires a completed predecessor")
    previous = context.previous_shots[-1]
    if previous.get("state", {}).get("status") != "completed" or not previous.get("attempt"):
        raise BackendExecutionError("Previous shot is not completed")
    return previous


from .logging import _log, _short_name
from .util import _env_flag, _relative
def _fake_prompt(
    input_snapshot: Dict[str, Any],
    visual_status: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
) -> str:
    lines = [
        input_snapshot["video_prompt"],
        "",
        "Visual element guidance:",
    ]
    for element in visual_status:
        lines.append(f"- {element['name']}: {element['status']} ({element['reason']})")
    predefined = _predefined_references_from_input(input_snapshot)
    if predefined:
        lines.append("")
        lines.append("Predefined references:")
        for index, reference in enumerate(predefined, start=1):
            lines.append(f"- Image {index}: {reference.get('label')}; {reference.get('guidance')}")
    if references:
        lines.append("")
        lines.append("Historical references:")
        for index, reference in enumerate(references, start=1):
            image_index = len(predefined) + index
            lines.append(
                f"- Image {image_index}: source shot {reference['source_shot_id']}, "
                f"use {', '.join(reference['covered_elements'])}."
            )
    return "\n".join(lines)
