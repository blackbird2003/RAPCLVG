import logging
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from extract_keyframes import _clip_preprocess_tensor, _get_clip_model


KEYFRAME_RE = re.compile(r"(?P<scene>\d+)_(?P<shot>\d+)_keyframe(?P<rank>\d+)\.jpg$")


@dataclass(frozen=True)
class RetrievalHit:
    path: str
    frame_score: float
    video_score: float
    score: float
    source_prompt: str


def build_default_memory_bank(
    all_memory: Sequence[str],
    max_memory_size: int,
    fix: int,
) -> List[str]:
    memory_bank = list(all_memory)
    if len(memory_bank) <= max_memory_size:
        return memory_bank

    fixed = max(0, min(fix, max_memory_size))
    recent = max_memory_size - fixed
    return memory_bank[:fixed] + (memory_bank[-recent:] if recent > 0 else [])


def get_memory_query(scene: dict, shot_index: int, prompt: str) -> str:
    for key in ("memory_queries", "retrieval_queries", "first_frame_prompt"):
        values = scene.get(key)
        if isinstance(values, list) and shot_index < len(values) and values[shot_index]:
            return str(values[shot_index])
    return prompt


def build_shot_prompt_map(story_script: dict) -> Dict[Tuple[int, int], str]:
    prompt_map: Dict[Tuple[int, int], str] = {}
    for scene in story_script.get("scenes", []):
        scene_num = int(scene.get("scene_num", len(prompt_map) + 1))
        prompts = scene.get("video_prompts", [])
        for idx, prompt in enumerate(prompts, start=1):
            prompt_map[(scene_num, idx)] = str(prompt)
    return prompt_map


def parse_keyframe_name(path: str) -> Optional[Tuple[int, int, int]]:
    match = KEYFRAME_RE.search(os.path.basename(path))
    if not match:
        return None
    return (
        int(match.group("scene")),
        int(match.group("shot")),
        int(match.group("rank")),
    )


def _dedupe(paths: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _load_image_chw(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _basename_list(paths: Iterable[str]) -> List[str]:
    return [os.path.basename(path) for path in paths]


def _hit_to_dict(hit: RetrievalHit) -> dict:
    return {
        "path": hit.path,
        "file": os.path.basename(hit.path),
        "score": hit.score,
        "frame_score": hit.frame_score,
        "video_score": hit.video_score,
        "source_prompt": hit.source_prompt,
    }


def _write_retrieval_jsonl(
    all_memory: Sequence[str],
    current_prompt: str,
    memory_query: str,
    max_memory_size: int,
    fix: int,
    retrieval_top_k: int,
    frame_weight: float,
    video_weight: float,
    min_score: Optional[float],
    clip_device: str,
    sink: Sequence[str],
    recent: Sequence[str],
    candidates: Sequence[str],
    hits: Sequence[RetrievalHit],
    selected_hits: Sequence[RetrievalHit],
    memory_bank: Sequence[str],
    policy: str,
    target_shot_id: Optional[str] = None,
    scene_num: Optional[int] = None,
    shot_num: Optional[int] = None,
    retrieval_log_path: Optional[str] = None,
    memory_policy: Optional[dict] = None,
) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    if not all_memory:
        return

    output_dir = os.path.dirname(os.path.abspath(all_memory[0]))
    log_path = retrieval_log_path or os.path.join(output_dir, "prompt_retrieval_log.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    selected_files = {os.path.basename(hit.path) for hit in selected_hits}
    candidate_files = set(_basename_list(candidates))
    sink_files = set(_basename_list(sink))
    recent_files = set(_basename_list(recent))
    scored = []
    for hit in hits:
        item = _hit_to_dict(hit)
        file_name = item["file"]
        item["selected"] = file_name in selected_files
        item["role"] = []
        if file_name in sink_files:
            item["role"].append("sink")
        if file_name in recent_files:
            item["role"].append("recent")
        if file_name in candidate_files:
            item["role"].append("candidate")
        if item["selected"]:
            item["role"].append("retrieved")
        scored.append(item)

    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_shot_id": target_shot_id,
        "scene_num": scene_num,
        "shot_num": shot_num,
        "policy": policy,
        "current_prompt": current_prompt,
        "memory_query": memory_query,
        "score_formula": "score = frame_weight * frame_score + video_weight * video_score",
        "weights": {
            "frame_weight": frame_weight,
            "video_weight": video_weight,
            "min_score": min_score,
        },
        "config": {
            "max_memory_size": max_memory_size,
            "fix": fix,
            "retrieval_top_k": retrieval_top_k,
            "clip_device": clip_device,
            "memory_policy": memory_policy,
        },
        "counts": {
            "all_memory": len(all_memory),
            "sink": len(sink),
            "recent": len(recent),
            "candidates": len(candidates),
            "selected": len(selected_hits),
            "final_memory": len(memory_bank),
        },
        "all_memory": _basename_list(all_memory),
        "sink_memory": _basename_list(sink),
        "recent_memory": _basename_list(recent),
        "candidate_memory": _basename_list(candidates),
        "selected": [_hit_to_dict(hit) for hit in selected_hits],
        "scored_keyframes": scored,
        "final_memory": _basename_list(memory_bank),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@torch.no_grad()
def _encode_texts(model, device: str, texts: Sequence[str]) -> torch.Tensor:
    import clip

    tokens = clip.tokenize(list(texts), truncate=True).to(device)
    features = model.encode_text(tokens)
    return F.normalize(features.float(), dim=-1)


@torch.no_grad()
def _encode_images(
    model,
    device: str,
    dtype: torch.dtype,
    paths: Sequence[str],
) -> torch.Tensor:
    images = []
    for path in paths:
        image = _load_image_chw(path).to(device)
        images.append(_clip_preprocess_tensor(image).to(device=device, dtype=dtype))
    batch = torch.cat(images, dim=0)
    features = model.encode_image(batch)
    return F.normalize(features.float(), dim=-1)


def rank_prompt_aware_keyframes(
    candidates: Sequence[str],
    memory_query: str,
    current_prompt: str,
    shot_prompt_map: Dict[Tuple[int, int], str],
    frame_weight: float = 0.7,
    video_weight: float = 0.3,
    clip_device: str = "cpu",
) -> List[RetrievalHit]:
    if not candidates:
        return []

    model, device, dtype = _get_clip_model(device=clip_device)
    image_features = _encode_images(model, device, dtype, candidates)
    query_feature = _encode_texts(model, device, [memory_query])[0]
    frame_scores = (image_features @ query_feature).detach().cpu().tolist()

    source_prompts = []
    for path in candidates:
        parsed = parse_keyframe_name(path)
        source_prompts.append(
            shot_prompt_map.get((parsed[0], parsed[1]), "") if parsed else ""
        )

    text_features = _encode_texts(model, device, [current_prompt] + source_prompts)
    current_feature = text_features[0]
    source_features = text_features[1:]
    video_scores = (source_features @ current_feature).detach().cpu().tolist()

    hits = []
    for path, frame_score, video_score, source_prompt in zip(
        candidates, frame_scores, video_scores, source_prompts
    ):
        score = frame_weight * float(frame_score) + video_weight * float(video_score)
        hits.append(
            RetrievalHit(
                path=path,
                frame_score=float(frame_score),
                video_score=float(video_score),
                score=score,
                source_prompt=source_prompt,
            )
        )
    return sorted(hits, key=lambda hit: hit.score, reverse=True)


def build_prompt_aware_memory_bank(
    all_memory: Sequence[str],
    current_prompt: str,
    memory_query: str,
    story_script: dict,
    max_memory_size: int,
    fix: int,
    retrieval_top_k: int,
    frame_weight: float = 0.7,
    video_weight: float = 0.3,
    min_score: Optional[float] = None,
    clip_device: str = "cpu",
    logger: Optional[logging.Logger] = None,
    target_shot_id: Optional[str] = None,
    scene_num: Optional[int] = None,
    shot_num: Optional[int] = None,
    retrieval_log_path: Optional[str] = None,
    explicit_policy: bool = False,
    include_sink: bool = True,
    include_recent: bool = True,
) -> Tuple[List[str], List[RetrievalHit]]:
    all_memory = list(all_memory)
    if explicit_policy:
        fixed_budget = max(0, min(fix, max_memory_size)) if include_sink else 0
        retrieved_budget = max(
            0, min(retrieval_top_k, max_memory_size - fixed_budget)
        )
        recent_budget = (
            max(0, max_memory_size - fixed_budget - retrieved_budget)
            if include_recent
            else 0
        )
        sink = all_memory[:fixed_budget]
        recent = all_memory[-recent_budget:] if recent_budget > 0 else []
        protected = set(sink + recent)
        candidates = [path for path in all_memory if path not in protected]
        hits = rank_prompt_aware_keyframes(
            candidates=candidates,
            memory_query=memory_query,
            current_prompt=current_prompt,
            shot_prompt_map=build_shot_prompt_map(story_script),
            frame_weight=frame_weight,
            video_weight=video_weight,
            clip_device=clip_device,
        )
        if min_score is not None:
            hits = [hit for hit in hits if hit.score >= min_score]
        selected_hits = hits[:retrieved_budget]
        retrieved = sorted(hit.path for hit in selected_hits)
        memory_bank = _dedupe(sink + retrieved + recent)
        _write_retrieval_jsonl(
            all_memory=all_memory,
            current_prompt=current_prompt,
            memory_query=memory_query,
            max_memory_size=max_memory_size,
            fix=fix,
            retrieval_top_k=retrieval_top_k,
            frame_weight=frame_weight,
            video_weight=video_weight,
            min_score=min_score,
            clip_device=clip_device,
            sink=sink,
            recent=recent,
            candidates=candidates,
            hits=hits,
            selected_hits=selected_hits,
            memory_bank=memory_bank,
            policy="controlled_sink_retrieve_recent",
            target_shot_id=target_shot_id,
            scene_num=scene_num,
            shot_num=shot_num,
            retrieval_log_path=retrieval_log_path,
            memory_policy={
                "sink": include_sink,
                "retrieve": True,
                "recent": include_recent,
            },
        )
        return memory_bank, selected_hits
    if retrieval_top_k <= 0:
        memory_bank = build_default_memory_bank(all_memory, max_memory_size, fix)
        _write_retrieval_jsonl(
            all_memory=all_memory,
            current_prompt=current_prompt,
            memory_query=memory_query,
            max_memory_size=max_memory_size,
            fix=fix,
            retrieval_top_k=retrieval_top_k,
            frame_weight=frame_weight,
            video_weight=video_weight,
            min_score=min_score,
            clip_device=clip_device,
            sink=[],
            recent=[],
            candidates=[],
            hits=[],
            selected_hits=[],
            memory_bank=memory_bank,
            policy="retrieval_disabled",
            target_shot_id=target_shot_id,
            scene_num=scene_num,
            shot_num=shot_num,
            retrieval_log_path=retrieval_log_path,
        )
        return memory_bank, []

    if len(all_memory) <= max_memory_size:
        memory_bank = build_default_memory_bank(all_memory, max_memory_size, fix)
        hits = rank_prompt_aware_keyframes(
            candidates=all_memory,
            memory_query=memory_query,
            current_prompt=current_prompt,
            shot_prompt_map=build_shot_prompt_map(story_script),
            frame_weight=frame_weight,
            video_weight=video_weight,
            clip_device=clip_device,
        )
        if min_score is not None:
            hits = [hit for hit in hits if hit.score >= min_score]
        _write_retrieval_jsonl(
            all_memory=all_memory,
            current_prompt=current_prompt,
            memory_query=memory_query,
            max_memory_size=max_memory_size,
            fix=fix,
            retrieval_top_k=retrieval_top_k,
            frame_weight=frame_weight,
            video_weight=video_weight,
            min_score=min_score,
            clip_device=clip_device,
            sink=[],
            recent=[],
            candidates=all_memory,
            hits=hits,
            selected_hits=[],
            memory_bank=memory_bank,
            policy="all_memory_within_budget",
            target_shot_id=target_shot_id,
            scene_num=scene_num,
            shot_num=shot_num,
            retrieval_log_path=retrieval_log_path,
        )
        return memory_bank, []

    fixed_budget = max(0, min(fix, max_memory_size))
    retrieved_budget = max(0, min(retrieval_top_k, max_memory_size - fixed_budget))
    recent_budget = max(0, max_memory_size - fixed_budget - retrieved_budget)

    sink = all_memory[:fixed_budget]
    recent = all_memory[-recent_budget:] if recent_budget > 0 else []
    protected = set(sink + recent)
    candidates = [path for path in all_memory if path not in protected]

    hits = rank_prompt_aware_keyframes(
        candidates=candidates,
        memory_query=memory_query,
        current_prompt=current_prompt,
        shot_prompt_map=build_shot_prompt_map(story_script),
        frame_weight=frame_weight,
        video_weight=video_weight,
        clip_device=clip_device,
    )
    if min_score is not None:
        hits = [hit for hit in hits if hit.score >= min_score]

    selected_hits = hits[:retrieved_budget]
    retrieved = sorted([hit.path for hit in selected_hits])
    memory_bank = _dedupe(sink + retrieved + recent)
    _write_retrieval_jsonl(
        all_memory=all_memory,
        current_prompt=current_prompt,
        memory_query=memory_query,
        max_memory_size=max_memory_size,
        fix=fix,
        retrieval_top_k=retrieval_top_k,
        frame_weight=frame_weight,
        video_weight=video_weight,
        min_score=min_score,
        clip_device=clip_device,
        sink=sink,
        recent=recent,
        candidates=candidates,
        hits=hits,
        selected_hits=selected_hits,
        memory_bank=memory_bank,
        policy="sink_retrieved_recent",
        target_shot_id=target_shot_id,
        scene_num=scene_num,
        shot_num=shot_num,
        retrieval_log_path=retrieval_log_path,
    )

    if logger is not None and selected_hits:
        for hit in selected_hits:
            logger.info(
                "Prompt-aware retrieval selected %s score=%.4f frame=%.4f video=%.4f",
                os.path.basename(hit.path),
                hit.score,
                hit.frame_score,
                hit.video_score,
            )

    return memory_bank, selected_hits
