from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json5

from extract_keyframes import save_keyframes
from memory_query_llm import MemoryQueryGenerator
from prompt_retrieval import (
    build_default_memory_bank,
    build_prompt_aware_memory_bank,
    build_shot_prompt_map,
    get_memory_query,
    parse_keyframe_name,
)
from seedance_client import SeedanceClient

from .artifacts import ArtifactStore, redact_urls
from .executor import complete_shot, prepare_shot, submit_shot
from .models import RunConfig, ShotRunRecord, ShotSpec
from .prompting import SMOOTH_CONTINUATION_INSTRUCTION, image_item_with_metadata, video_item_with_metadata
from .reporting import write_memory_report
from .visual_element_memory import VisualElementMemory


def init_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "seedance_pipeline.log"), mode="a", encoding="utf-8"),
        ],
    )


def iter_shots(story_script: dict) -> List[ShotSpec]:
    shots: List[ShotSpec] = []
    for scene in story_script["scenes"]:
        scene_num = int(scene["scene_num"])
        cuts = scene.get("cut", [True] * len(scene["video_prompts"]))
        for idx, prompt in enumerate(scene["video_prompts"]):
            shots.append(
                ShotSpec(
                    scene=scene,
                    scene_num=scene_num,
                    shot_num=idx + 1,
                    prompt=str(prompt),
                    is_cut=bool(cuts[idx]),
                    first_frame_prompt=get_first_frame_prompt(scene, idx),
                    duration_seconds=get_shot_duration(scene, idx),
                )
            )
    return shots


def get_shot_duration(scene: dict, shot_index: int) -> Optional[int]:
    values = scene.get("durations")
    if isinstance(values, list) and shot_index < len(values) and values[shot_index] is not None:
        return int(values[shot_index])
    return None


def get_first_frame_prompt(scene: dict, shot_index: int) -> str:
    values = scene.get("first_frame_prompt")
    if isinstance(values, list) and shot_index < len(values) and values[shot_index]:
        return str(values[shot_index])
    return ""


def build_reference_images(
    config: RunConfig,
    store: ArtifactStore,
    story_script: dict,
    shot: ShotSpec,
    is_first: bool,
    memory_query_generator: Optional[MemoryQueryGenerator],
    previous_source: Optional[Tuple[int, int, str]],
) -> List[dict]:
    if is_first:
        return []
    memory_records = select_memory(
        config=config,
        store=store,
        story_script=story_script,
        shot=shot,
        memory_query_generator=memory_query_generator,
    )
    refs = []
    if not shot.is_cut:
        last_frame = os.path.join(config.output_dir, "last_frame.jpg")
        if os.path.exists(last_frame):
            previous_scene, previous_shot, previous_prompt = previous_source or (None, None, "")
            refs.append(
                image_item_with_metadata(
                    last_frame,
                    {
                        "label": "previous_last_frame",
                        "roles": ["previous_last_frame"],
                        "source_path": last_frame,
                        "file": os.path.basename(last_frame),
                        "source_scene_num": previous_scene,
                        "source_shot_num": previous_shot,
                        "source_prompt": previous_prompt,
                    },
                )
            )
    for metadata in memory_records:
        if len(refs) >= config.max_reference_images:
            break
        refs.append(image_item_with_metadata(metadata["source_path"], metadata))
    for idx, item in enumerate(refs, start=1):
        item.setdefault("metadata", {})["reference_index"] = idx
    store.append_reference_records(shot, [item.get("metadata", {}) for item in refs])
    return refs


def select_memory(
    config: RunConfig,
    store: ArtifactStore,
    story_script: dict,
    shot: ShotSpec,
    memory_query_generator: Optional[MemoryQueryGenerator],
    memory_policy: Optional[Dict[str, bool]] = None,
) -> List[dict]:
    all_memory = store.memory_keyframes()
    shot_prompt_map = build_shot_prompt_map(story_script)
    controlled = memory_policy is not None
    sink_enabled = True if memory_policy is None else bool(memory_policy.get("sink"))
    retrieve_enabled = (
        config.prompt_retrieval and shot.is_cut
        if memory_policy is None
        else bool(memory_policy.get("retrieve"))
    )
    recent_enabled = True if memory_policy is None else bool(memory_policy.get("recent"))
    if retrieve_enabled:
        if config.llm_memory_query:
            if memory_query_generator is None:
                raise RuntimeError("--llm_memory_query requires a MemoryQueryGenerator")
            memory_query = memory_query_generator.get_or_generate(
                scene_num=shot.scene_num,
                shot_num=shot.shot_num,
                video_prompt=shot.prompt,
                first_frame_prompt=shot.first_frame_prompt,
            )
            logging.info("LLM-generated memory query for Scene %d / Shot %d: %s", shot.scene_num, shot.shot_num, memory_query)
        else:
            memory_query = get_memory_query(shot.scene, shot.shot_num - 1, shot.prompt)
        memory_bank, hits = build_prompt_aware_memory_bank(
            all_memory=all_memory,
            current_prompt=shot.prompt,
            memory_query=memory_query,
            story_script=story_script,
            max_memory_size=config.max_memory_size,
            fix=config.fix,
            retrieval_top_k=config.retrieval_top_k,
            frame_weight=config.retrieval_frame_weight,
            video_weight=config.retrieval_video_weight,
            min_score=config.retrieval_min_score,
            clip_device=config.retrieval_clip_device,
            logger=logging,
            target_shot_id=f"scene-{shot.scene_num}-shot-{shot.shot_num}",
            scene_num=shot.scene_num,
            shot_num=shot.shot_num,
            retrieval_log_path=os.path.join(config.output_dir, "prompt_retrieval_log.jsonl"),
            explicit_policy=controlled,
            include_sink=sink_enabled,
            include_recent=recent_enabled,
        )
        logging.info("Prompt-aware memory query: %s | selected=%d | memory=%d/%d", memory_query, len(hits), len(memory_bank), len(all_memory))
        role_by_path = (
            controlled_memory_roles(
                all_memory=all_memory,
                memory_bank=memory_bank,
                selected_hits=hits,
                max_memory_size=config.max_memory_size,
                fix=config.fix,
                retrieval_top_k=config.retrieval_top_k,
                sink_enabled=sink_enabled,
                retrieve_enabled=True,
                recent_enabled=recent_enabled,
            )
            if controlled
            else prompt_retrieval_roles(
                all_memory=all_memory,
                memory_bank=memory_bank,
                selected_hits=hits,
                max_memory_size=config.max_memory_size,
                fix=config.fix,
                retrieval_top_k=config.retrieval_top_k,
            )
        )
        hit_by_path = {hit.path: hit for hit in hits}
    else:
        if controlled:
            memory_bank = controlled_memory_bank(
                all_memory,
                config.max_memory_size,
                config.fix,
                sink_enabled=sink_enabled,
                recent_enabled=recent_enabled,
            )
            role_by_path = controlled_memory_roles(
                all_memory=all_memory,
                memory_bank=memory_bank,
                selected_hits=[],
                max_memory_size=config.max_memory_size,
                fix=config.fix,
                retrieval_top_k=0,
                sink_enabled=sink_enabled,
                retrieve_enabled=False,
                recent_enabled=recent_enabled,
            )
        else:
            memory_bank = build_default_memory_bank(all_memory, config.max_memory_size, config.fix)
            role_by_path = default_memory_roles(memory_bank, config.max_memory_size, config.fix)
        hit_by_path = {}

    records = []
    for path in memory_bank:
        metadata = source_metadata_for_keyframe(path, shot_prompt_map)
        metadata["label"] = "memory_keyframe"
        metadata["roles"] = role_by_path.get(path, ["default_memory"])
        if path in hit_by_path:
            hit = hit_by_path[path]
            metadata.update(
                {
                    "score": hit.score,
                    "frame_score": hit.frame_score,
                    "video_score": hit.video_score,
                }
            )
        records.append(metadata)
    return records


def source_metadata_for_keyframe(path: str, shot_prompt_map: Dict[Tuple[int, int], str]) -> dict:
    parsed = parse_keyframe_name(path)
    metadata = {
        "source_path": path,
        "file": os.path.basename(path),
    }
    if parsed:
        source_scene, source_shot, keyframe_rank = parsed
        metadata.update(
            {
                "source_scene_num": source_scene,
                "source_shot_num": source_shot,
                "keyframe_rank": keyframe_rank,
                "source_prompt": shot_prompt_map.get((source_scene, source_shot), ""),
            }
        )
    return metadata


def default_memory_roles(all_memory: List[str], max_memory_size: int, fix: int) -> Dict[str, List[str]]:
    if not all_memory:
        return {}
    fixed = max(0, min(fix, max_memory_size, len(all_memory)))
    role_by_path: Dict[str, List[str]] = {}
    for path in all_memory[:fixed]:
        role_by_path.setdefault(path, []).append("early_sink_memory")
    for path in all_memory[fixed:]:
        role_by_path.setdefault(path, []).append("recent_window_memory")
    return role_by_path


def controlled_memory_bank(
    all_memory: List[str],
    max_memory_size: int,
    fix: int,
    *,
    sink_enabled: bool,
    recent_enabled: bool,
) -> List[str]:
    fixed_budget = max(0, min(fix, max_memory_size)) if sink_enabled else 0
    recent_budget = max(0, max_memory_size - fixed_budget) if recent_enabled else 0
    sink = all_memory[:fixed_budget]
    recent = all_memory[-recent_budget:] if recent_budget > 0 else []
    return list(dict.fromkeys(sink + recent))


def controlled_memory_roles(
    *,
    all_memory: List[str],
    memory_bank: List[str],
    selected_hits: List,
    max_memory_size: int,
    fix: int,
    retrieval_top_k: int,
    sink_enabled: bool,
    retrieve_enabled: bool,
    recent_enabled: bool,
) -> Dict[str, List[str]]:
    fixed_budget = max(0, min(fix, max_memory_size)) if sink_enabled else 0
    retrieved_budget = (
        max(0, min(retrieval_top_k, max_memory_size - fixed_budget))
        if retrieve_enabled
        else 0
    )
    recent_budget = (
        max(0, max_memory_size - fixed_budget - retrieved_budget)
        if recent_enabled
        else 0
    )
    sink = set(all_memory[:fixed_budget])
    retrieved = {hit.path for hit in selected_hits[:retrieved_budget]}
    recent = set(all_memory[-recent_budget:]) if recent_budget > 0 else set()
    roles: Dict[str, List[str]] = {}
    for path in memory_bank:
        if path in sink:
            roles.setdefault(path, []).append("early_sink_memory")
        if path in retrieved:
            roles.setdefault(path, []).append("prompt_retrieved_memory")
        if path in recent:
            roles.setdefault(path, []).append("recent_window_memory")
    return roles


def prompt_retrieval_roles(
    all_memory: List[str],
    memory_bank: List[str],
    selected_hits: List,
    max_memory_size: int,
    fix: int,
    retrieval_top_k: int,
) -> Dict[str, List[str]]:
    if len(all_memory) <= max_memory_size:
        return default_memory_roles(all_memory, max_memory_size, fix)

    fixed_budget = max(0, min(fix, max_memory_size))
    retrieved_budget = max(0, min(retrieval_top_k, max_memory_size - fixed_budget))
    recent_budget = max(0, max_memory_size - fixed_budget - retrieved_budget)
    role_by_path: Dict[str, List[str]] = {}
    for path in all_memory[:fixed_budget]:
        role_by_path.setdefault(path, []).append("early_sink_memory")
    for hit in selected_hits[:retrieved_budget]:
        role_by_path.setdefault(hit.path, []).append("prompt_retrieved_memory")
    if recent_budget > 0:
        for path in all_memory[-recent_budget:]:
            role_by_path.setdefault(path, []).append("recent_window_memory")
    for path in memory_bank:
        role_by_path.setdefault(path, []).append("default_memory")
    return role_by_path


def run_story(config: RunConfig) -> Optional[str]:
    init_logging(config.output_dir)
    store = ArtifactStore(config.output_dir)
    story_script = json5.load(open(config.story_script_path, "r", encoding="utf-8"))
    store.init_manifest(config, story_script.get("story_name", "unknown"))
    store.update_manifest("running")

    client = SeedanceClient(api_base=config.seedance_api_base, model=config.seedance_model)
    memory_query_generator = (
        MemoryQueryGenerator(
            cache_path=os.path.join(config.output_dir, "llm_memory_queries.jsonl"),
            base_url=config.llm_memory_query_base_url,
            model=config.llm_memory_query_model,
        )
        if config.llm_memory_query
        else None
    )
    memory_report_records: List[dict] = []
    if config.resume:
        completed_by_shot: Dict[Tuple[int, int], dict] = {}
        for existing in store.load_jsonl(store.shot_log_path):
            if existing.get("status") not in {"downloaded", "skipped_existing"}:
                continue
            output_video = str(existing.get("output_video", ""))
            if output_video and os.path.exists(output_video):
                completed_by_shot[(int(existing["scene_num"]), int(existing["shot_num"]))] = {
                    "scene_num": int(existing["scene_num"]),
                    "shot_num": int(existing["shot_num"]),
                    "prompt": str(existing.get("prompt", "")),
                    "is_cut": bool(existing.get("is_cut", True)),
                    "output_video": output_video,
                    "references": list(existing.get("references", [])),
            }
        memory_report_records = [completed_by_shot[key] for key in sorted(completed_by_shot)]
    shots = iter_shots(story_script)
    visual_memory = VisualElementMemory(config, story_script, shots) if config.visual_element_memory else None
    completed_clips: List[Dict[str, Any]] = []

    for shot_no, shot in enumerate(shots, start=1):
        if shot_no > config.max_shots:
            break
        store.update_manifest("running", current_shot=shot)
        out_video = store.output_video_path(shot)
        if config.resume and os.path.exists(out_video):
            logging.info("Resume: skipping existing %s", out_video)
            if not config.skip_keyframes and not store.has_keyframes(shot):
                store.update_manifest("extracting_keyframes", current_shot=shot)
                save_keyframes(
                    out_video,
                    keyframe_profile=config.keyframe_profile,
                    keyframe_config_path=config.keyframe_config_path,
                )
            store.append_shot_record(
                ShotRunRecord(
                    scene_num=shot.scene_num,
                    shot_num=shot.shot_num,
                    prompt=shot.prompt,
                    is_cut=shot.is_cut,
                    duration_seconds=_effective_duration(config, shot),
                    output_video=out_video,
                    references=[],
                    status="skipped_existing",
                    generation_mode="default",
                )
            )
            continue

        is_first = shot_no == 1
        previous_source = None
        if shot_no > 1:
            previous = shots[shot_no - 2]
            previous_source = (previous.scene_num, previous.shot_num, previous.prompt)
        task_id = None
        refs_metadata: List[dict] = []
        record: Dict[str, object] = {}
        visual_report_extra: Dict[str, object] = {}
        generation_mode = _generation_mode_for_shot(config, shot_no, shot)
        try:
            record = store.find_resume_task(shot.scene_num, shot.shot_num) if config.resume else None
            if record:
                task_id = str(record["task_id"])
                refs_metadata = record.get("reference_labels", [])
                logging.info("Resume: waiting for existing Seedance task %s for Scene %d / Shot %d", task_id, shot.scene_num, shot.shot_num)
            else:
                prompt_override = None
                if visual_memory is not None:
                    visual_plan = visual_memory.plan_visual_elements_for_shot(shot_no - 1, shot)
                    visual_selection = visual_memory.select_historical_references_for_shot(
                        shot_no - 1,
                        shot,
                        visual_plan,
                    )
                    refs = visual_selection.selected_references
                    prompt_override = visual_selection.prompt_context
                    visual_report_extra = visual_selection.report_record
                else:
                    refs = build_reference_images(
                        config=config,
                        store=store,
                        story_script=story_script,
                        shot=shot,
                        is_first=is_first,
                        memory_query_generator=memory_query_generator,
                        previous_source=previous_source,
                    )
                if generation_mode == "smooth":
                    refs = _prepend_smooth_reference(
                        config=config,
                        shot=shot,
                        previous_shot=shots[shot_no - 2],
                        previous_video=store.output_video_path(shots[shot_no - 2]),
                        refs=refs,
                    )
                    if prompt_override is not None:
                        prompt_override = (
                            f"{SMOOTH_CONTINUATION_INSTRUCTION}\n\n"
                            "Input media numbering for this Smooth shot:\n"
                            "- Video 1 是上一段原始视频的尾部片段。当前 shot 必须向后延长 Video 1，从 Video 1 最后一帧之后的下一时刻自然继续。\n"
                            "- Static reference images are numbered separately as Image 1, Image 2, ... after Video 1. "
                            "When the prompt below says Image 1, it means the first static image reference, not Video 1.\n\n"
                            f"{prompt_override}"
                        )
                if visual_memory is not None:
                    store.append_reference_records(shot, [item.get("metadata", {}) for item in refs])
                prepared = prepare_shot(
                    config=config,
                    story_script=story_script,
                    shot=shot,
                    references=refs,
                    is_first=is_first,
                    generation_mode=generation_mode,
                    prompt_override=prompt_override,
                )
                refs_metadata = prepared.reference_metadata
                input_media_summary = _input_media_summary(prepared.content)
                logging.info(
                    "Creating Seedance task for Scene %d / Shot %d | cut=%s | refs=%d | media=%s",
                    shot.scene_num,
                    shot.shot_num,
                    shot.is_cut,
                    len(refs),
                    input_media_summary,
                )
                store.update_manifest("submitting", current_shot=shot)
                submission = submit_shot(config=config, client=client, prepared=prepared)
                created = submission.response
                task_id = submission.task_id
                record = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "scene_num": shot.scene_num,
                    "shot_num": shot.shot_num,
                    "prompt": shot.prompt,
                    "is_cut": shot.is_cut,
                    "duration_seconds": _effective_duration(config, shot),
                    "reference_count": len(refs),
                    "reference_labels": refs_metadata,
                    "input_media_summary": input_media_summary,
                    "enhanced_text_prompt": config.enhanced_text_prompt,
                    "generation_mode": generation_mode,
                    "submitted_prompt": prepared.submitted_prompt,
                    "task_id": task_id,
                    "create_response": created,
                }
                store.append_task_record(record)
                memory_report_records.append(
                    {
                        "scene_num": shot.scene_num,
                        "shot_num": shot.shot_num,
                        "prompt": shot.prompt,
                        "is_cut": shot.is_cut,
                        "duration_seconds": _effective_duration(config, shot),
                        "output_video": out_video,
                        "references": refs_metadata,
                        "submitted_prompt": prepared.submitted_prompt,
                        "generation_mode": generation_mode,
                        **visual_report_extra,
                    }
                )
                write_memory_report(config.output_dir, story_script, memory_report_records, config)

            logging.info("Waiting for Seedance task %s", task_id)
            store.update_manifest("waiting_for_seedance", current_shot=shot, task_id=task_id)
            completed = complete_shot(
                config=config,
                client=client,
                task_id=task_id,
                output_video=out_video,
                on_status=lambda status: store.update_manifest(
                    status,
                    current_shot=shot,
                    task_id=task_id,
                ),
            )
            result = completed.response
            video_url = completed.video_url
            record.update({"result_response": redact_urls(result), "video_url": "<redacted_url>", "output_video": out_video})
            store.append_task_record(record)
            store.append_shot_record(
                ShotRunRecord(
                    scene_num=shot.scene_num,
                    shot_num=shot.shot_num,
                    prompt=shot.prompt,
                    is_cut=shot.is_cut,
                    duration_seconds=_effective_duration(config, shot),
                    output_video=out_video,
                    references=refs_metadata,
                    status="downloaded",
                    task_id=task_id,
                    generation_mode=generation_mode,
                )
            )
            completed_clips.append(
                _clip_record(
                    config=config,
                    shot=shot,
                    output_video=out_video,
                    task_id=task_id,
                    generation_mode=generation_mode,
                )
            )
            logging.info("Saved %s", out_video)
            if config.incremental_concat:
                store.update_manifest("concatenating", current_shot=shot, task_id=task_id)
                _update_cli_final_video(config, store, completed_clips)
            if not config.skip_keyframes:
                store.update_manifest("extracting_keyframes", current_shot=shot, task_id=task_id)
                keyframe_result = save_keyframes(
                    out_video,
                    keyframe_profile=config.keyframe_profile,
                    keyframe_config_path=config.keyframe_config_path,
                )
                if visual_memory is not None:
                    annotations = visual_memory.annotate_completed_shot(
                        shot_index=shot_no,
                        shot=shot,
                        keyframe_paths=keyframe_result.get("keyframe_paths", []),
                    )
                    if memory_report_records:
                        memory_report_records[-1]["visual_element_current_annotations"] = annotations
                        if visual_memory.records:
                            memory_report_records[-1]["visual_element_current_annotation_visualizations"] = (
                                visual_memory.records[-1].get("current_annotation_visualizations", [])
                            )
                        write_memory_report(config.output_dir, story_script, memory_report_records, config)
        except Exception as exc:
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            latest_task = None
            if task_id:
                try:
                    latest_task = client.get_task(task_id)
                    error["task_response"] = redact_urls(latest_task)
                except Exception as task_exc:
                    error["task_lookup_error"] = f"{type(task_exc).__name__}: {task_exc}"
            logging.exception("Failed Scene %d / Shot %d", shot.scene_num, shot.shot_num)
            store.update_manifest("failed", current_shot=shot, task_id=task_id, error=error)
            failure_record = dict(record or {})
            failure_record.update(
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "scene_num": shot.scene_num,
                    "shot_num": shot.shot_num,
                    "prompt": shot.prompt,
                    "is_cut": shot.is_cut,
                    "reference_count": len(refs_metadata),
                    "reference_labels": refs_metadata,
                    "task_id": task_id,
                    "status": "failed",
                    "error": error,
                }
            )
            if latest_task is not None:
                failure_record["result_response"] = redact_urls(latest_task)
            store.append_task_record(failure_record)
            store.append_shot_record(
                ShotRunRecord(
                    scene_num=shot.scene_num,
                    shot_num=shot.shot_num,
                    prompt=shot.prompt,
                    is_cut=shot.is_cut,
                    duration_seconds=_effective_duration(config, shot),
                    output_video=out_video,
                    references=refs_metadata,
                    status="failed",
                    task_id=task_id,
                    generation_mode=locals().get("generation_mode", "default"),
                ).to_dict()
                | {"error": error}
            )
            raise

    final_video = _update_cli_final_video(config, store, completed_clips)
    if visual_memory is not None:
        visual_memory.finish()
    store.update_manifest("completed", final_video=final_video)
    logging.info("Finished Seedance pipeline.")
    return final_video


def _generation_mode_for_shot(config: RunConfig, shot_no: int, shot: ShotSpec) -> str:
    if config.smooth_non_cut and shot_no > 1 and not shot.is_cut:
        return "smooth"
    return "default"


def _effective_duration(config: RunConfig, shot: ShotSpec) -> int:
    return shot.duration_seconds or config.duration


def _input_media_summary(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for index, item in enumerate(content, start=1):
        if item.get("type") == "text":
            summary.append({"index": index, "type": "text", "chars": len(str(item.get("text", "")))})
            continue
        metadata = item.get("metadata") or {}
        entry = {
            "index": index,
            "type": item.get("type"),
            "role": item.get("role"),
            "file": metadata.get("file"),
            "source_path": metadata.get("source_path"),
            "input_origin": metadata.get("input_origin"),
        }
        if item.get("type") == "video_url":
            entry["url_kind"] = "web_url"
        elif item.get("type") == "image_url":
            entry["url_kind"] = "data_url"
        summary.append({key: value for key, value in entry.items() if value is not None})
    return summary


def _prepend_smooth_reference(
    *,
    config: RunConfig,
    shot: ShotSpec,
    previous_shot: ShotSpec,
    previous_video: str,
    refs: List[dict],
) -> List[dict]:
    from storymem_web.reference_video import (
        configured_reference_video_publisher,
        extract_video_tail,
        require_public_https_url,
    )

    if not os.path.exists(previous_video):
        raise RuntimeError(f"Smooth mode requires previous raw video: {previous_video}")
    tail_dir = Path(config.output_dir) / "smooth_reference_tails"
    tail_path = tail_dir / f"{shot.scene_num:02d}_{shot.shot_num:02d}_previous_tail.mp4"
    extract_video_tail(previous_video, tail_path, seconds=config.smooth_reference_seconds)
    public_url = require_public_https_url(configured_reference_video_publisher().publish(tail_path))
    metadata = {
        "label": "previous_tail_video",
        "roles": ["previous_tail_video"],
        "source_path": str(tail_path),
        "file": tail_path.name,
        "source_scene_num": previous_shot.scene_num,
        "source_shot_num": previous_shot.shot_num,
        "source_prompt": previous_shot.prompt,
        "input_origin": "published_previous_raw_tail",
        "duration_seconds": config.smooth_reference_seconds,
    }
    return [video_item_with_metadata(public_url, metadata), *refs]


def _clip_record(
    *,
    config: RunConfig,
    shot: ShotSpec,
    output_video: str,
    task_id: Optional[str],
    generation_mode: str,
) -> Dict[str, Any]:
    attempt_id = task_id or f"{shot.scene_num:02d}_{shot.shot_num:02d}"
    attempt_dir = Path(config.output_dir) / "smooth_attempts" / f"{shot.scene_num:02d}_{shot.shot_num:02d}_{attempt_id}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    return {
        "attempt_id": attempt_id,
        "output_video": output_video,
        "generation_mode": generation_mode,
        "attempt_dir": str(attempt_dir),
    }


def _update_cli_final_video(
    config: RunConfig,
    store: ArtifactStore,
    completed_clips: List[Dict[str, Any]],
) -> Optional[str]:
    if not completed_clips:
        return None
    output = os.path.join(config.output_dir, f"{os.path.basename(config.output_dir)}.mp4")
    if any(item.get("generation_mode") == "smooth" for item in completed_clips):
        from storymem_web.smooth_transition import SmoothVideoAssembler

        return SmoothVideoAssembler()(completed_clips, output)
    return store.concat_videos()
