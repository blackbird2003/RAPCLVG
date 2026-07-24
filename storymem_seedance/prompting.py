from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


SMOOTH_CONTINUATION_INSTRUCTION = (
    "向后延长 Video 1。Video 1 是上一段原始视频的最后一小段，不是普通风格参考。"
    "请从 Video 1 最后一帧之后的下一时刻开始生成新内容，延续其中的人物动作、物体运动、"
    "镜头运动方向和运动速度，让当前 shot 像是在 Video 1 的基础上自然继续。"
    "不要突然跳到另一个姿态或景别，不要突然改变运镜或人物运动，不要倒回、重复、暂停、"
    "复刻、重播或把 Video 1 的任何部分再次包含在新生成视频中。"
    " Extend Video 1 forward as a temporal prefix: start immediately after its final frame, "
    "preserve and advance character/object/camera motion, and avoid any visual jump."
)


def image_data_url(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_item_with_metadata(path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(metadata)
    metadata.setdefault("source_path", path)
    metadata.setdefault("file", os.path.basename(path))
    return {
        "type": "image_url",
        "image_url": {"url": image_data_url(path)},
        "role": "reference_image",
        "metadata": metadata,
    }


def video_item_with_metadata(url: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "video_url",
        "video_url": {"url": url},
        "role": "reference_video",
        "metadata": dict(metadata),
    }


def escape_prompt_line(value: str) -> str:
    return " ".join(str(value).split())


def shot_label(scene_num: Optional[int], shot_num: Optional[int]) -> str:
    if scene_num is None or shot_num is None:
        return "unknown source shot"
    return f"Scene {scene_num} / Shot {shot_num}"


def role_title(role: str) -> str:
    return {
        "previous_last_frame": "previous ending frame",
        "previous_tail_video": "previous ending video",
        "early_sink_memory": "early sink memory",
        "prompt_retrieved_memory": "prompt-retrieved historical memory",
        "recent_window_memory": "recent window memory",
        "default_memory": "default historical memory",
        "all_memory_within_budget": "historical memory within memory budget",
        "visual_element_memory": "visual-element retrieved memory",
        "visual_sink_memory": "early visual sink anchor",
    }.get(role, role.replace("_", " "))


def story_prompt_outline(story_script: dict) -> str:
    lines = []
    for scene in story_script.get("scenes", []):
        scene_num = int(scene.get("scene_num", len(lines) + 1))
        for shot_num, prompt in enumerate(scene.get("video_prompts", []), start=1):
            lines.append(f"- Scene {scene_num} / Shot {shot_num}: {escape_prompt_line(prompt)}")
    return "\n".join(lines)


def reference_prompt_outline(refs: List[Dict[str, Any]]) -> str:
    if not refs:
        return "No reference images are provided for this segment."
    lines = []
    for idx, item in enumerate(refs, start=1):
        meta = item.get("metadata", {})
        roles = meta.get("roles") or [meta.get("role") or meta.get("label", "reference_image")]
        role_text = ", ".join(role_title(str(role)) for role in roles)
        source = shot_label(meta.get("source_scene_num"), meta.get("source_shot_num"))
        source_prompt = meta.get("source_prompt")
        media_type = "video" if item.get("type") == "video_url" else "image"
        detail = f"- Reference {media_type} {idx}: {role_text}; from {source}; file {meta.get('file', os.path.basename(meta.get('source_path', '')))}."
        if source_prompt:
            detail += f" Source shot prompt: {escape_prompt_line(source_prompt)}"
        if meta.get("score") is not None:
            detail += f" Retrieval score: {float(meta['score']):.4f}."
        lines.append(detail)
    return "\n".join(lines)


def compose_prompt(
    prompt: str,
    is_first: bool,
    is_cut: bool,
    refs: List[Dict[str, Any]],
    story_script: dict,
    scene_num: int,
    shot_num: int,
    enhanced_text_prompt: bool,
    generation_mode: str = "default",
) -> str:
    previous_last_frame_indices = [
        index
        for index, item in enumerate(refs, start=1)
        if "previous_last_frame" in (item.get("metadata", {}).get("roles") or [])
    ]
    previous_last_frame_index = previous_last_frame_indices[0] if previous_last_frame_indices else None
    previous_last_frame_is_final = previous_last_frame_index == len(refs) if previous_last_frame_index else False
    if is_first:
        prefix = "Generate the first story segment from text only."
    elif is_cut:
        prefix = (
            "This is a scene transition. Use the reference images as historical visual memory "
            "for character, object, location, and style consistency; do not treat them as a strict first frame."
        )
    elif generation_mode == "smooth":
        prefix = (
            "向后延长 Video 1。Video 1 是上一段原始视频的尾部片段；当前 shot "
            "必须从 Video 1 最后一帧之后的下一时刻自然继续，延续人物/物体/镜头运动，"
            "不要把 Video 1 当作普通参考图或重新生成一个相似开头。"
        )
    elif previous_last_frame_is_final:
        prefix = (
            "This continues the previous shot. Use the final reference image as the previous ending frame "
            "and first-frame continuity constraint for the new video; use earlier reference images as historical memory."
        )
    elif previous_last_frame_index:
        prefix = (
            f"This continues the previous shot. Use reference image {previous_last_frame_index} as the previous ending frame "
            "for temporal continuity (you MUST ensure that the first frame of the new video is EXACTLY THE SAME as this image), "
            "and use the remaining reference images as historical memory."
        )
    else:
        prefix = (
            "This continues the previous shot. Use the reference images as historical visual memory for continuity, "
            "but no strict previous ending frame is provided."
        )
    constraints = (
        "Keep the same story character design and visual style. "
        "Maintain a stable shot scale through the end of the clip. "
        "Do NOT zoom out, pull the camera back, fade out, dim the image, or add an ending transition at the end of video, as next shot may continue the same scene!!"
        "No subtitles, no text overlays, no watermark."
    )
    if not enhanced_text_prompt:
        result = (
            f"{prefix}\n"
            f"Reference {'media' if generation_mode == 'smooth' else 'image'} count: {len(refs)}.\n"
            f"{constraints}\n"
            f"Story prompt: {prompt}"
        )
        return result

    transition_type = "scene transition / cut" if is_cut else "continuous shot"
    reference_heading = "REFERENCE MEDIA GUIDE" if generation_mode == "smooth" else "REFERENCE IMAGE GUIDE"
    if generation_mode == "smooth":
        reference_intro = (
            "Video 1 是上一段原始视频的尾部片段。请向后延长 Video 1：新内容从 Video 1 "
            "最后一帧之后立即开始，保持并推进其中的动作和镜头运动。不要倒回、重复、暂停、"
            "复刻、重播或把 Video 1 的任何部分再次包含在新生成视频中。静态参考图片在 "
            "Video 1 之后单独编号为 Image 1、Image 2 等，仅用于身份、物体、地点和风格一致性。"
        )
    elif previous_last_frame_is_final:
        reference_intro = (
            "The reference images are ordered exactly as listed below. The final reference image is the previous ending frame "
            "and should constrain the first frame of the new video. Earlier reference images are visual memory references: "
            "early sink memory supports stable identity/style, prompt-retrieved memory supports returning scenes or objects, "
            "and recent window memory supports local continuity.\n"
            "Attention!!: as algorithm is not perfect, not all reference images are relevant to the current shot prompt, some may be from unrelated scenes, some objects in references should not appear in the new video, and sometimes we need to create a new scene rather than use the old ones (depending on the context). Please understand what object in these reference are really needed. Use only the relevant ones to support visual continuity and style."
        )
    else:
        reference_intro = (
            "The reference images are ordered exactly as listed below. Use their stated roles carefully: "
            "early sink memory supports stable identity/style, prompt-retrieved memory supports returning scenes or objects, "
            "and recent window memory supports local continuity.\n"
            "Attention!!: as algorithm is not perfect, not all reference images are relevant to the current shot prompt, some may be from unrelated scenes, some objects in references should not appear in the new video, and sometimes we need to create a new scene rather than use the old ones (depending on the context). Please understand what object in these reference are really needed. Use only the relevant ones to support visual continuity and style."
        )
    result = (
        "You are generating exactly one video segment in a longer story-video sequence.\n\n"
        "CURRENT GENERATION TASK\n"
        f"- Segment to generate: Scene {scene_num} / Shot {shot_num} ({transition_type}).\n"
        f"- Current video prompt: {escape_prompt_line(prompt)}\n"
        f"- Local task guidance: {prefix}\n\n"
        "FULL STORY SHOT PLAN FOR GLOBAL CONTEXT\n"
        "Use this only to understand the full narrative arc, recurring characters, locations, and objects. "
        "Do not skip ahead; generate only the current segment.\n"
        f"{story_prompt_outline(story_script)}\n\n"
        f"{reference_heading}\n"
        f"{reference_intro}\n"
        f"{reference_prompt_outline(refs)}\n\n"
        "VISUAL AND EDITING CONSTRAINTS\n"
        f"{constraints}\n"
        "Prioritize the current video prompt over unrelated details in historical reference images."
    )
    if generation_mode == "smooth":
        return f"{SMOOTH_CONTINUATION_INSTRUCTION}\n\n{result}"
    return result
