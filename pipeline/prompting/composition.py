from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

SMOOTH_CONTINUATION_INSTRUCTION = (
    "Extend Video 1 forward. Video 1 is the final segment of the preceding raw video, not an ordinary style reference. "
    "You must ensure the first frame of the generated video is EXACTLY THE SAME as Video 1's final frame. "
    "Start the new content immediately after Video 1's final frame and continue its character actions, object motion, "
    "camera direction, and camera speed naturally. Do not jump to a different pose or framing, change the camera or "
    "character motion abruptly, rewind, repeat, pause, recreate, replay, or include any part of Video 1 again."
)


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
            "Extend Video 1 forward. Video 1 is the tail of the preceding raw video; the current shot "
            "must continue naturally from the instant after its final frame, advancing character, object, and camera motion. "
            "Do not treat Video 1 as an ordinary reference image or generate a similar opening again."
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
            "Video 1 is the tail segment of the preceding raw video. Extend Video 1 forward: start new content "
            "immediately after its final frame and preserve and advance its action and camera motion. Do not rewind, repeat, "
            "pause, recreate, replay, or include any part of Video 1 again. Static reference images are numbered separately "
            "after Video 1 as Image 1, Image 2, and so on; use them only for identity, object, location, and style consistency."
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
