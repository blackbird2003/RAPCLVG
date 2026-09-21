from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .seedance_client import SeedanceClient, extract_task_id, extract_video_url

from ..models import RunConfig, ShotSpec
from ..prompting import compose_prompt


@dataclass
class PreparedShot:
    shot: ShotSpec
    references: List[Dict[str, Any]]
    reference_metadata: List[Dict[str, Any]]
    submitted_prompt: str

    @property
    def content(self) -> List[Dict[str, Any]]:
        return [{"type": "text", "text": self.submitted_prompt}] + self.references


@dataclass
class TaskSubmission:
    task_id: str
    response: Dict[str, Any]


@dataclass
class CompletedTask:
    task_id: str
    response: Dict[str, Any]
    video_url: str
    output_video: str


def prepare_shot(
    *,
    config: RunConfig,
    story_script: Dict[str, Any],
    shot: ShotSpec,
    references: List[Dict[str, Any]],
    is_first: bool,
    generation_mode: str = "default",
    prompt_override: Optional[str] = None,
) -> PreparedShot:
    submitted_prompt = (
        prompt_override
        if prompt_override is not None
        else compose_prompt(
            prompt=shot.prompt,
            is_first=is_first,
            is_cut=shot.is_cut,
            refs=references,
            story_script=story_script,
            scene_num=shot.scene_num,
            shot_num=shot.shot_num,
            enhanced_text_prompt=config.enhanced_text_prompt,
            generation_mode=generation_mode,
        )
    )
    return PreparedShot(
        shot=shot,
        references=references,
        reference_metadata=[item.get("metadata", {}) for item in references],
        submitted_prompt=submitted_prompt,
    )


def submit_shot(
    *,
    config: RunConfig,
    client: SeedanceClient,
    prepared: PreparedShot,
) -> TaskSubmission:
    created = client.create_task(
        content=prepared.content,
        duration=prepared.shot.duration_seconds if prepared.shot.duration_seconds is not None else config.duration,
        ratio=config.ratio,
        resolution=config.resolution,
        generate_audio=config.generate_audio,
        watermark=config.watermark,
        return_last_frame=config.return_last_frame,
        execution_expires_after=config.execution_expires_after,
        callback_url=config.callback_url,
    )
    return TaskSubmission(task_id=extract_task_id(created), response=created)


def complete_shot(
    *,
    config: RunConfig,
    client: SeedanceClient,
    task_id: str,
    output_video: str,
    should_cancel: Optional[Callable[[], bool]] = None,
    on_status: Optional[Callable[[str], None]] = None,
) -> CompletedTask:
    if on_status:
        on_status("waiting_for_seedance")
    result = client.wait_task(
        task_id,
        poll_interval=config.poll_interval,
        max_wait_seconds=config.max_wait_seconds,
        should_cancel=should_cancel,
    )
    video_url = extract_video_url(result)
    if on_status:
        on_status("downloading")
    client.download_video(video_url, output_video)
    return CompletedTask(
        task_id=task_id,
        response=result,
        video_url=video_url,
        output_video=output_video,
    )


def extract_shot_memory(
    video_path: str,
    *,
    existing_memory_paths: Optional[List[str]] = None,
    keyframe_profile: Optional[str] = None,
    keyframe_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    from ..keyframes.extract import save_keyframes

    return save_keyframes(
        video_path,
        memory_paths=existing_memory_paths,
        keyframe_profile=keyframe_profile,
        keyframe_config_path=keyframe_config_path,
    )
