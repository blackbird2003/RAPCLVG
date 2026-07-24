from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict

from storymem_seedance.models import RunConfig


DEFAULT_GENERATION_CONFIG: Dict[str, Any] = {
    "pipeline_version": "classic",
    "duration": 8,
    "ratio": "16:9",
    "resolution": "720p",
    "generate_audio": True,
    "watermark": False,
    "return_last_frame": True,
    "execution_expires_after": 86400,
    # Match the original StoryMem default memory-bank size. Seedance accepts at
    # most 9 images, so the runner trims submitted references after selection.
    "max_memory_size": 10,
    "fix": 3,
    "prompt_retrieval": False,
    "retrieval_top_k": 2,
    "retrieval_frame_weight": 0.7,
    "retrieval_video_weight": 0.3,
    "retrieval_min_score": None,
    "retrieval_clip_device": "cpu",
    "llm_memory_query": False,
    "enhanced_text_prompt": False,
    "poll_interval": 10,
    "max_wait_seconds": 1800,
    "max_reference_images": 9,
    "skip_keyframes": False,
    "keyframe_profile": "storymem_original",
    "keyframe_config_path": None,
    "incremental_concat": True,
    "visual_element_memory": False,
    "visual_element_model": "doubao-seed-2-1-turbo-260628",
    "visual_element_timeout": 90,
    "visual_element_max_parse_retries": 1,
    "visual_element_selection_mode": "greedy_coverage",
    "visual_element_sink_frame_count": 0,
    "visual_element_max_retrieved_frames": 4,
}


def merged_generation_config(value: Dict[str, Any] | None) -> Dict[str, Any]:
    config = dict(DEFAULT_GENERATION_CONFIG)
    config.update(value or {})
    return config


def build_run_config(value: Dict[str, Any] | None, output_dir: str) -> RunConfig:
    allowed = {field.name for field in fields(RunConfig)}
    config = merged_generation_config(value)
    config["output_dir"] = output_dir
    config["max_shots"] = 1
    config["resume"] = False
    return RunConfig(**{key: item for key, item in config.items() if key in allowed})
