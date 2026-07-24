from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RunConfig:
    story_script_path: str = "./story/little_prince.json"
    output_dir: str = ""
    max_shots: int = 2
    duration: int = -1
    ratio: str = "16:9"
    resolution: str = field(default_factory=lambda: os.getenv("SEEDANCE_RESOLUTION", "720p"))
    generate_audio: bool = True
    watermark: bool = False
    return_last_frame: bool = True
    execution_expires_after: int = 86400
    callback_url: Optional[str] = field(default_factory=lambda: os.getenv("SEEDANCE_CALLBACK_URL"))
    max_memory_size: int = 8
    fix: int = 3
    prompt_retrieval: bool = False
    retrieval_top_k: int = 2
    retrieval_frame_weight: float = 0.7
    retrieval_video_weight: float = 0.3
    retrieval_min_score: Optional[float] = None
    retrieval_clip_device: str = "cpu"
    llm_memory_query: bool = False
    llm_memory_query_model: str = field(default_factory=lambda: os.getenv("MEMORY_QUERY_LLM_MODEL", "deepseek-chat"))
    llm_memory_query_base_url: str = field(default_factory=lambda: os.getenv("MEMORY_QUERY_LLM_BASE_URL", "https://api.deepseek.com"))
    enhanced_text_prompt: bool = False
    seedance_model: str = field(default_factory=lambda: os.getenv("SEEDANCE_MODEL", "doubao-seedance-2-0-260128"))
    seedance_api_base: str = field(default_factory=lambda: os.getenv("SEEDANCE_API_BASE", "https://ark.cn-beijing.volces.com/api/v3"))
    poll_interval: int = 10
    max_wait_seconds: int = 1800
    max_reference_images: int = 9
    skip_keyframes: bool = False
    keyframe_profile: Optional[str] = None
    keyframe_config_path: Optional[str] = None
    resume: bool = False
    incremental_concat: bool = True
    visual_element_memory: bool = False
    visual_element_model: str = field(default_factory=lambda: os.getenv("VISUAL_ELEMENT_MODEL", "doubao-seed-2-1-turbo-260628"))
    visual_element_timeout: int = 90
    visual_element_max_parse_retries: int = 2
    visual_element_selection_mode: str = "greedy_coverage"
    visual_element_sink_frame_count: int = 3
    visual_element_max_retrieved_frames: int = 4
    visual_element_weight_character: float = 3.0
    visual_element_weight_scene: float = 2.0
    visual_element_weight_object: float = 1.5
    visual_element_weight_reference_uncovered: float = 1.0
    visual_element_weight_reference_covered: float = 0.2
    visual_element_weight_optional: float = 0.1
    visual_element_weight_exclude: float = -0.1
    visual_element_weight_quality_full: float = 1.0
    visual_element_weight_quality_partial: float = 0.2
    visual_element_weight_quality_weak: float = 0.1
    smooth_non_cut: bool = False
    smooth_reference_seconds: float = 2.0

    @classmethod
    def from_namespace(cls, args: Any) -> "RunConfig":
        values = {field_name: getattr(args, field_name) for field_name in cls.__dataclass_fields__ if hasattr(args, field_name)}
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShotSpec:
    scene: Dict[str, Any]
    scene_num: int
    shot_num: int
    prompt: str
    is_cut: bool
    first_frame_prompt: str = ""
    duration_seconds: Optional[int] = None

    @property
    def key(self) -> Tuple[int, int]:
        return (self.scene_num, self.shot_num)


@dataclass
class ReferenceImage:
    source_path: str
    file: str
    label: str = "reference_image"
    roles: List[str] = field(default_factory=list)
    reference_index: Optional[int] = None
    source_scene_num: Optional[int] = None
    source_shot_num: Optional[int] = None
    keyframe_rank: Optional[int] = None
    source_prompt: str = ""
    score: Optional[float] = None
    frame_score: Optional[float] = None
    video_score: Optional[float] = None

    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any]) -> "ReferenceImage":
        roles = metadata.get("roles") or []
        if isinstance(roles, str):
            roles = [roles]
        return cls(
            source_path=str(metadata.get("source_path", "")),
            file=str(metadata.get("file") or os.path.basename(str(metadata.get("source_path", "")))),
            label=str(metadata.get("label", "reference_image")),
            roles=[str(role) for role in roles],
            reference_index=metadata.get("reference_index"),
            source_scene_num=metadata.get("source_scene_num"),
            source_shot_num=metadata.get("source_shot_num"),
            keyframe_rank=metadata.get("keyframe_rank"),
            source_prompt=str(metadata.get("source_prompt", "")),
            score=metadata.get("score"),
            frame_score=metadata.get("frame_score"),
            video_score=metadata.get("video_score"),
        )

    def to_metadata(self) -> Dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None and value != []}


@dataclass
class ShotRunRecord:
    scene_num: int
    shot_num: int
    prompt: str
    is_cut: bool
    duration_seconds: int
    output_video: str
    references: List[Dict[str, Any]]
    status: str = "pending"
    task_id: Optional[str] = None
    generation_mode: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}
