from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ProjectCreateRequest(BaseModel):
    name: str = Field(default="Untitled video project", min_length=1, max_length=160)


class ProjectUpdateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=160)


class ProjectSettingsUpdateRequest(BaseModel):
    pipeline_version: Literal["classic", "visual_element_v1"]
    visual_element_sink_frame_count: int = Field(ge=0, le=9)
    visual_element_max_retrieved_frames: int = Field(ge=0, le=9)


class ShotUpdateRequest(BaseModel):
    video_prompt: Optional[str] = Field(default=None, min_length=1)
    is_cut: Optional[bool] = None
    generation_mode: Optional[Literal["default", "last_frame_only", "smooth"]] = None
    duration_seconds: Optional[int] = Field(default=None, ge=2, le=15)
    memory_sink: Optional[bool] = None
    memory_retrieve: Optional[bool] = None
    memory_recent: Optional[bool] = None
    expected_row_version: Optional[int] = None


class RunRequest(BaseModel):
    mode: str = "all"
    shot_id: Optional[str] = None
