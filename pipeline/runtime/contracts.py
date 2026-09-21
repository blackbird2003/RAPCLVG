from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Protocol


@dataclass(frozen=True)
class ShotExecutionContext:
    project_id: str
    bundle: Dict[str, Any]
    shot: Dict[str, Any]
    previous_shots: List[Dict[str, Any]]
    attempt_id: str
    attempt_dir: Path
    input_snapshot: Dict[str, Any]
    started_at: str
    should_cancel: Callable[[], bool] = lambda: False


@dataclass(frozen=True)
class ShotExecutionResult:
    visual_element_status: List[Dict[str, Any]]
    selected_references: List[Dict[str, Any]]
    produced_visual_memory: List[Dict[str, Any]]
    submitted_prompt: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    raw_video_asset_id: str
    assets: Dict[str, Dict[str, Any]]
    task_id: str
    usage: Dict[str, Any] = field(default_factory=dict)
    postprocess: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class VideoPostprocessContext:
    project_id: str
    bundle: Dict[str, Any]
    shot: Dict[str, Any]
    attempt_id: str
    attempt_dir: Path
    video_asset_id: str
    video_path: Path
    visual_element_status: List[Dict[str, Any]]
    dry_run: bool


@dataclass(frozen=True)
class VideoPostprocessResult:
    assets: Dict[str, Dict[str, Any]]
    produced_visual_memory: List[Dict[str, Any]]
    postprocess: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class VisualMemoryPlanContext:
    project_id: str
    bundle: Dict[str, Any]
    shot: Dict[str, Any]
    previous_shots: List[Dict[str, Any]]
    attempt_id: str
    attempt_dir: Path
    input_snapshot: Dict[str, Any]


@dataclass(frozen=True)
class VisualMemoryPlanResult:
    visual_element_status: List[Dict[str, Any]]
    selected_references: List[Dict[str, Any]] = field(default_factory=list)
    prompt_context: str | None = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceSelectionContext:
    project_id: str
    bundle: Dict[str, Any]
    shot: Dict[str, Any]
    previous_shots: List[Dict[str, Any]]
    attempt_id: str
    attempt_dir: Path
    input_snapshot: Dict[str, Any]
    visual_element_status: List[Dict[str, Any]]
    visual_element_details: Dict[str, Any]


@dataclass(frozen=True)
class ReferenceSelectionResult:
    selected_references: List[Dict[str, Any]]
    prompt_context: str | None = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SeedancePromptCompositionResult:
    submitted_prompt: str
    request_content_summary: List[Dict[str, Any]]
    details: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SeedanceGenerationContext:
    project_id: str
    bundle: Dict[str, Any]
    shot: Dict[str, Any]
    previous_shots: List[Dict[str, Any]]
    attempt_id: str
    attempt_dir: Path
    input_snapshot: Dict[str, Any]
    visual_element_status: List[Dict[str, Any]]
    selected_references: List[Dict[str, Any]]
    prompt_context: str | None
    should_cancel: Callable[[], bool] = lambda: False


@dataclass(frozen=True)
class SeedanceGenerationResult:
    submitted_prompt: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    raw_video_asset_id: str
    assets: Dict[str, Dict[str, Any]]
    task_id: str
    usage: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)


class RunnerBackend(Protocol):
    name: str

    def execute(self, context: ShotExecutionContext) -> ShotExecutionResult:
        """Run one shot and return persisted-attempt payloads."""


class BackendExecutionError(RuntimeError):
    pass


class RunnerCancelledError(RuntimeError):
    pass


class SeedanceLikeClient(Protocol):
    def create_task(self, content, **kwargs):
        ...

    def wait_task(self, task_id, **kwargs):
        ...

    def download_video(self, url, output_path):
        ...


class VideoPostprocessor(Protocol):
    name: str

    def process(self, context: VideoPostprocessContext) -> VideoPostprocessResult:
        ...


class VisualMemoryPlanner(Protocol):
    name: str

    def plan(self, context: VisualMemoryPlanContext) -> VisualMemoryPlanResult:
        ...

    def select_references(self, context: ReferenceSelectionContext) -> ReferenceSelectionResult:
        ...
