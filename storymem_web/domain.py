from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


class StoryValidationError(ValueError):
    """Raised when an imported story cannot be normalized safely."""


EDITABLE_SHOT_STATES = {"draft", "queued", "stale", "failed", "interrupted"}
ACTIVE_SHOT_STATES = {
    "preparing",
    "planning_visual_elements",
    "selecting_visual_references",
    "submitting",
    "waiting_for_seedance",
    "downloading",
    "extracting_keyframes",
    "annotating_visual_memory",
    "smoothing_transition",
}
TERMINAL_SHOT_STATES = {"completed", "failed", "interrupted"}
SHOT_STATES = EDITABLE_SHOT_STATES | ACTIVE_SHOT_STATES | TERMINAL_SHOT_STATES
GENERATION_MODES = {"default", "last_frame_only", "smooth"}
PIPELINE_VERSIONS = {"classic", "visual_element_v1"}
DEFAULT_PIPELINE_VERSION = "classic"
LEGACY_PIPELINE_VERSION = "classic"
DEFAULT_SHOT_DURATION_SECONDS = 8
MIN_SHOT_DURATION_SECONDS = 2
MAX_SHOT_DURATION_SECONDS = 15
DEFAULT_VISUAL_ELEMENT_SINK_FRAME_COUNT = 0
DEFAULT_VISUAL_ELEMENT_MAX_RETRIEVED_FRAMES = 4
MAX_SEEDANCE_REFERENCE_IMAGES = 9


@dataclass(frozen=True)
class NormalizedShot:
    shot_id: str
    order_index: int
    scene_num: int
    shot_num: int
    video_prompt: str
    is_cut: bool
    duration_seconds: int | None = None
    first_frame_prompt: str = ""
    memory_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedStory:
    story_name: str
    story_overview: str
    shots: List[NormalizedShot]
    source: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "story_name": self.story_name,
            "story_overview": self.story_overview,
            "shots": [shot.to_dict() for shot in self.shots],
            "source": self.source,
        }


DEFAULT_EXAMPLE_PROMPT = (
    "A traveler opens the door of a quiet workshop at sunrise, warm light "
    "revealing half-finished inventions on the tables. Medium-wide shot, "
    "gentle camera movement, calm cinematic atmosphere."
)


def empty_story() -> Dict[str, Any]:
    return {
        "story_name": "Untitled video project",
        "story_overview": "",
        "scenes": [
            {
                "scene_num": 1,
                "video_prompts": [DEFAULT_EXAMPLE_PROMPT],
                "first_frame_prompt": [""],
                "cut": [True],
            }
        ],
    }


def normalize_story(story: Dict[str, Any]) -> NormalizedStory:
    if not isinstance(story, dict):
        raise StoryValidationError("Story must be a JSON object")
    scenes = story.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise StoryValidationError("Story must contain a non-empty scenes list")

    shots: List[NormalizedShot] = []
    seen_scene_numbers = set()
    for scene_position, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            raise StoryValidationError(f"Scene {scene_position} must be an object")
        try:
            scene_num = int(scene.get("scene_num", scene_position))
        except (TypeError, ValueError) as exc:
            raise StoryValidationError(f"Scene {scene_position} has an invalid scene_num") from exc
        if scene_num in seen_scene_numbers:
            raise StoryValidationError(f"Duplicate scene_num: {scene_num}")
        seen_scene_numbers.add(scene_num)

        prompts = scene.get("video_prompts")
        if not isinstance(prompts, list) or not prompts:
            raise StoryValidationError(f"Scene {scene_num} must contain video_prompts")
        cuts = _optional_list(scene, "cut", len(prompts))
        durations = _optional_list(scene, "durations", len(prompts))
        first_frames = _optional_list(scene, "first_frame_prompt", len(prompts))
        memory_queries = _query_list(scene, len(prompts))

        for shot_position, raw_prompt in enumerate(prompts, start=1):
            prompt = str(raw_prompt).strip() if raw_prompt is not None else ""
            if not prompt:
                raise StoryValidationError(
                    f"Scene {scene_num} / Shot {shot_position} has an empty video prompt"
                )
            order_index = len(shots)
            shots.append(
                NormalizedShot(
                    shot_id=f"shot-{order_index + 1:04d}",
                    order_index=order_index,
                    scene_num=scene_num,
                    shot_num=shot_position,
                    video_prompt=prompt,
                    is_cut=True if cuts is None else _as_bool(cuts[shot_position - 1], scene_num, shot_position),
                    duration_seconds=(
                        None
                        if durations is None
                        else validate_shot_duration(durations[shot_position - 1])
                    ),
                    first_frame_prompt=_string_at(first_frames, shot_position - 1),
                    memory_query=_string_at(memory_queries, shot_position - 1),
                )
            )

    return NormalizedStory(
        story_name=str(story.get("story_name") or "Untitled video project").strip(),
        story_overview=str(story.get("story_overview") or "").strip(),
        shots=shots,
        source=story,
    )


def validate_shot_duration(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("duration_seconds must be an integer")
    try:
        duration = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration_seconds must be an integer") from exc
    if not MIN_SHOT_DURATION_SECONDS <= duration <= MAX_SHOT_DURATION_SECONDS:
        raise ValueError(
            f"duration_seconds must be between {MIN_SHOT_DURATION_SECONDS} and "
            f"{MAX_SHOT_DURATION_SECONDS}"
        )
    return duration


def normalize_generation_config(
    config: Dict[str, Any] | None,
    *,
    default_pipeline_version: str = DEFAULT_PIPELINE_VERSION,
) -> Dict[str, Any]:
    normalized = dict(config or {})
    pipeline_version = str(
        normalized.get("pipeline_version") or default_pipeline_version
    ).strip()
    if pipeline_version not in PIPELINE_VERSIONS:
        raise ValueError(f"Unknown pipeline_version: {pipeline_version}")
    normalized["pipeline_version"] = pipeline_version

    sink_count = _non_negative_int(
        normalized.get(
            "visual_element_sink_frame_count",
            DEFAULT_VISUAL_ELEMENT_SINK_FRAME_COUNT,
        ),
        "visual_element_sink_frame_count",
    )
    max_retrieved = _non_negative_int(
        normalized.get(
            "visual_element_max_retrieved_frames",
            DEFAULT_VISUAL_ELEMENT_MAX_RETRIEVED_FRAMES,
        ),
        "visual_element_max_retrieved_frames",
    )
    if sink_count + max_retrieved > MAX_SEEDANCE_REFERENCE_IMAGES:
        raise ValueError(
            "visual element sink frames plus retrieved frames must be <= "
            f"{MAX_SEEDANCE_REFERENCE_IMAGES}"
        )
    normalized["visual_element_sink_frame_count"] = sink_count
    normalized["visual_element_max_retrieved_frames"] = max_retrieved
    return normalized


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer") from exc
    if integer < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return integer


def _optional_list(scene: Dict[str, Any], key: str, expected_length: int) -> List[Any] | None:
    value = scene.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise StoryValidationError(f"Scene {scene.get('scene_num')} field {key} must be a list")
    if len(value) != expected_length:
        raise StoryValidationError(
            f"Scene {scene.get('scene_num')} field {key} has {len(value)} values; "
            f"expected {expected_length}"
        )
    return value


def _query_list(scene: Dict[str, Any], expected_length: int) -> List[Any] | None:
    for key in ("memory_queries", "retrieval_queries", "first_frame_prompt"):
        value = scene.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or len(value) != expected_length:
            raise StoryValidationError(
                f"Scene {scene.get('scene_num')} field {key} must contain {expected_length} values"
            )
        return value
    return None


def _string_at(values: List[Any] | None, index: int) -> str:
    if values is None or values[index] is None:
        return ""
    return str(values[index]).strip()


def _as_bool(value: Any, scene_num: int, shot_num: int) -> bool:
    if isinstance(value, bool):
        return value
    raise StoryValidationError(
        f"Scene {scene_num} / Shot {shot_num} cut value must be true or false"
    )
