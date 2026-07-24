from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from keyframe_settings import DEFAULT_PROFILE as DEFAULT_KEYFRAME_PROFILE
from seedance_client import DEFAULT_MODEL as DEFAULT_SEEDANCE_MODEL


SCHEMA_VERSION = 1
PIPELINE_NAME = "visual_element_memory_v1"
DEFAULT_PROJECT_NAME = "Untitled video project"
AUTO_DURATION_SECONDS = -1
DEFAULT_DURATION_SECONDS = AUTO_DURATION_SECONDS
MIN_DURATION_SECONDS = 4
MAX_DURATION_SECONDS = 15
DEFAULT_SINK_FRAME_COUNT = 0
DEFAULT_MAX_RETRIEVED_FRAMES = 4
DEFAULT_SMOOTH_REFERENCE_SECONDS = 2.0
DEFAULT_NON_CUT_GENERATION_MODE = "smooth"
MAX_REFERENCE_IMAGES = 9
VISUAL_SELECTION_MODES = {"greedy_coverage", "static_top_k", "naive_top_k", "storymem_memory", "none"}
GENERATION_MODES = {"default", "last_frame_only", "smooth"}
SEEDANCE_RESOLUTIONS = {"480p", "720p", "1080p"}
SEEDANCE_RATIOS = {"16:9", "9:16", "1:1", "4:3", "3:4"}
DEFAULT_VISUAL_SCORING = {
    "character_weight": 3.0,
    "scene_weight": 2.0,
    "object_weight": 1.5,
    "reference_uncovered_weight": 1.0,
    "reference_covered_weight": 0.2,
    "optional_weight": 0.1,
    "exclude_weight": -0.1,
    "quality_full_weight": 1.0,
    "quality_partial_weight": 0.2,
    "quality_weak_weight": 0.1,
}
STEP_SEQUENCE = (
    "visual_plan",
    "reference_selection",
    "seedance_prompt",
    "seedance_generation",
    "keyframe_maintaining",
)
STEP_LABELS = {
    "visual_plan": "Visual Elements Plan",
    "reference_selection": "Historical Reference Selection",
    "seedance_prompt": "Seedance Prompt Composition",
    "seedance_generation": "Seedance Video Generation",
    "keyframe_maintaining": "Keyframe Maintaining",
}
STEP_STATUSES = {
    "draft",
    *{f"{step}_running" for step in STEP_SEQUENCE},
    *{f"{step}_completed" for step in STEP_SEQUENCE},
    *{f"{step}_failed" for step in STEP_SEQUENCE},
    "keyframe_maintaining_queued",
    "completed",
    "interrupted",
}
SHOT_STATES = STEP_STATUSES

DEFAULT_EXAMPLE_PROMPT = (
    "A traveler opens the door of a quiet workshop at sunrise, warm light "
    "revealing half-finished inventions on the tables. Medium-wide shot, "
    "gentle camera movement, calm cinematic atmosphere."
)


class StoryValidationError(ValueError):
    """Raised when an imported story cannot be normalized."""


@dataclass(frozen=True)
class NormalizedShot:
    shot_id: str
    order_index: int
    scene_num: int
    shot_num: int
    video_prompt: str
    is_cut: bool
    generation_mode: str
    duration_seconds: int
    predefined_references: List[Dict[str, Any]]

    def to_story_dict(self) -> Dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "scene_num": self.scene_num,
            "shot_num": self.shot_num,
            "video_prompt": self.video_prompt,
            "is_cut": self.is_cut,
            "generation_mode": self.generation_mode,
            "duration_seconds": self.duration_seconds,
            "predefined_references": [dict(item) for item in self.predefined_references],
        }


@dataclass(frozen=True)
class NormalizedStory:
    title: str
    overview: str
    shots: List[NormalizedShot]
    source: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "overview": self.overview,
            "shots": [shot.to_story_dict() for shot in self.shots],
            "source": self.source,
        }


def empty_story(name: str = DEFAULT_PROJECT_NAME) -> Dict[str, Any]:
    return {
        "story_name": name,
        "story_overview": "",
        "scenes": [
            {
                "scene_num": 1,
                "video_prompts": [DEFAULT_EXAMPLE_PROMPT],
                "cut": [True],
                "durations": [DEFAULT_DURATION_SECONDS],
            }
        ],
    }


def default_shot_inputs(
    *,
    has_predecessor: bool,
    default_non_cut_mode: str = DEFAULT_NON_CUT_GENERATION_MODE,
) -> Dict[str, Any]:
    generation_mode = (
        validate_generation_mode(default_non_cut_mode, is_cut=False, has_predecessor=True)
        if has_predecessor
        else "default"
    )
    return {
        "video_prompt": DEFAULT_EXAMPLE_PROMPT,
        "is_cut": not has_predecessor,
        "generation_mode": generation_mode,
        "duration_seconds": DEFAULT_DURATION_SECONDS,
        "predefined_references": [],
    }


def normalize_story(
    story: Dict[str, Any],
    *,
    default_duration: int = DEFAULT_DURATION_SECONDS,
    default_non_cut_mode: str = DEFAULT_NON_CUT_GENERATION_MODE,
) -> NormalizedStory:
    if not isinstance(story, dict):
        raise StoryValidationError("Story must be a JSON object")
    scenes = story.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise StoryValidationError("Story must contain a non-empty scenes list")

    default_duration = validate_duration(default_duration)
    default_non_cut_mode = validate_generation_mode(
        default_non_cut_mode,
        is_cut=False,
        has_predecessor=True,
    )
    shots: List[NormalizedShot] = []
    seen_scenes = set()
    for scene_position, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            raise StoryValidationError(f"Scene {scene_position} must be an object")
        try:
            scene_num = int(scene.get("scene_num", scene_position))
        except (TypeError, ValueError) as exc:
            raise StoryValidationError(f"Scene {scene_position} has an invalid scene_num") from exc
        if scene_num in seen_scenes:
            raise StoryValidationError(f"Duplicate scene_num: {scene_num}")
        seen_scenes.add(scene_num)

        prompts = scene.get("video_prompts")
        if not isinstance(prompts, list) or not prompts:
            raise StoryValidationError(f"Scene {scene_num} must contain video_prompts")
        cuts = _optional_list(scene, "cut", len(prompts))
        durations = _optional_list(scene, "durations", len(prompts))
        modes = _optional_list(scene, "generation_modes", len(prompts))
        predefined_references = _optional_list(scene, "predefined_references", len(prompts))

        for shot_position, raw_prompt in enumerate(prompts, start=1):
            prompt = str(raw_prompt).strip() if raw_prompt is not None else ""
            if not prompt:
                raise StoryValidationError(
                    f"Scene {scene_num} / Shot {shot_position} has an empty video prompt"
                )
            is_cut = True if cuts is None else _as_bool(cuts[shot_position - 1], scene_num, shot_position)
            duration = (
                default_duration
                if durations is None
                else validate_duration(durations[shot_position - 1])
            )
            default_mode = "default" if is_cut or not shots else default_non_cut_mode
            generation_mode = (
                default_mode
                if modes is None
                else validate_generation_mode(modes[shot_position - 1], is_cut=is_cut, has_predecessor=bool(shots))
            )
            shots.append(
                NormalizedShot(
                    shot_id=f"{len(shots) + 1:04d}",
                    order_index=len(shots),
                    scene_num=scene_num,
                    shot_num=shot_position,
                    video_prompt=prompt,
                    is_cut=is_cut,
                    generation_mode=generation_mode,
                    duration_seconds=duration,
                    predefined_references=_normalize_predefined_references(
                        [] if predefined_references is None else predefined_references[shot_position - 1],
                        scene_num,
                        shot_position,
                    ),
                )
            )

    return NormalizedStory(
        title=str(story.get("story_name") or story.get("title") or DEFAULT_PROJECT_NAME).strip(),
        overview=str(story.get("story_overview") or story.get("overview") or "").strip(),
        shots=shots,
        source=story,
    )


def default_settings() -> Dict[str, Any]:
    return {
        "generation": {
            "default_duration_seconds": DEFAULT_DURATION_SECONDS,
            "default_cut_mode": "default",
            "default_non_cut_mode": DEFAULT_NON_CUT_GENERATION_MODE,
            "audio": True,
            "auto_run_step_review_delay_seconds": 10,
            "algorithm_step_max_attempts": 5,
            "smooth_reference_seconds": DEFAULT_SMOOTH_REFERENCE_SECONDS,
            "require_human_confirmation_before_seedance": False,
            "auto_reflect_visual_plan": True,
            "force_animation_style": True,
        },
        "visual_element_memory": {
            "sink_frame_count": DEFAULT_SINK_FRAME_COUNT,
            "max_retrieved_frames": DEFAULT_MAX_RETRIEVED_FRAMES,
            "selection_mode": "greedy_coverage",
            "scoring": dict(DEFAULT_VISUAL_SCORING),
        },
        "seedance": {
            "model": DEFAULT_SEEDANCE_MODEL,
            "resolution": "720p",
            "ratio": "16:9",
        },
        "prompt_modules": {
            "full_script_context": True,
            "visual_element_plan": True,
            "holistic_guidance": True,
            "should_reference": True,
            "should_exclude": True,
        },
        "evaluation": {
            "auto_submit_eval": False,
        },
        "keyframes": {"profile": DEFAULT_KEYFRAME_PROFILE},
    }


def validate_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    normalized = default_settings()
    _deep_update(normalized, settings or {})
    generation = normalized["generation"]
    generation["default_duration_seconds"] = validate_duration(
        generation.get("default_duration_seconds", DEFAULT_DURATION_SECONDS)
    )
    generation["default_non_cut_mode"] = validate_generation_mode(
        generation.get("default_non_cut_mode", DEFAULT_NON_CUT_GENERATION_MODE),
        is_cut=False,
        has_predecessor=True,
    )
    generation["audio"] = bool(generation.get("audio", True))
    delay = _non_negative_int(
        generation.get("auto_run_step_review_delay_seconds", 10),
        "auto_run_step_review_delay_seconds",
    )
    generation["auto_run_step_review_delay_seconds"] = delay if delay in {0, 10, 30, 60, 120} else 10
    step_attempts = _non_negative_int(
        generation.get("algorithm_step_max_attempts", 5),
        "algorithm_step_max_attempts",
    )
    generation["algorithm_step_max_attempts"] = min(max(step_attempts, 1), 10)
    smooth_reference_seconds = _float_value(
        generation.get("smooth_reference_seconds", DEFAULT_SMOOTH_REFERENCE_SECONDS),
        "smooth_reference_seconds",
    )
    if smooth_reference_seconds <= 0:
        smooth_reference_seconds = DEFAULT_SMOOTH_REFERENCE_SECONDS
    generation["smooth_reference_seconds"] = min(max(smooth_reference_seconds, 0.5), 10.0)
    generation["require_human_confirmation_before_seedance"] = bool(
        generation.get("require_human_confirmation_before_seedance", False)
    )
    generation["auto_reflect_visual_plan"] = bool(generation.get("auto_reflect_visual_plan", False))
    generation["force_animation_style"] = bool(generation.get("force_animation_style", True))

    memory = normalized["visual_element_memory"]
    sink = _non_negative_int(memory.get("sink_frame_count", DEFAULT_SINK_FRAME_COUNT), "sink_frame_count")
    retrieved = _non_negative_int(
        memory.get("max_retrieved_frames", DEFAULT_MAX_RETRIEVED_FRAMES),
        "max_retrieved_frames",
    )
    if sink + retrieved > MAX_REFERENCE_IMAGES:
        raise ValueError(f"sink_frame_count + max_retrieved_frames must be <= {MAX_REFERENCE_IMAGES}")
    memory["sink_frame_count"] = sink
    memory["max_retrieved_frames"] = retrieved
    selection_mode = str(memory.get("selection_mode") or "greedy_coverage").strip()
    if selection_mode not in VISUAL_SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {sorted(VISUAL_SELECTION_MODES)}")
    memory["selection_mode"] = selection_mode
    scoring = memory.setdefault("scoring", {})
    for key, default in DEFAULT_VISUAL_SCORING.items():
        scoring[key] = _float_value(scoring.get(key, default), f"scoring.{key}")
    seedance = normalized["seedance"]
    resolution = str(seedance.get("resolution") or "720p").strip()
    seedance["resolution"] = resolution if resolution in SEEDANCE_RESOLUTIONS else "720p"
    ratio = str(seedance.get("ratio") or "16:9").strip()
    seedance["ratio"] = ratio if ratio in SEEDANCE_RATIOS else "16:9"
    prompt_modules = normalized.setdefault("prompt_modules", {})
    for key, default in default_settings()["prompt_modules"].items():
        prompt_modules[key] = bool(prompt_modules.get(key, default))
    evaluation = normalized.setdefault("evaluation", {})
    evaluation["auto_submit_eval"] = bool(evaluation.get("auto_submit_eval", False))
    return normalized


def validate_duration(value: Any) -> int:
    if isinstance(value, bool):
        return AUTO_DURATION_SECONDS
    try:
        duration = int(value)
    except (TypeError, ValueError):
        return AUTO_DURATION_SECONDS
    if duration == AUTO_DURATION_SECONDS:
        return duration
    if not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
        return AUTO_DURATION_SECONDS
    return duration


def validate_generation_mode(value: Any, *, is_cut: bool, has_predecessor: bool) -> str:
    mode = str(value or "default").strip()
    if mode not in GENERATION_MODES:
        raise ValueError(f"Unknown generation mode: {mode}")
    if is_cut or not has_predecessor:
        return "default"
    return mode


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


def _normalize_predefined_references(value: Any, scene_num: int, shot_num: int) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise StoryValidationError(
            f"Scene {scene_num} / Shot {shot_num} predefined_references must be a list"
        )
    references: List[Dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise StoryValidationError(
                f"Scene {scene_num} / Shot {shot_num} predefined reference {index} must be an object"
            )
        image_path = str(item.get("image_path") or item.get("path") or "").strip()
        if not image_path:
            raise StoryValidationError(
                f"Scene {scene_num} / Shot {shot_num} predefined reference {index} missing image_path"
            )
        references.append(
            {
                "image_path": image_path,
                "label": str(item.get("label") or "").strip(),
                "guidance": str(item.get("guidance") or item.get("description") or "").strip(),
            }
        )
    return references


def _as_bool(value: Any, scene_num: int, shot_num: int) -> bool:
    if isinstance(value, bool):
        return value
    raise StoryValidationError(
        f"Scene {scene_num} / Shot {shot_num} cut value must be true or false"
    )


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


def _float_value(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
