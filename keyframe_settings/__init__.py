from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_PROFILE = "loose"
PROFILE_ENV = "STORYMEM_KEYFRAME_PROFILE"
CONFIG_ENV = "STORYMEM_KEYFRAME_CONFIG"
PROFILE_DIR = Path(__file__).resolve().parent

_REQUIRED_KEYS = {
    "image_factor",
    "min_tokens",
    "max_tokens",
    "min_pixels",
    "max_pixels",
    "max_ratio",
    "video_min_tokens",
    "video_max_tokens",
    "video_min_pixels",
    "video_max_pixels",
    "video_total_pixels",
    "min_frame_similarity",
    "max_keyframe_num",
    "adaptive_alpha",
    "hpsv3_quality_threshold",
    "compare_with_history",
}
_INT_KEYS = {
    "image_factor",
    "min_tokens",
    "max_tokens",
    "min_pixels",
    "max_pixels",
    "max_ratio",
    "video_min_tokens",
    "video_max_tokens",
    "video_min_pixels",
    "video_max_pixels",
    "video_total_pixels",
    "max_keyframe_num",
}
_FLOAT_KEYS = {
    "min_frame_similarity",
    "adaptive_alpha",
    "hpsv3_quality_threshold",
}
_BOOL_KEYS = {
    "compare_with_history",
}


def available_profiles() -> List[str]:
    return sorted(path.stem for path in PROFILE_DIR.glob("*.json"))


def resolve_profile_name(profile: Optional[str] = None) -> str:
    return (profile or os.getenv(PROFILE_ENV) or DEFAULT_PROFILE).strip()


def load_keyframe_settings(
    profile: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    explicit_path = (config_path or os.getenv(CONFIG_ENV) or "").strip()
    if explicit_path:
        payload = _load_json(Path(explicit_path))
        return _validate_and_normalize(_merge_with_default(payload), source=explicit_path)
    profile_name = resolve_profile_name(profile)
    path = PROFILE_DIR / f"{profile_name}.json"
    payload = _load_json(path)
    if profile_name == DEFAULT_PROFILE:
        return _validate_and_normalize(payload, source=str(path))
    return _validate_and_normalize(_merge_with_default(payload), source=str(path))


def _merge_with_default(payload: Dict[str, Any]) -> Dict[str, Any]:
    default_path = PROFILE_DIR / f"{DEFAULT_PROFILE}.json"
    merged = _load_json(default_path)
    merged.update(payload)
    return merged


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing keyframe settings file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid keyframe settings JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Keyframe settings must be a JSON object: {path}")
    return data


def _validate_and_normalize(payload: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    missing = sorted(_REQUIRED_KEYS - payload.keys())
    if missing:
        raise ValueError(f"Missing keyframe settings in {source}: {', '.join(missing)}")
    normalized: Dict[str, Any] = {}
    for key in _REQUIRED_KEYS:
        value = payload[key]
        if key in _INT_KEYS:
            normalized[key] = int(value)
        elif key in _FLOAT_KEYS:
            normalized[key] = float(value)
        elif key in _BOOL_KEYS:
            normalized[key] = _as_bool(value)
        else:
            normalized[key] = value
    return normalized


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
