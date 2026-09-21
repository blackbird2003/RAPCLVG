from __future__ import annotations

DEFAULT_ALGORITHM_STEP_MAX_ATTEMPTS = 5
DEFAULT_STEP_RETRY_DELAY_SECONDS = 60
RETRYABLE_ALGORITHM_STEPS = {"visual_plan", "reference_selection", "seedance_generation"}
MEDIA_PREFLIGHT_TIMEOUT_SECONDS = 15
FORCE_ANIMATION_PROMPT_PREFIX = "Top Priority Constraint:  Regardless of the script description, all human faces—whether characters or images appearing on screens/photos—must be rendered in a non-photorealistic 2D animated style. Photorealistic faces are strictly prohibited; and all characters must be originally designed, do not directly use any copyrighted or trademarked characters. If generating audio, please ensure all character dialogue is in English, with no other languages mixed in."
STYLE_REFERENCE_GUIDANCE = "This reference image may not contain any visual elements required by the new video. Use it as a visual style reference."
NAIVE_TOP_K_SELECTION_MODE = "naive_top_k"
SINK_RECENT_MEMORY_SELECTION_MODE = "sink_recent_memory"
SINK_RECENT_MEMORY_MAX_SIZE = 10
SINK_RECENT_MEMORY_FIX = 3
SINK_RECENT_KEYFRAME_PROFILE = "sink_recent_original"
