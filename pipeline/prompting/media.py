from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def image_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_item_with_metadata(path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(metadata)
    metadata.setdefault("source_path", path)
    metadata.setdefault("file", os.path.basename(path))
    return {
        "type": "image_url",
        "image_url": {"url": image_data_url(path)},
        "role": "reference_image",
        "metadata": metadata,
    }


def video_item_with_metadata(url: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "video_url",
        "video_url": {"url": url},
        "role": "reference_video",
        "metadata": dict(metadata),
    }


def audio_item_with_metadata(url: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "audio_url",
        "audio_url": {"url": url},
        "role": "reference_audio",
        "metadata": dict(metadata),
    }
