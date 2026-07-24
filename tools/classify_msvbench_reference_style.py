#!/usr/bin/env python3
"""Classify one MSVBench character reference image per story with Seed VLM.

This is a small diagnostic script for quickly separating MSVBench stories whose
sample character reference looks realistic vs. non-realistic/animated.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from storymem_seedance.visual_element_memory import (  # noqa: E402
    DEFAULT_VISUAL_ELEMENT_MODEL,
    _call_ark_chat,
)


DEFAULT_MSV_ROOT = Path("/home/wxh/world_model_projects/MSVBench")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_seedance_key() -> None:
    if os.getenv("SEEDANCE_API_KEY") or os.getenv("ARK_API_KEY"):
        return
    key_path = Path.home() / ".seedance_api_key"
    if key_path.exists():
        os.environ["SEEDANCE_API_KEY"] = key_path.read_text(encoding="utf-8-sig").strip()


def image_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".bmp":
        return "image/bmp"
    return "application/octet-stream"


def data_url(path: Path) -> str:
    return f"data:{image_mime_type(path)};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def first_image_in_dir(path: Path) -> Path | None:
    if not path.exists():
        return None
    images = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)
    return images[0] if images else None


def find_reference_image(msv_root: Path, story_id: str, script_data: dict[str, Any]) -> tuple[str, Path]:
    characters = script_data.get("characters") or []
    characters_root = msv_root / "Dataset" / "Dataset" / "characters" / story_id

    for character in characters:
        name = str(character.get("name") or "").strip()
        if not name:
            continue
        candidate_dir = characters_root / name.replace(" ", "_")
        image = first_image_in_dir(candidate_dir)
        if image:
            return name, image

    fallback_images = sorted(
        item
        for item in characters_root.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )
    if fallback_images:
        return fallback_images[0].parent.name.replace("_", " "), fallback_images[0]
    raise FileNotFoundError(f"No character reference image found for story {story_id} in {characters_root}")


def parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("VLM response JSON is not an object")
    return data


def classify_image(image_path: Path, character_name: str, model: str, timeout: int) -> dict[str, Any]:
    prompt = f"""请判断这张 MSVBench 角色参考图的视觉风格。

角色名称：{character_name}

分类标准：
- realistic：照片感、真人影视感、写实 3D 或接近真实摄影/真实人物外观。
- non_realistic：动画、漫画、插画、绘本、卡通、二次元、明显非写实或高度风格化。
- uncertain：无法可靠判断。

只返回 JSON 对象，不要 Markdown：
{{
  "style": "realistic|non_realistic|uncertain",
  "confidence": 0.0,
  "reason": "一句简短中文理由"
}}
"""
    raw_text, metadata = _call_ark_chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url(image_path)}},
                ],
            }
        ],
        model=model,
        max_tokens=1024,
        timeout_seconds=timeout,
    )
    data = parse_json_object(raw_text)
    style = str(data.get("style") or "").strip().lower()
    if style not in {"realistic", "non_realistic", "uncertain"}:
        style = "uncertain"
    return {
        "style": style,
        "confidence": data.get("confidence"),
        "reason": str(data.get("reason") or "").strip(),
        "raw_response": raw_text,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msv-root", type=Path, default=DEFAULT_MSV_ROOT)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / ".runtime" / "msvbench_reference_style.json")
    parser.add_argument("--model", default=DEFAULT_VISUAL_ELEMENT_MODEL)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    load_seedance_key()

    script_dir = args.msv_root / "Dataset" / "Dataset" / "script"
    script_paths = sorted(script_dir.glob("*.json"))
    if not script_paths:
        raise FileNotFoundError(f"No MSVBench scripts found in {script_dir}")

    results: list[dict[str, Any]] = []
    for script_path in script_paths:
        story_id = script_path.stem
        script_data = json.loads(script_path.read_text(encoding="utf-8"))
        character_name, image_path = find_reference_image(args.msv_root, story_id, script_data)
        print(f"[{story_id}] {character_name}: {image_path}", flush=True)
        try:
            classification = classify_image(image_path, character_name, args.model, args.timeout)
            record = {
                "story_id": story_id,
                "character_name": character_name,
                "image_path": str(image_path),
                **classification,
            }
        except Exception as exc:  # Keep the batch going for manual inspection.
            record = {
                "story_id": story_id,
                "character_name": character_name,
                "image_path": str(image_path),
                "style": "uncertain",
                "confidence": None,
                "reason": f"ERROR: {type(exc).__name__}: {exc}",
            }
        print(f"  -> {record['style']} ({record.get('confidence')}): {record.get('reason')}", flush=True)
        results.append(record)

    summary = {
        "model": args.model,
        "realistic_stories": [item["story_id"] for item in results if item["style"] == "realistic"],
        "non_realistic_stories": [item["story_id"] for item in results if item["style"] == "non_realistic"],
        "uncertain_stories": [item["story_id"] for item in results if item["style"] == "uncertain"],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\nSummary")
    print("realistic:", ", ".join(summary["realistic_stories"]) or "(none)")
    print("non_realistic:", ", ".join(summary["non_realistic_stories"]) or "(none)")
    print("uncertain:", ", ".join(summary["uncertain_stories"]) or "(none)")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
