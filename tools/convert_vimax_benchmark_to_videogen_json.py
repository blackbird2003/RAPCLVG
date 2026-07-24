#!/usr/bin/env python3
"""Convert ViMax-Bench JSON scripts into videogen_notebook story JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_VIMAX_ROOT = Path("/home/wxh/world_model_projects/ViMax/vimax_benchmark")
DEFAULT_OUTPUT_ROOT = Path("/home/wxh/world_model_projects/StoryMem/data/vimax_benchmark_videogen")
DEFAULT_DURATION_SECONDS = 8

FIRST_FRAME_PREFIX = "The following is the description of the first frame:"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def script_files(vimax_root: Path) -> list[Path]:
    index_path = vimax_root / "benchmark_index.json"
    if index_path.exists():
        index = read_json(index_path)
        ordered: list[Path] = []
        for item in index.get("stories") or []:
            file_name = item.get("file")
            if file_name:
                path = vimax_root / str(file_name)
                if path.exists():
                    ordered.append(path)
        if ordered:
            return ordered
    return [path for path in sorted(vimax_root.glob("*.json")) if path.name != "benchmark_index.json"]


def merged_prompt(video_prompt: Any, first_frame: Any) -> str:
    prompt = str(video_prompt or "").strip()
    first_frame_text = str(first_frame or "").strip()
    if not prompt:
        raise ValueError("empty video_prompt")
    if not first_frame_text:
        return prompt
    return f"{prompt}\n\n{FIRST_FRAME_PREFIX}\n{first_frame_text}"


def convert_story(path: Path) -> dict[str, Any]:
    data = read_json(path)
    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError(f"{path.name}: missing scenes")

    converted_scenes: list[dict[str, Any]] = []
    total_shots = 0
    for scene_index, scene in enumerate(scenes, start=1):
        raw_shots = scene.get("shots")
        if not isinstance(raw_shots, list) or not raw_shots:
            raise ValueError(f"{path.name}: scene {scene_index} missing shots")
        prompts: list[str] = []
        original_shot_ids: list[Any] = []
        first_frames: list[str] = []
        for shot in raw_shots:
            if not isinstance(shot, dict):
                raise ValueError(f"{path.name}: scene {scene_index} contains non-object shot")
            prompts.append(merged_prompt(shot.get("video_prompt"), shot.get("first_frame")))
            original_shot_ids.append(shot.get("shot_id"))
            first_frames.append(str(shot.get("first_frame") or "").strip())
        total_shots += len(prompts)
        converted_scenes.append(
            {
                "scene_num": int(scene.get("scene_num", scene_index)),
                "video_prompts": prompts,
                "cut": [True] * len(prompts),
                "durations": [DEFAULT_DURATION_SECONDS] * len(prompts),
                "source_shot_ids": original_shot_ids,
                "source_first_frames": first_frames,
            }
        )

    story_name = f"ViMax-Bench_{path.stem}"
    return {
        "story_name": story_name,
        "story_overview": str(data.get("story_overview") or "").strip(),
        "source_dataset": "ViMax-Bench",
        "source_file": path.name,
        "source_consistency_type": data.get("consistency_type"),
        "source_metadata": data.get("metadata") or {},
        "shot_count": total_shots,
        "scenes": converted_scenes,
    }


def convert(vimax_root: Path, output_root: Path) -> dict[str, Any]:
    files = script_files(vimax_root)
    written: list[str] = []
    total_shots = 0
    by_type: dict[str, int] = {}
    for source in files:
        story = convert_story(source)
        out_path = output_root / f"{source.stem}_videogen.json"
        write_json(out_path, story)
        written.append(str(out_path))
        total_shots += int(story["shot_count"])
        type_name = str(story.get("source_consistency_type") or "unknown")
        by_type[type_name] = by_type.get(type_name, 0) + 1
    summary = {
        "source_root": str(vimax_root),
        "output_root": str(output_root),
        "story_files": len(written),
        "total_shots": total_shots,
        "stories_by_type": by_type,
        "files": written,
    }
    write_json(output_root / "conversion_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vimax-root", type=Path, default=DEFAULT_VIMAX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if not args.vimax_root.exists():
        raise FileNotFoundError(args.vimax_root)
    summary = convert(args.vimax_root, args.output_root)
    print(f"stories: {summary['story_files']}")
    print(f"shots: {summary['total_shots']}")
    print(f"by type: {summary['stories_by_type']}")
    print(f"output: {summary['output_root']}")


if __name__ == "__main__":
    main()
