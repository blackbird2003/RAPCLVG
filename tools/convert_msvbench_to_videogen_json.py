#!/usr/bin/env python3
"""Convert MSVBench scripts into videogen_notebook story JSON files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_MSV_ROOT = Path("/home/wxh/world_model_projects/MSVBench/Dataset/Dataset")
DEFAULT_OUTPUT_ROOT = Path("/home/wxh/world_model_projects/StoryMem/data/msvbench_videogen")
MAX_SEEDANCE_REFERENCE_IMAGES = 9

OUTPUT_MODES = {
    "no_predefined": {"character": False, "shot": False, "filename_suffix": "no_predefined"},
    "character_references": {"character": True, "shot": False, "filename_suffix": "character_reference"},
    "character_and_shot_references": {
        "character": True,
        "shot": True,
        "filename_suffix": "character_and_shot_references",
    },
}

CHARACTER_GUIDANCE_TEMPLATE = (
    "这张图片是角色{character_name}的形象参考图，其中可能包含角色的多张不同姿态的参考图，"
    "请按照剧本要求合理选择参考。角色描述：{character_prompt}"
)
SHOT_GUIDANCE = "这张图片是该Shot的整体参考图,请结合剧本要求合理参考。"


def load_rows(msv_root: Path) -> dict[str, list[dict[str, Any]]]:
    path = msv_root / "MSVBench.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_story: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        story_id = str(row["story_id"]).zfill(2)
        by_story[story_id].append(row)
    for story_rows in by_story.values():
        story_rows.sort(key=lambda item: int(item["shot_id_in_story"]))
    return dict(sorted(by_story.items()))


def read_prompt_lines(msv_root: Path, story_id: str) -> list[str]:
    path = msv_root / "prompt" / f"{story_id}.txt"
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def character_merge_path(msv_root: Path, story_id: str, character_name: str) -> Path:
    character_dir_name = character_name.replace(" ", "_")
    path = msv_root / "characters" / story_id / character_dir_name / "merge.jpg"
    if path.exists():
        return path
    matches = sorted((msv_root / "characters" / story_id).glob("*/merge.jpg"))
    for candidate in matches:
        if candidate.parent.name.replace("_", " ") == character_name:
            return candidate
    raise FileNotFoundError(f"Missing merge.jpg for story {story_id} character {character_name}: {path}")


def shot_reference_path(msv_root: Path, story_id: str, shot_id_in_story: int) -> Path:
    shot_label = f"shot_{shot_id_in_story - 1:02d}"
    candidates = [
        msv_root / "gpt4o" / str(int(story_id)) / f"{shot_label}.png",
        msv_root / "gpt4o" / story_id / f"{shot_label}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing shot reference for story {story_id} {shot_label}: {candidates[0]}")


def character_references(msv_root: Path, row: dict[str, Any]) -> list[dict[str, str]]:
    story_id = str(row["story_id"]).zfill(2)
    names = row.get("shot_characters_appearing_names_en") or []
    prompts = row.get("shot_character_prompts_en") or []
    references: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, name in enumerate(names):
        character_name = str(name).strip()
        if not character_name or character_name in seen:
            continue
        seen.add(character_name)
        character_prompt = str(prompts[index]).strip() if index < len(prompts) else ""
        references.append(
            {
                "image_path": str(character_merge_path(msv_root, story_id, character_name)),
                "label": character_name,
                "guidance": CHARACTER_GUIDANCE_TEMPLATE.format(
                    character_name=character_name,
                    character_prompt=character_prompt,
                ),
            }
        )
    return references


def shot_reference(msv_root: Path, row: dict[str, Any]) -> dict[str, str]:
    story_id = str(row["story_id"]).zfill(2)
    shot_id = int(row["shot_id_in_story"])
    shot_label = f"shot_{shot_id - 1:02d}"
    return {
        "image_path": str(shot_reference_path(msv_root, story_id, shot_id)),
        "label": shot_label,
        "guidance": SHOT_GUIDANCE,
    }


def predefined_references_for_mode(
    msv_root: Path,
    row: dict[str, Any],
    *,
    include_character: bool,
    include_shot: bool,
) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    if include_character:
        references.extend(character_references(msv_root, row))
    if include_shot:
        references.append(shot_reference(msv_root, row))
    if len(references) > MAX_SEEDANCE_REFERENCE_IMAGES:
        story_id = str(row["story_id"]).zfill(2)
        shot_id = int(row["shot_id_in_story"])
        raise ValueError(
            f"Story {story_id} shot {shot_id} has {len(references)} predefined references; "
            f"Seedance static image limit is {MAX_SEEDANCE_REFERENCE_IMAGES}"
        )
    return references


def story_overview(story_id: str, rows: list[dict[str, Any]]) -> str:
    story_type = str(rows[0].get("story_type_en") or "").strip()
    pieces = [str(row.get("shot_plot_correspondence_en") or "").strip() for row in rows]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        return f"MSVBench story {story_id}."
    selected = [pieces[0]]
    if len(pieces) > 2:
        selected.append(pieces[len(pieces) // 2])
    if len(pieces) > 1:
        selected.append(pieces[-1])
    prefix = f"MSVBench story {story_id}"
    if story_type:
        prefix += f" ({story_type})"
    return prefix + ". " + " ".join(selected)


def build_story(
    msv_root: Path,
    story_id: str,
    rows: list[dict[str, Any]],
    prompts: list[str],
    *,
    include_character: bool,
    include_shot: bool,
    story_name: str,
) -> dict[str, Any]:
    if len(rows) != len(prompts):
        raise ValueError(f"Story {story_id}: {len(rows)} metadata rows but {len(prompts)} prompt lines")
    predefined = [
        predefined_references_for_mode(
            msv_root,
            row,
            include_character=include_character,
            include_shot=include_shot,
        )
        for row in rows
    ]
    return {
        "story_name": story_name,
        "story_overview": story_overview(story_id, rows),
        "source_dataset": "MSVBench",
        "source_story_id": story_id,
        "scenes": [
            {
                "scene_num": 1,
                "video_prompts": prompts,
                "cut": [True] * len(prompts),
                "durations": [-1] * len(prompts),
                "predefined_references": predefined,
            }
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def convert(msv_root: Path, output_root: Path) -> dict[str, Any]:
    by_story = load_rows(msv_root)
    summary: dict[str, Any] = {
        "msv_root": str(msv_root),
        "output_root": str(output_root),
        "stories": len(by_story),
        "modes": {},
    }
    prompt_cache = {story_id: read_prompt_lines(msv_root, story_id) for story_id in by_story}
    for mode_name, options in OUTPUT_MODES.items():
        mode_dir = output_root / mode_name
        written: list[str] = []
        max_references = 0
        for story_id, rows in by_story.items():
            file_stem = f"msvbench_{story_id}_{options['filename_suffix']}"
            story = build_story(
                msv_root,
                story_id,
                rows,
                prompt_cache[story_id],
                include_character=options["character"],
                include_shot=options["shot"],
                story_name=file_stem,
            )
            for refs in story["scenes"][0]["predefined_references"]:
                max_references = max(max_references, len(refs))
            out_path = mode_dir / f"{file_stem}.json"
            write_json(out_path, story)
            written.append(str(out_path))
        summary["modes"][mode_name] = {
            "directory": str(mode_dir),
            "json_files": len(written),
            "max_predefined_references_per_shot": max_references,
            "files": written,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msv-root", type=Path, default=DEFAULT_MSV_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    if not args.msv_root.exists():
        raise FileNotFoundError(args.msv_root)
    summary = convert(args.msv_root, args.output_root)
    summary_path = args.summary or args.output_root / "conversion_summary.json"
    write_json(summary_path, summary)

    print(f"stories: {summary['stories']}")
    for mode_name, mode_summary in summary["modes"].items():
        print(
            f"{mode_name}: {mode_summary['json_files']} files, "
            f"max refs/shot={mode_summary['max_predefined_references_per_shot']}, "
            f"dir={mode_summary['directory']}"
        )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
