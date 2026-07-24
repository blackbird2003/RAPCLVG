from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import RunConfig
from .prompting import role_title, shot_label


def write_memory_report(output_dir: str, story_script: dict, records: List[Dict[str, Any]], config: RunConfig) -> None:
    report_path = os.path.join(output_dir, "memory_report.md")
    sink = []
    seen_sink = set()
    for record in records:
        for ref in record.get("references", []):
            roles = ref.get("roles") or []
            if (
                ("early_sink_memory" in roles or "visual_sink_memory" in roles)
                and ref.get("source_path") not in seen_sink
            ):
                sink.append(ref)
                seen_sink.add(ref.get("source_path"))

    lines = [
        "# Seedance Memory Report",
        "",
        "## Experiment Config",
        "",
        f"- Story: `{_md_escape(story_script.get('story_name', 'unknown'))}`",
        f"- Enhanced text prompt: `{bool(config.enhanced_text_prompt)}`",
        f"- Prompt-aware retrieval: `{bool(config.prompt_retrieval)}`",
        f"- Visual element memory: `{bool(getattr(config, 'visual_element_memory', False))}`",
        f"- Visual element sink frames: `{getattr(config, 'visual_element_sink_frame_count', 3)}`",
        f"- Visual element max retrieved frames: `{getattr(config, 'visual_element_max_retrieved_frames', 4)}`",
        f"- LLM memory query: `{bool(config.llm_memory_query)}`",
        f"- Default duration: `{_duration_label(config.duration)}`",
        f"- Max memory size: `{config.max_memory_size}`",
        f"- Early sink budget (`fix`): `{config.fix}`",
        f"- Retrieval top-k: `{config.retrieval_top_k}`",
        "",
        "## Early Sink Frames Observed",
        "",
    ]
    if sink:
        lines.extend(["| Preview | File | Source shot | Source prompt |", "| --- | --- | --- | --- |"])
        for ref in sink:
            source = shot_label(ref.get("source_scene_num"), ref.get("source_shot_num"))
            file_link = _relative_to_output(ref.get("source_path", ""), output_dir)
            preview = _md_image(file_link)
            lines.append(
                f"| {preview} | `{_md_escape(file_link)}` | {_md_escape(source)} | "
                f"{_md_escape(ref.get('source_prompt', ''))} |"
            )
    else:
        lines.append("_No early sink frames have been used yet._")

    lines.extend(["", "## Generated Segments", ""])
    for record in records:
        lines.extend(
            [
                f"### Scene {record['scene_num']} / Shot {record['shot_num']}",
                "",
                f"- Cut / transition: `{record['is_cut']}`",
                f"- Generation mode: `{_md_escape(record.get('generation_mode', 'default'))}`",
                f"- Generation duration: `{_duration_label(record.get('duration_seconds', config.duration))}`",
                f"- Output video: `{_md_escape(_relative_to_output(record.get('output_video', ''), output_dir))}`",
                "",
                "**Video prompt**",
                "",
                record["prompt"],
                "",
                "**Submitted prompt**",
                "",
                "<details>",
                "<summary>Show full prompt sent to Seedance</summary>",
                "",
                "```text",
                str(record.get("submitted_prompt") or "").rstrip(),
                "```",
                "",
                "</details>",
                "",
                "**Reference images**",
                "",
            ]
        )
        if record.get("visual_element_enabled"):
            plan = record.get("visual_element_plan") or {}
            lines.extend(
                [
                    "**Visual element plan**",
                    "",
                    "- Should reference: "
                    + _element_names(plan.get("should_reference", [])),
                    "- Should exclude: "
                    + _element_names(plan.get("should_exclude", [])),
                    "- Optional or uncertain: "
                    + _element_names(plan.get("optional_or_uncertain", [])),
                    "- Newly introduced in current prompt: "
                    + _element_names(plan.get("new_elements", [])),
                    "",
                ]
            )
        refs = record.get("references", [])
        if not refs:
            lines.append("_No reference images._")
        else:
            lines.extend(
                [
                    "| # | Preview | Role | File | Source shot | Score | Visual element intent | Source prompt |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for ref in refs:
                roles = ", ".join(role_title(role) for role in (ref.get("roles") or [ref.get("label", "reference_image")]))
                score = "" if ref.get("score") is None else f"{float(ref['score']):.4f}"
                file_link = _relative_to_output(ref.get("source_path", ""), output_dir)
                preview_path = ref.get("visualization_path") or ref.get("source_path", "")
                preview = _md_media(_relative_to_output(preview_path, output_dir))
                source = shot_label(ref.get("source_scene_num"), ref.get("source_shot_num"))
                lines.append(
                    f"| {ref.get('reference_index', '')} | {preview} | {_md_escape(roles)} | `{_md_escape(file_link)}` | "
                    f"{_md_escape(source)} | {_md_escape(score)} | {_md_escape(ref.get('reference_intent', ''))} | "
                    f"{_md_escape(ref.get('source_prompt', ''))} |"
                )
        current_visualizations = record.get("visual_element_current_annotation_visualizations") or []
        if current_visualizations:
            lines.extend(["", "**Current shot keyframe annotations**", ""])
            for path in current_visualizations:
                relative = _relative_to_output(path, output_dir)
                lines.append(_md_image(relative))
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _md_escape(value: Optional[Any]) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _duration_label(value: Any) -> str:
    try:
        duration = int(value)
    except (TypeError, ValueError):
        return "auto (-1)"
    if duration == -1:
        return "auto (-1)"
    return f"{duration} seconds"


def _relative_to_output(path: str, output_dir: str) -> str:
    try:
        return os.path.relpath(path, output_dir)
    except ValueError:
        return path


def _md_image(path: str, alt: Optional[str] = None) -> str:
    if not path:
        return ""
    safe_path = path.replace("\\", "/")
    label = alt or os.path.basename(safe_path) or "reference image"
    safe_src = _md_escape(safe_path).replace('"', "&quot;")
    safe_alt = _md_escape(label).replace('"', "&quot;")
    return f'<img src="{safe_src}" alt="{safe_alt}" width="180">'


def _md_media(path: str, alt: Optional[str] = None) -> str:
    if Path(path).suffix.lower() in {".mp4", ".mov", ".webm"}:
        safe_path = path.replace("\\", "/")
        safe_src = _md_escape(safe_path).replace('"', "&quot;")
        return f'<video src="{safe_src}" width="180" controls muted></video>'
    return _md_image(path, alt)


def _element_names(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "`none`"
    return ", ".join(f"`{_md_escape(item.get('name', ''))}`" for item in items)
