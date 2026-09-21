from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .language import detect_story_language, output_language_instruction
from .visual_element_memory import (
    DEFAULT_VISUAL_ELEMENT_MODEL,
    VISUAL_ELEMENT_MAX_OUTPUT_TOKENS,
    VisualElementMemoryError,
    _call_ark_chat,
)


ALLOWED_TYPES = {
    "character",
    "object",
    "scene",
}
ALLOWED_STATUSES = {
    "should_reference",
    "should_exclude",
    "optional_or_uncertain",
    "new",
}
ALLOWED_ACTIONS = {"no_change", "modify"}
ALLOWED_FIELDS = {"name", "type", "status", "notes", "reason"}


@dataclass(frozen=True)
class VisualPlanReflectionRequest:
    story_title: str
    shots_until_current: list[dict[str, Any]]
    current_shot_id: str
    current_visual_plan: list[dict[str, Any]]
    project_overview: str = ""


@dataclass(frozen=True)
class VisualPlanReflectionResult:
    summary: str
    rows: list[dict[str, Any]]
    warnings: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def build_visual_plan_reflection_prompt(request: VisualPlanReflectionRequest) -> str:
    """Build the Visual Elements Plan reflection prompt in the script language."""

    rows_json = json.dumps(request.current_visual_plan, ensure_ascii=False, indent=2)
    shot_lines = []
    for item in request.shots_until_current:
        shot_id = str(item.get("shot_id") or "")
        scene_num = item.get("scene_num")
        shot_num = item.get("shot_num")
        prompt = str(item.get("video_prompt") or item.get("prompt") or "").strip()
        cut = item.get("is_cut")
        mode = item.get("generation_mode")
        shot_lines.append(
            f"- Shot {shot_id} (Scene {scene_num}, Shot {shot_num}, cut={cut}, mode={mode}): {prompt}"
        )
    shots_text = "\n".join(shot_lines)
    output_language = detect_story_language(
        [request.project_overview] + [
            item.get("video_prompt") or item.get("prompt") or ""
            for item in request.shots_until_current
        ]
    )

    return f"""You are the Visual Elements Plan reflector for a long-form video generation system.

Purpose of the Visual Elements Plan:
1. Maintain a visual-element registry across shots so characters, locations, key objects, and stable environments remain consistent throughout a long video.
2. For the current shot, assign every existing or newly introduced element a state:
   - should_reference: it should be referenced and kept consistent in the current frame;
   - should_exclude: it should be avoided or prevented from entering through historical reference images;
   - optional_or_uncertain: it may appear but is not central, or the textual evidence is insufficient;
   - new: it is introduced by the current shot and should enter the registry.
3. The Visual Plan affects historical-reference selection, so names, types, notes, and status reasons must be clear, stable, and actionable. {output_language_instruction(output_language)}

Task:
Using the script from the first shot through the current shot, inspect every element in the current Visual Plan and determine whether its Name, Type, Status, Notes, and Status reason are appropriate.

Constraints:
1. element_id is fixed and must never change.
2. introduced_at is fixed and must never change, including when it is empty.
3. Do not add or delete elements. This pass may only propose edits to existing rows.
4. Most elements should be no_change. Modify only when necessary.
5. Do not invent visual details absent from the script. Notes may add stable visual features only when supported by the script or a reasonable visible interpretation of the element name.
6. Name must be short, specific, and stable. Avoid generic labels such as "character", "object", or "scene"; do not put actions, emotions, or camera language in Name.
7. Notes must describe visually reproducible stable features rather than plot interpretation, mental states, or invisible settings.
8. Status must match the current shot's visual needs, not the element's importance to the whole story.
9. Type may only be character, object, or scene. Places, environments, and locations are scene; actions and styles are not types.
10. Status may only be should_reference, should_exclude, optional_or_uncertain, or new.

Story title: {request.story_title}
Story overview: {request.project_overview or "None"}
Current shot ID: {request.current_shot_id}

Script from the first shot through the current shot:
{shots_text}

Current Visual Plan JSON:
{rows_json}

Return exactly one JSON object. Do not output Markdown or explanatory prefixes/suffixes. Schema:
{{
  "summary": "One sentence in the detected script language summarizing this reflection.",
  "rows": [
    {{
      "element_id": "original element_id, exactly unchanged",
      "action": "no_change or modify",
      "change_fields": [],
      "suggested": {{
        "name": "suggested name in the detected script language; retain the original value when unchanged",
        "type": "suggested type; retain the original value when unchanged",
        "status": "suggested status; retain the original value when unchanged",
        "notes": "suggested notes in the detected script language; retain the original value when unchanged",
        "reason": "suggested status reason in the detected script language; retain the original value when unchanged"
      }},
      "rationale": "rationale in the detected script language for no change or the proposed modification"
    }}
  ],
  "warnings": [
    {{
      "type": "possible_duplicate, weak_notes, or status_uncertain",
      "element_ids": ["element-xxxx"],
      "message": "optional warning in the detected script language"
    }}
  ]
}}

Validation requirements:
1. The number of rows must exactly equal the number of input Visual Plan rows.
2. The row order must exactly match the input Visual Plan order.
3. Every input element_id must appear exactly once.
4. When action=no_change, change_fields must be an empty array.
5. When action=modify, change_fields may contain only fields actually modified among name/type/status/notes/reason.
"""


def reflect_visual_plan_with_llm(
    request: VisualPlanReflectionRequest,
    *,
    model: str = DEFAULT_VISUAL_ELEMENT_MODEL,
    max_tokens: int = VISUAL_ELEMENT_MAX_OUTPUT_TOKENS,
    timeout_seconds: int = 180,
) -> VisualPlanReflectionResult:
    prompt = build_visual_plan_reflection_prompt(request)
    raw_response, metadata = _call_ark_chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )
    parsed = parse_visual_plan_reflection(raw_response, request.current_visual_plan)
    return VisualPlanReflectionResult(
        summary=str(parsed.get("summary") or ""),
        rows=list(parsed.get("rows") or []),
        warnings=list(parsed.get("warnings") or []),
        raw_response=raw_response,
        prompt=prompt,
        metadata=metadata,
    )


def parse_visual_plan_reflection(raw_text: str, source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    data = _load_json_object(raw_text)
    if not isinstance(data.get("summary"), str):
        raise VisualElementMemoryError("Reflection output is missing a summary string")
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise VisualElementMemoryError("Reflection output is missing a rows array")
    if len(rows) != len(source_rows):
            raise VisualElementMemoryError(f"Reflection rows count mismatch: {len(rows)} != {len(source_rows)}")

    normalized_rows = []
    seen_ids: set[str] = set()
    for index, (source, row) in enumerate(zip(source_rows, rows), start=1):
        if not isinstance(row, dict):
            raise VisualElementMemoryError(f"Reflection row {index} is not an object")
        source_id = _element_id(source)
        row_id = str(row.get("element_id") or "")
        if row_id != source_id:
            raise VisualElementMemoryError(f"Reflection row {index} element_id mismatch: {row_id} != {source_id}")
        if row_id in seen_ids:
            raise VisualElementMemoryError(f"Reflection element_id is duplicated: {row_id}")
        seen_ids.add(row_id)

        action = str(row.get("action") or "")
        if action not in ALLOWED_ACTIONS:
            raise VisualElementMemoryError(f"Reflection row {index} has an invalid action: {action}")
        change_fields = row.get("change_fields")
        if not isinstance(change_fields, list) or any(str(item) not in ALLOWED_FIELDS for item in change_fields):
            raise VisualElementMemoryError(f"Reflection row {index} has invalid change_fields")
        change_fields = [str(item) for item in change_fields]
        if action == "no_change" and change_fields:
            raise VisualElementMemoryError(f"Reflection row {index} with action no_change must not contain change_fields")
        if action == "modify" and not change_fields:
            raise VisualElementMemoryError(f"Reflection row {index} with action modify must contain change_fields")

        suggested = row.get("suggested")
        if not isinstance(suggested, dict):
            raise VisualElementMemoryError(f"Reflection row {index} is missing the suggested object")
        normalized = {
            "element_id": row_id,
            "action": action,
            "change_fields": change_fields,
            "suggested": {
                "name": _required_str(suggested, "name", index),
                "type": _validate_type(_required_str(suggested, "type", index), index),
                "status": _validate_status(_required_str(suggested, "status", index), index),
                "notes": str(suggested.get("notes") or ""),
                "reason": str(suggested.get("reason") or ""),
            },
            "rationale": str(row.get("rationale") or ""),
        }
        normalized_rows.append(normalized)

    warnings = data.get("warnings") or []
    if not isinstance(warnings, list):
        raise VisualElementMemoryError("Reflection warnings must be an array")
    return {"summary": data["summary"], "rows": normalized_rows, "warnings": warnings}


def apply_visual_plan_reflection(source_rows: list[dict[str, Any]], reflection_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a revised Visual Plan while preserving element_id and introduced_at."""

    revised = []
    for source, reflection in zip(source_rows, reflection_rows):
        item = dict(source)
        suggested = reflection.get("suggested") if isinstance(reflection, dict) else {}
        if reflection.get("action") == "modify" and isinstance(suggested, dict):
            item["name"] = str(suggested.get("name") or item.get("name") or "")
            item["type"] = str(suggested.get("type") or item.get("type") or "")
            item["status"] = str(suggested.get("status") or item.get("status") or "")
            item["notes"] = str(suggested.get("notes") if suggested.get("notes") is not None else item.get("notes") or "")
            item["reason"] = str(suggested.get("reason") if suggested.get("reason") is not None else item.get("reason") or "")
        revised.append(item)
    return revised


def request_from_project(project_dir: str | Path, shot_id: str, attempt_id: str | None = None) -> VisualPlanReflectionRequest:
    project_dir = Path(project_dir)
    project = _read_json(project_dir / "project.json")
    story = _read_json(project_dir / "story.json")
    shots = []
    current_order = None
    for shot_file in sorted((project_dir / "shots").glob("*/shot.json")):
        shot = _read_json(shot_file)
        if str(shot.get("shot_id")) == shot_id:
            current_order = int(shot.get("order_index", 0))
        shots.append(shot)
    if current_order is None:
        raise VisualElementMemoryError(f"Shot does not exist in the project: {shot_id}")
    current_shot = shots[current_order]
    attempt = attempt_id or str((current_shot.get("state") or {}).get("current_attempt_id") or "a001")
    rows_path = project_dir / "shots" / shot_id / "attempts" / attempt / "visual_plan" / "visual_element_status.json"
    rows = _read_json(rows_path)
    if not isinstance(rows, list):
        raise VisualElementMemoryError(f"Visual Plan is not an array: {rows_path}")
    return VisualPlanReflectionRequest(
        story_title=str(project.get("name") or story.get("title") or ""),
        project_overview=str(story.get("overview") or ""),
        current_shot_id=shot_id,
        shots_until_current=[
            {
                "shot_id": item.get("shot_id"),
                "scene_num": item.get("scene_num"),
                "shot_num": item.get("shot_num"),
                "video_prompt": (item.get("inputs") or {}).get("video_prompt", item.get("video_prompt", "")),
                "is_cut": (item.get("inputs") or {}).get("is_cut", item.get("is_cut")),
                "generation_mode": (item.get("inputs") or {}).get("generation_mode", item.get("generation_mode")),
            }
            for item in shots[: current_order + 1]
        ],
        current_visual_plan=rows,
    )


def _load_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise VisualElementMemoryError("Reflection output must be a JSON object")
    return data


def _element_id(row: dict[str, Any]) -> str:
    element_id = str(row.get("element_id") or row.get("id") or "")
    if not element_id:
        raise VisualElementMemoryError("Input Visual Plan contains a row with a missing element_id")
    return element_id


def _required_str(data: dict[str, Any], key: str, row_index: int) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise VisualElementMemoryError(f"Reflection row {row_index} suggested.{key} must be a string")
    return value


def _validate_type(value: str, row_index: int) -> str:
    if value not in ALLOWED_TYPES:
        raise VisualElementMemoryError(f"Reflection row {row_index} has an invalid type: {value}")
    return value


def _validate_status(value: str, row_index: int) -> str:
    if value not in ALLOWED_STATUSES:
        raise VisualElementMemoryError(f"Reflection row {row_index} has an invalid status: {value}")
    return value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_key_from_default_file() -> None:
    if os.getenv("SEEDANCE_API_KEY") or os.getenv("ARK_API_KEY"):
        return
    for path in (Path.home() / ".seedance_api_key", Path.home() / ".ark_api_key"):
        if path.is_file():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                os.environ["SEEDANCE_API_KEY"] = value
                return


def _main() -> None:
    parser = argparse.ArgumentParser(description="Prototype Visual Plan Reflection call.")
    parser.add_argument("project_dir", help="Path to a project directory")
    parser.add_argument("--shot-id", required=True)
    parser.add_argument("--attempt-id")
    parser.add_argument("--output", help="Optional path for the reflection JSON")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    _load_key_from_default_file()
    request = request_from_project(args.project_dir, args.shot_id, args.attempt_id)
    result = reflect_visual_plan_with_llm(request, timeout_seconds=args.timeout)
    payload = {
        "summary": result.summary,
        "rows": result.rows,
        "warnings": result.warnings,
        "metadata": result.metadata,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
