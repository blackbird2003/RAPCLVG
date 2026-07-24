from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    """Build the Chinese prompt for Visual Elements Plan reflection."""

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

    return f"""你是一个长视频生成系统中的 Visual Elements Plan 反思器。

我们的 Visual Elements Plan 目的：
1. 维护跨镜头视觉元素集合，使角色、地点、关键物品和稳定环境能在长视频中保持一致。
2. 对当前 Shot 判断每个已有或新增元素的状态：
   - should_reference：当前画面应参考并保持一致的元素。
   - should_exclude：当前画面应避免出现或避免被历史参考图带入的元素。
   - optional_or_uncertain：可出现但不关键，或文本证据不足。
   - new：当前 Shot 新引入、应进入集合的元素。
3. Visual Plan 的输出会影响后续历史参考帧选择，因此名称、类型、备注和状态理由必须清楚、稳定、可执行。

你的任务：
请从第一个 Shot 到当前 Shot 的剧本出发，逐一检查“当前 Visual Plan”中的每一个元素，判断它的 Name、Type、Status、Notes、Status reason 是否合理。

重要约束：
1. element_id 是固定信息，绝对不能修改。
2. introduced_at 是固定信息，绝对不能修改；即使为空也保持为空。
3. 不要新增或删除元素；本轮只对已有行提出修改建议。
4. 多数元素应为 no_change。只有确有必要时才修改。
5. 不要编造剧本没有给出的具体外观细节。Notes 可以补充“稳定视觉特征”，但必须来自剧本文本或元素名称的合理可见信息。
6. Name 应短、具体、稳定，避免“人物/物品/场景”等泛称；不要把动作、情绪或镜头语言写入 Name。
7. Notes 应描述可视觉复现的稳定特征，避免剧情解释、心理状态和不可见设定。
8. Status 必须和当前 Shot 的画面需求一致，而不是和整个故事的重要性一致。
9. Type 只可使用：character, object, scene。地点、环境、场所统一归入 scene；动作、风格不要作为 Type。
10. Status 只可使用：should_reference, should_exclude, optional_or_uncertain, new。

故事标题：{request.story_title}
故事概览：{request.project_overview or "无"}
当前 Shot ID：{request.current_shot_id}

从第一个 Shot 到当前 Shot 的剧本：
{shots_text}

当前 Visual Plan JSON：
{rows_json}

请严格输出一个 JSON 对象，不要输出 Markdown，不要输出解释性前后缀。格式如下：
{{
  "summary": "一句话概括本次反思结果",
  "rows": [
    {{
      "element_id": "原 element_id，必须完全一致",
      "action": "no_change 或 modify",
      "change_fields": [],
      "suggested": {{
        "name": "建议后的 name；无修改也填写原值",
        "type": "建议后的 type；无修改也填写原值",
        "status": "建议后的 status；无修改也填写原值",
        "notes": "建议后的 notes；无修改也填写原值",
        "reason": "建议后的 status reason；无修改也填写原值"
      }},
      "rationale": "为什么无需修改或为什么建议修改"
    }}
  ],
  "warnings": [
    {{
      "type": "possible_duplicate 或 weak_notes 或 status_uncertain",
      "element_ids": ["element-xxxx"],
      "message": "可选警告"
    }}
  ]
}}

校验要求：
1. rows 数量必须和输入 Visual Plan 行数完全一致。
2. rows 顺序必须和输入 Visual Plan 顺序一致。
3. 每个输入 element_id 必须出现且只出现一次。
4. action=no_change 时 change_fields 必须为空数组。
5. action=modify 时 change_fields 只能包含 name/type/status/notes/reason 中实际建议修改的字段。
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
        raise VisualElementMemoryError("Reflection 输出缺少 summary 字符串")
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise VisualElementMemoryError("Reflection 输出缺少 rows 数组")
    if len(rows) != len(source_rows):
        raise VisualElementMemoryError(f"Reflection rows 数量不匹配: {len(rows)} != {len(source_rows)}")

    normalized_rows = []
    seen_ids: set[str] = set()
    for index, (source, row) in enumerate(zip(source_rows, rows), start=1):
        if not isinstance(row, dict):
            raise VisualElementMemoryError(f"Reflection row {index} 不是对象")
        source_id = _element_id(source)
        row_id = str(row.get("element_id") or "")
        if row_id != source_id:
            raise VisualElementMemoryError(f"Reflection row {index} element_id 不匹配: {row_id} != {source_id}")
        if row_id in seen_ids:
            raise VisualElementMemoryError(f"Reflection element_id 重复: {row_id}")
        seen_ids.add(row_id)

        action = str(row.get("action") or "")
        if action not in ALLOWED_ACTIONS:
            raise VisualElementMemoryError(f"Reflection row {index} action 非法: {action}")
        change_fields = row.get("change_fields")
        if not isinstance(change_fields, list) or any(str(item) not in ALLOWED_FIELDS for item in change_fields):
            raise VisualElementMemoryError(f"Reflection row {index} change_fields 非法")
        change_fields = [str(item) for item in change_fields]
        if action == "no_change" and change_fields:
            raise VisualElementMemoryError(f"Reflection row {index} no_change 不应包含 change_fields")
        if action == "modify" and not change_fields:
            raise VisualElementMemoryError(f"Reflection row {index} modify 必须包含 change_fields")

        suggested = row.get("suggested")
        if not isinstance(suggested, dict):
            raise VisualElementMemoryError(f"Reflection row {index} 缺少 suggested 对象")
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
        raise VisualElementMemoryError("Reflection warnings 必须是数组")
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
        raise VisualElementMemoryError(f"项目中不存在 shot: {shot_id}")
    current_shot = shots[current_order]
    attempt = attempt_id or str((current_shot.get("state") or {}).get("current_attempt_id") or "a001")
    rows_path = project_dir / "shots" / shot_id / "attempts" / attempt / "visual_plan" / "visual_element_status.json"
    rows = _read_json(rows_path)
    if not isinstance(rows, list):
        raise VisualElementMemoryError(f"Visual Plan 不是数组: {rows_path}")
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
        raise VisualElementMemoryError("Reflection 输出必须是 JSON 对象")
    return data


def _element_id(row: dict[str, Any]) -> str:
    element_id = str(row.get("element_id") or row.get("id") or "")
    if not element_id:
        raise VisualElementMemoryError("输入 Visual Plan 存在缺失 element_id 的行")
    return element_id


def _required_str(data: dict[str, Any], key: str, row_index: int) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise VisualElementMemoryError(f"Reflection row {row_index} suggested.{key} 必须是字符串")
    return value


def _validate_type(value: str, row_index: int) -> str:
    if value not in ALLOWED_TYPES:
        raise VisualElementMemoryError(f"Reflection row {row_index} type 非法: {value}")
    return value


def _validate_status(value: str, row_index: int) -> str:
    if value not in ALLOWED_STATUSES:
        raise VisualElementMemoryError(f"Reflection row {row_index} status 非法: {value}")
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
    parser.add_argument("project_dir", help="Path to a videogen_notebook project directory")
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
