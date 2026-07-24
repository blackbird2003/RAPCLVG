#!/usr/bin/env python3
"""通过剧本 Prompt 与关键帧验证视觉元素集合维护的实验。

本脚本不生成视频。它按 Shot 顺序先用 Seed 2.1 Turbo 维护视觉元素集合，
再只针对当前 Shot 已保存的代表帧调用 VLM 标注集合中已有元素。历史帧不被
重复送入 VLM，而是仅根据已有标注和当前 Shot 的元素状态重新归类。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import requests
from PIL import Image, ImageDraw, ImageFont


API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
DEFAULT_MODEL = "doubao-seed-2-1-turbo-260628"
ELEMENT_TYPES = {"character", "scene", "object"}
ELEMENT_STATES = {"should_reference", "should_exclude", "optional_or_uncertain"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
NORMALIZED_COORDINATE_MAX = 1000
CURRENT_ANNOTATION_COLORS = {
    "character": (220, 70, 70),
    "scene": (150, 70, 210),
    "object": (235, 130, 30),
}
EVALUATION_COLORS = {
    "should_reference": (30, 180, 80),
    "should_exclude": (50, 130, 240),
    "optional_or_uncertain": (240, 190, 35),
}
EVALUATION_LABELS = {
    "should_reference": "应参考",
    "should_exclude": "应排除",
    "optional_or_uncertain": "不确定",
}
CHINESE_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


@dataclass
class VisualElement:
    """项目内稳定存在的视觉元素。"""

    id: str
    name: str
    type: Literal["character", "scene", "object"]
    introduced_at: str
    notes: str = ""


@dataclass
class ElementRegistry:
    """按脚本推进的视觉元素集合。"""

    elements: list[VisualElement] = field(default_factory=list)

    def next_id(self) -> str:
        return f"element-{len(self.elements) + 1:04d}"

    def names(self) -> set[str]:
        return {_normalize_name(element.name) for element in self.elements}


@dataclass
class Shot:
    """从原始剧本展平后的最小 Shot 信息。"""

    id: str
    scene_num: int
    shot_num: int
    video_prompt: str
    is_cut: bool | None


@dataclass
class ExistingElementState:
    """某一个已有元素在当前 Shot 的临时可用状态。"""

    id: str
    state: Literal["should_reference", "should_exclude", "optional_or_uncertain"]
    reason: str


@dataclass
class NewElement:
    """模型建议写入集合的新元素，尚未由模型决定内部 ID。"""

    name: str
    type: Literal["character", "scene", "object"]
    notes: str


@dataclass
class ShotDecision:
    """通过本地校验后的 Shot 决策。"""

    existing_element_states: list[ExistingElementState]
    new_elements: list[NewElement]
    shot_notes: list[str]


@dataclass
class ParseResult:
    """解析失败时保留原因，供总流程决定是否重试模型调用。"""

    decision: ShotDecision | None
    error: str | None = None


@dataclass
class AnnotatedElement:
    """一张代表帧中被 VLM 确认可见的既有元素。"""

    id: str
    name: str
    type: Literal["character", "scene", "object"]
    bbox_1000: list[float]
    bbox_pixels: list[int]


@dataclass
class FrameAnnotation:
    """代表帧的闭集标注记录。"""

    source_shot_id: str
    frame_path: str
    width: int
    height: int
    elements: list[AnnotatedElement]
    attempts: list[dict[str, Any]]


@dataclass
class FrameParseResult:
    """关键帧标注解析结果。"""

    elements: list[AnnotatedElement] | None
    error: str | None = None


class LLMCallError(RuntimeError):
    """Ark 文本模型调用失败。"""


def load_initial_script(script_path: Path) -> list[Shot]:
    """读取剧本，并按照 Scene/Shot 的生成顺序展平。"""
    try:
        payload = json.loads(script_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取剧本 {script_path}: {exc}") from exc

    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("剧本缺少 scenes 列表")

    shots: list[Shot] = []
    global_index = 0
    for scene_index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            raise ValueError(f"scene {scene_index} 不是对象")
        prompts = scene.get("video_prompts")
        if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
            raise ValueError(f"scene {scene_index} 的 video_prompts 不合法")

        cuts = scene.get("cut")
        if cuts is not None and (not isinstance(cuts, list) or len(cuts) != len(prompts)):
            raise ValueError(f"scene {scene_index} 的 cut 与 video_prompts 长度不一致")

        scene_num = int(scene.get("scene_num", scene_index))
        for shot_num, video_prompt in enumerate(prompts, start=1):
            global_index += 1
            is_cut = bool(cuts[shot_num - 1]) if cuts is not None else None
            shots.append(
                Shot(
                    id=f"shot-{global_index:04d}",
                    scene_num=scene_num,
                    shot_num=shot_num,
                    video_prompt=video_prompt.strip(),
                    is_cut=is_cut,
                )
            )
    if not shots:
        raise ValueError("剧本中没有可处理的 Shot")
    return shots


def build_llm_prompt(
    registry: ElementRegistry,
    current_shot: Shot,
    previous_shots: Iterable[Shot],
    retry_feedback: str | None = None,
) -> str:
    """以原始历史 Prompt、元素集合和当前 Prompt 拼装中文决策提示词。"""
    previous_prompt_text = "\n".join(
        f"- {shot.id}: {shot.video_prompt}" for shot in previous_shots
    ) or "（当前是第一个 Shot，没有前序 Prompt。）"
    registry_text = json.dumps(
        [asdict(element) for element in registry.elements],
        ensure_ascii=False,
        indent=2,
    )
    retry_section = ""
    if retry_feedback:
        retry_section = (
            "\n你上一次的回答未通过本地格式或一致性校验。请只修正下述问题，"
            "不要改变任务目标：\n"
            f"{retry_feedback}\n"
        )

    return f"""你是长视频生成项目中的视觉元素规划助手。请只根据文字剧本，维护一个跨 Shot 的视觉元素集合。

视觉元素仅允许三类：
- character：有姿态、表情、语言或动作变化的角色实体；
- scene：通过画面整体判断的宏观背景或地点；
- object：画面中可检测或分割的物体。

工作原则：
1. 元素的名称应该能够稳定指代这个元素（例如，[场景]中[位置]/[明显特征]/[物主]的[物品名]，但注意不要将随时可能变化的特征(例如角色的表情/心理/姿态)写进元素名称中）。相同大类不表示同一元素；例如不同的玫瑰、背包、星球不能因为都属于同类而合并。
2. 只有当前 Prompt 明确引入一个与现有集合不同、且对画面叙事有意义的角色、场景或物体时，才放入 new_elements。不要为泛泛的氛围词、镜头语言、瞬时动作创建元素。
3. 必须给集合中的每个已有元素分配一个状态：
   - should_reference：当前 Shot 需要它作为视觉一致性的参考；
   - should_exclude：当前 Prompt 有明确证据表明它不应出现在画面中；例如，该元素明显与特定场景相关联，而当前 Shot 的场景已发生改变。
   - optional_or_uncertain：可能有帮助，但既非必需也非明确禁止。
4. Prompt 提到/没有提到某元素，不等于 should_reference/should_exclude；请根据上下文判断它是否一定会/一定不/有可能出现在画面中。
5. 姿态、动作、人物关系和镜头变化写入 shot_notes，不要把它们新增为全局元素。
6. 输出必须是一个 JSON 对象，不得使用 Markdown、解释文字或代码块。

现有元素集合：
{registry_text}

此前所有 Shot 的原始 video_prompt（仅作叙事上下文，不要把其中未登记的细节自动新增）：
{previous_prompt_text}

当前待处理 Shot：
- id: {current_shot.id}
- scene: {current_shot.scene_num}
- shot: {current_shot.shot_num}
- cut: {current_shot.is_cut}
- video_prompt: {current_shot.video_prompt}

请严格按下面模式返回。existing_element_states 必须恰好覆盖现有集合中的每一个 id，且每个 id 只能出现一次：
{{
  "existing_element_states": [
    {{"id": "已有元素 id", "state": "should_reference|should_exclude|optional_or_uncertain", "reason": "简短中文理由"}}
  ],
  "new_elements": [
    {{"name": "具体、详细的视觉元素名称", "type": "character|scene|object", "notes": "简短中文备注"}}
  ],
  "shot_notes": ["当前 Shot 才有效的动作、姿态、关系或镜头说明"]
}}
{retry_section}"""


def call_llm_decision(
    prompt: str,
    model: str,
    timeout_seconds: int,
) -> tuple[str, dict[str, Any]]:
    """调用 Seed 2.1 Turbo，并返回原始文本与调用元信息。"""
    return _call_ark_chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=1800,
        timeout_seconds=timeout_seconds,
    )


def call_vlm_annotation(
    frame_path: Path,
    registry: ElementRegistry,
    model: str,
    timeout_seconds: int,
    retry_feedback: str | None = None,
) -> tuple[str, dict[str, Any], int, int]:
    """对单张代表帧做闭集标注，只允许识别已有元素。"""
    try:
        with Image.open(frame_path) as image:
            width, height = image.size
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法读取关键帧 {frame_path}: {exc}") from exc

    mime_type = _image_mime_type(frame_path)
    image_url = f"data:{mime_type};base64," + base64.b64encode(frame_path.read_bytes()).decode("ascii")
    element_text = json.dumps(
        [
            {"id": element.id, "name": element.name, "type": element.type, "notes": element.notes}
            for element in registry.elements
        ],
        ensure_ascii=False,
        indent=2,
    )
    retry_section = ""
    if retry_feedback:
        retry_section = f"\n上一次输出未通过校验：{retry_feedback}\n请仅修正格式或坐标问题。\n"

    prompt = f"""你是长视频项目中的关键帧视觉标注助手。请检查输入图片，但只能从给定的视觉元素集合中选择可见元素进行标注。

严格规则：
1. 不要识别、命名或新增集合以外的元素；即使图片中有其他物体也忽略。
2. 仅当元素在图片中可见且可辨认时才输出它。不可见、被遮挡到无法判断、或不确定的元素不要输出。
3. 坐标使用 0 到 1000 的归一化 xyxy 格式 [x1, y1, x2, y2]，左上为原点。
4. 如果 type 是 scene，且该场景可见，bbox_1000 必须严格写为 [0, 0, 1000, 1000]。
5. character 和 object 使用覆盖该元素主体的紧致边界框；一个元素在一张图中最多输出一次。
6. 只返回 JSON 对象，不得使用 Markdown、解释文字或代码块。

当前视觉元素集合：
{element_text}

请返回：
{{
  "visible_elements": [
    {{"id": "element-0001", "bbox_1000": [0, 0, 1000, 1000]}}
  ]
}}
{retry_section}"""
    raw_text, metadata = _call_ark_chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        model=model,
        max_tokens=1400,
        timeout_seconds=timeout_seconds,
    )
    return raw_text, metadata, width, height


def _call_ark_chat(
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    timeout_seconds: int,
) -> tuple[str, dict[str, Any]]:
    """调用 Ark Chat API，供文本规划和图像标注共用。"""
    api_key = os.getenv("SEEDANCE_API_KEY") or os.getenv("ARK_API_KEY")
    if not api_key:
        raise LLMCallError("缺少 SEEDANCE_API_KEY 或 ARK_API_KEY")

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise LLMCallError(f"调用模型时发生网络错误: {exc}") from exc

    try:
        response_data = response.json()
    except ValueError as exc:
        raise LLMCallError(f"模型返回非 JSON HTTP {response.status_code}: {response.text[:500]}") from exc
    if response.status_code >= 400:
        raise LLMCallError(
            f"模型调用失败 HTTP {response.status_code}: {json.dumps(response_data, ensure_ascii=False)}"
        )

    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMCallError(f"模型响应缺少文本内容: {json.dumps(response_data, ensure_ascii=False)}") from exc
    if not isinstance(content, str):
        raise LLMCallError("模型响应 content 不是字符串")

    metadata = {
        "model": response_data.get("model", model),
        "usage": response_data.get("usage", {}),
        "request_id": response.headers.get("x-request-id"),
    }
    return content, metadata


def parse_and_validate_frame_annotation(
    raw_text: str,
    registry: ElementRegistry,
    width: int,
    height: int,
) -> FrameParseResult:
    """校验 VLM 闭集标注，并将归一化坐标转换为图片像素坐标。"""
    try:
        data = _extract_json_object(raw_text)
    except ValueError as exc:
        return FrameParseResult(elements=None, error=str(exc))
    if not isinstance(data, dict) or not isinstance(data.get("visible_elements"), list):
        return FrameParseResult(elements=None, error="标注输出缺少 visible_elements 列表")

    known_elements = {element.id: element for element in registry.elements}
    annotated: list[AnnotatedElement] = []
    seen_ids: set[str] = set()
    for item in data["visible_elements"]:
        if not isinstance(item, dict):
            return FrameParseResult(elements=None, error="visible_elements 包含非对象项")
        element_id = item.get("id")
        bbox = item.get("bbox_1000")
        if not isinstance(element_id, str) or element_id not in known_elements:
            return FrameParseResult(elements=None, error=f"标注引用了未知元素 {element_id}")
        if element_id in seen_ids:
            return FrameParseResult(elements=None, error=f"元素 {element_id} 在同一帧中被重复标注")
        if not _is_valid_bbox(bbox):
            return FrameParseResult(elements=None, error=f"元素 {element_id} 的 bbox_1000 非法")

        element = known_elements[element_id]
        normalized_bbox = [float(value) for value in bbox]
        if element.type == "scene":
            normalized_bbox = [0.0, 0.0, 1000.0, 1000.0]
        pixel_bbox = _normalized_bbox_to_pixels(normalized_bbox, width, height)
        if element.type != "scene" and (pixel_bbox[2] <= pixel_bbox[0] or pixel_bbox[3] <= pixel_bbox[1]):
            return FrameParseResult(elements=None, error=f"元素 {element_id} 的边界框没有有效面积")

        seen_ids.add(element_id)
        annotated.append(
            AnnotatedElement(
                id=element.id,
                name=element.name,
                type=element.type,
                bbox_1000=normalized_bbox,
                bbox_pixels=pixel_bbox,
            )
        )
    return FrameParseResult(elements=annotated)


def parse_and_validate_decision(raw_text: str, registry: ElementRegistry) -> ParseResult:
    """提取模型 JSON，并保证已有元素状态完整、唯一且与集合一致。"""
    try:
        data = _extract_json_object(raw_text)
    except ValueError as exc:
        return ParseResult(decision=None, error=str(exc))

    if not isinstance(data, dict):
        return ParseResult(decision=None, error="模型输出不是 JSON 对象")

    raw_states = data.get("existing_element_states")
    raw_new_elements = data.get("new_elements")
    raw_notes = data.get("shot_notes", [])
    if not isinstance(raw_states, list) or not isinstance(raw_new_elements, list) or not isinstance(raw_notes, list):
        return ParseResult(decision=None, error="三个顶层字段必须分别为列表")

    expected_ids = {element.id for element in registry.elements}
    parsed_states: list[ExistingElementState] = []
    seen_ids: set[str] = set()
    for item in raw_states:
        if not isinstance(item, dict):
            return ParseResult(decision=None, error="existing_element_states 包含非对象项")
        element_id, state, reason = item.get("id"), item.get("state"), item.get("reason")
        if not isinstance(element_id, str) or not isinstance(state, str) or not isinstance(reason, str):
            return ParseResult(decision=None, error="已有元素状态缺少字符串 id、state 或 reason")
        if element_id in seen_ids:
            return ParseResult(decision=None, error=f"已有元素 {element_id} 被重复分配状态")
        if state not in ELEMENT_STATES:
            return ParseResult(decision=None, error=f"已有元素 {element_id} 使用了非法状态 {state}")
        seen_ids.add(element_id)
        parsed_states.append(ExistingElementState(id=element_id, state=state, reason=reason.strip()))

    if seen_ids != expected_ids:
        missing = sorted(expected_ids - seen_ids)
        unknown = sorted(seen_ids - expected_ids)
        return ParseResult(
            decision=None,
            error=f"已有元素状态必须完整覆盖集合；缺少={missing}，未知={unknown}",
        )

    known_names = registry.names()
    parsed_new_elements: list[NewElement] = []
    seen_new_names: set[str] = set()
    for item in raw_new_elements:
        if not isinstance(item, dict):
            return ParseResult(decision=None, error="new_elements 包含非对象项")
        name, element_type, notes = item.get("name"), item.get("type"), item.get("notes", "")
        if not isinstance(name, str) or not isinstance(element_type, str) or not isinstance(notes, str):
            return ParseResult(decision=None, error="新元素缺少字符串 name、type 或 notes")
        normalized_name = _normalize_name(name)
        if not normalized_name:
            return ParseResult(decision=None, error="新元素名称不能为空")
        if element_type not in ELEMENT_TYPES:
            return ParseResult(decision=None, error=f"新元素 {name} 使用了非法类型 {element_type}")
        if normalized_name in known_names or normalized_name in seen_new_names:
            return ParseResult(decision=None, error=f"新元素 {name} 与已有或本轮新元素重名")
        seen_new_names.add(normalized_name)
        parsed_new_elements.append(NewElement(name=name.strip(), type=element_type, notes=notes.strip()))

    if not all(isinstance(note, str) for note in raw_notes):
        return ParseResult(decision=None, error="shot_notes 必须只包含字符串")

    return ParseResult(
        decision=ShotDecision(
            existing_element_states=parsed_states,
            new_elements=parsed_new_elements,
            shot_notes=[note.strip() for note in raw_notes if note.strip()],
        )
    )


def apply_decision(registry: ElementRegistry, decision: ShotDecision, shot: Shot) -> list[VisualElement]:
    """将通过验证的新元素写入集合，并返回本 Shot 实际新增的元素。"""
    inserted: list[VisualElement] = []
    for proposed in decision.new_elements:
        element = VisualElement(
            id=registry.next_id(),
            name=proposed.name,
            type=proposed.type,
            introduced_at=shot.id,
            notes=proposed.notes,
        )
        registry.elements.append(element)
        inserted.append(element)
    return inserted


def discover_representative_frames(
    project_dir: Path,
    shots: Iterable[Shot],
    web_database_path: Path,
) -> dict[str, list[Path]]:
    """读取 Web 项目当前 Attempt 的关键帧，避免混入已失效的历史 Attempt。"""
    current_attempts: dict[str, str] = {}
    project_id = project_dir.name
    if web_database_path.is_file():
        try:
            with sqlite3.connect(web_database_path) as connection:
                rows = connection.execute(
                    """
                    SELECT shot_id, current_attempt_id
                    FROM shots
                    WHERE project_id = ? AND current_attempt_id IS NOT NULL
                    """,
                    (project_id,),
                ).fetchall()
            current_attempts = {str(shot_id): str(attempt_id) for shot_id, attempt_id in rows}
        except sqlite3.Error:
            # 数据库读取失败时仍可回退到项目目录，实验不依赖 Web 服务在线。
            current_attempts = {}

    frames_by_shot: dict[str, list[Path]] = {}
    for shot in shots:
        attempts_dir = project_dir / "shots" / shot.id / "attempts"
        candidate_root = attempts_dir / current_attempts[shot.id] if shot.id in current_attempts else None
        if candidate_root is not None and candidate_root.is_dir():
            frames = _find_keyframes(candidate_root)
        else:
            # 回退策略只用于旧项目或迁移后的项目目录：选择修改时间最新的 Attempt。
            attempt_dirs = [path for path in attempts_dir.iterdir() if path.is_dir()] if attempts_dir.is_dir() else []
            newest_attempt = max(attempt_dirs, key=lambda path: path.stat().st_mtime, default=None)
            frames = _find_keyframes(newest_attempt) if newest_attempt else []
        frames_by_shot[shot.id] = frames
    return frames_by_shot


def annotate_current_frames(
    shot: Shot,
    frame_paths: Iterable[Path],
    registry: ElementRegistry,
    model: str,
    timeout_seconds: int,
    max_parse_retries: int,
) -> list[FrameAnnotation]:
    """对当前 Shot 的代表帧各标注一次；历史帧不会经过此函数。"""
    annotations: list[FrameAnnotation] = []
    for frame_path in frame_paths:
        attempts: list[dict[str, Any]] = []
        last_error: str | None = None
        annotation: FrameAnnotation | None = None
        for attempt_index in range(1, max_parse_retries + 2):
            raw_text, metadata, width, height = call_vlm_annotation(
                frame_path=frame_path,
                registry=registry,
                model=model,
                timeout_seconds=timeout_seconds,
                retry_feedback=last_error,
            )
            parsed = parse_and_validate_frame_annotation(raw_text, registry, width, height)
            attempts.append(
                {
                    "attempt": attempt_index,
                    "raw_response": raw_text,
                    "metadata": metadata,
                    "parse_error": parsed.error,
                }
            )
            if parsed.elements is not None:
                annotation = FrameAnnotation(
                    source_shot_id=shot.id,
                    frame_path=str(frame_path),
                    width=width,
                    height=height,
                    elements=parsed.elements,
                    attempts=attempts,
                )
                break
            last_error = parsed.error
        if annotation is None:
            raise RuntimeError(
                f"关键帧 {frame_path} 连续 {max_parse_retries + 1} 次未通过 VLM 标注校验: {last_error}"
            )
        annotations.append(annotation)
    return annotations


def evaluate_historical_frames(
    historical_annotations: Iterable[FrameAnnotation],
    decision: ShotDecision,
) -> list[dict[str, Any]]:
    """只依据已有帧标注和当前元素状态，重新归类历史帧，不调用 VLM。"""
    state_by_id = {item.id: item.state for item in decision.existing_element_states}
    evaluations: list[dict[str, Any]] = []
    for annotation in historical_annotations:
        groups: dict[str, list[dict[str, str]]] = {
            "should_reference": [],
            "should_exclude": [],
            "optional_or_uncertain": [],
        }
        for element in annotation.elements:
            state = state_by_id.get(element.id, "optional_or_uncertain")
            groups[state].append({"id": element.id, "name": element.name, "type": element.type})
        evaluations.append(
            {
                "source_shot_id": annotation.source_shot_id,
                "frame_path": annotation.frame_path,
                "should_reference": groups["should_reference"],
                "should_exclude": groups["should_exclude"],
                "optional_or_uncertain": groups["optional_or_uncertain"],
            }
        )
    return evaluations


def write_shot_visualizations(
    shot_id: str,
    current_annotations: Iterable[FrameAnnotation],
    historical_annotations: Iterable[FrameAnnotation],
    decision: ShotDecision,
    visualization_root: Path,
) -> dict[str, list[str]]:
    """在同一轮次目录中写入当前标注图和历史评判图。"""
    round_dir = visualization_root / shot_id
    round_dir.mkdir(parents=True, exist_ok=True)
    current_paths: list[str] = []
    history_paths: list[str] = []

    for annotation in current_annotations:
        output_path = round_dir / f"current_{Path(annotation.frame_path).stem}_annotation.jpg"
        _draw_frame_visualization(
            source_path=Path(annotation.frame_path),
            elements=[asdict(element) for element in annotation.elements],
            output_path=output_path,
            color_for_element=lambda element: CURRENT_ANNOTATION_COLORS[element["type"]],
            label_for_element=lambda element: f"{element['id']}  {element['name']}",
        )
        current_paths.append(str(output_path))

    state_by_id = {item.id: item.state for item in decision.existing_element_states}
    for annotation in historical_annotations:
        output_path = round_dir / f"history_{annotation.source_shot_id}_{Path(annotation.frame_path).stem}_evaluation.jpg"
        _draw_frame_visualization(
            source_path=Path(annotation.frame_path),
            elements=[asdict(element) for element in annotation.elements],
            output_path=output_path,
            color_for_element=lambda element: EVALUATION_COLORS[
                state_by_id.get(element["id"], "optional_or_uncertain")
            ],
            label_for_element=lambda element: (
                f"{EVALUATION_LABELS[state_by_id.get(element['id'], 'optional_or_uncertain')]}  "
                f"{element['id']}  {element['name']}"
            ),
        )
        history_paths.append(str(output_path))
    return {
        "current_annotation_visualizations": current_paths,
        "historical_evaluation_visualizations": history_paths,
    }


def render_log_visualizations(log_path: Path, visualization_root: Path | None = None) -> dict[str, int]:
    """根据已完成的实验日志重绘可视化，不触发任何模型调用。"""
    try:
        log = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取实验日志 {log_path}: {exc}") from exc
    shots = log.get("shots")
    if not isinstance(shots, list):
        raise ValueError("实验日志缺少 shots 列表")

    root = visualization_root or log_path.parent / f"{log_path.stem}_visualizations"
    historical_annotations: list[FrameAnnotation] = []
    current_count = 0
    history_count = 0
    for shot_log in shots:
        shot_data = shot_log.get("shot", {})
        shot_id = shot_data.get("id")
        decision_data = shot_log.get("decision")
        frame_data = shot_log.get("current_frame_annotations", [])
        if not isinstance(shot_id, str) or not isinstance(decision_data, dict) or not isinstance(frame_data, list):
            continue

        decision = _decision_from_dict(decision_data)
        current_annotations = [_frame_annotation_from_dict(item) for item in frame_data]
        visualization_paths = write_shot_visualizations(
            shot_id=shot_id,
            current_annotations=current_annotations,
            historical_annotations=historical_annotations,
            decision=decision,
            visualization_root=root,
        )
        shot_log.update(visualization_paths)
        current_count += len(visualization_paths["current_annotation_visualizations"])
        history_count += len(visualization_paths["historical_evaluation_visualizations"])
        historical_annotations.extend(current_annotations)

    log["visualization_root"] = str(root)
    record_log(log_path, log)
    return {"current_annotations": current_count, "historical_evaluations": history_count}


def _draw_frame_visualization(
    source_path: Path,
    elements: Iterable[dict[str, Any]],
    output_path: Path,
    color_for_element: Any,
    label_for_element: Any,
) -> None:
    """在原图副本绘制边界框和中文标签，绝不修改输入关键帧。"""
    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_visualization_font(image.height)
    line_width = max(3, image.height // 180)
    for element in elements:
        bbox = [int(value) for value in element["bbox_pixels"]]
        color = color_for_element(element)
        draw.rectangle(bbox, outline=color, width=line_width)
        _draw_label(draw, bbox, label_for_element(element), color, font, image.width, image.height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def _load_visualization_font(image_height: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """优先加载本机 CJK 字体，确保中文标签正确显示。"""
    font_size = max(18, min(30, image_height // 28))
    if Path(CHINESE_FONT_PATH).is_file():
        return ImageFont.truetype(CHINESE_FONT_PATH, font_size)
    return ImageFont.load_default()


def _draw_label(
    draw: ImageDraw.ImageDraw,
    bbox: list[int],
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    image_width: int,
    image_height: int,
) -> None:
    """把标签放在框的上方；空间不足时放到框内顶部。"""
    left, top, _, _ = bbox
    label_box = draw.textbbox((0, 0), label, font=font)
    text_width = label_box[2] - label_box[0]
    text_height = label_box[3] - label_box[1]
    x = min(max(0, left), max(0, image_width - text_width - 8))
    y = top - text_height - 10
    if y < 0:
        y = min(max(0, top + 4), max(0, image_height - text_height - 8))
    draw.rectangle((x, y, x + text_width + 8, y + text_height + 6), fill=color)
    draw.text((x + 4, y + 2), label, fill=(255, 255, 255), font=font)


def _decision_from_dict(data: dict[str, Any]) -> ShotDecision:
    """从已经通过校验的日志恢复决策对象，供纯离线可视化使用。"""
    return ShotDecision(
        existing_element_states=[ExistingElementState(**item) for item in data["existing_element_states"]],
        new_elements=[NewElement(**item) for item in data["new_elements"]],
        shot_notes=list(data.get("shot_notes", [])),
    )


def _frame_annotation_from_dict(data: dict[str, Any]) -> FrameAnnotation:
    """从日志恢复帧标注对象，保留已验证的像素坐标。"""
    return FrameAnnotation(
        source_shot_id=data["source_shot_id"],
        frame_path=data["frame_path"],
        width=int(data["width"]),
        height=int(data["height"]),
        elements=[AnnotatedElement(**element) for element in data["elements"]],
        attempts=list(data.get("attempts", [])),
    )


def record_log(output_path: Path, log: dict[str, Any]) -> None:
    """原子写入运行日志，使中断时已完成的 Shot 也可检查。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)


def run_visual_element_memory(
    script_path: Path,
    keyframe_project_dir: Path,
    web_database_path: Path,
    output_path: Path,
    visualization_root: Path,
    model: str,
    timeout_seconds: int,
    max_parse_retries: int,
) -> dict[str, Any]:
    """组织元素规划、历史帧评判和当前帧标注的完整工作流。"""
    shots = load_initial_script(script_path)
    if not keyframe_project_dir.is_dir():
        raise ValueError(f"关键帧项目目录不存在: {keyframe_project_dir}")
    frames_by_shot = discover_representative_frames(keyframe_project_dir, shots, web_database_path)
    registry = ElementRegistry()
    historical_annotations: list[FrameAnnotation] = []
    log: dict[str, Any] = {
        "experiment": "visual_element_memory_text_and_keyframes",
        "script_path": str(script_path),
        "keyframe_project_dir": str(keyframe_project_dir),
        "web_database_path": str(web_database_path),
        "model": model,
        "started_at_unix": time.time(),
        "status": "running",
        "shots": [],
        "final_registry": [],
    }
    record_log(output_path, log)

    for index, shot in enumerate(shots):
        previous_shots = shots[:index]
        attempts: list[dict[str, Any]] = []
        decision: ShotDecision | None = None
        last_error: str | None = None

        for attempt_index in range(1, max_parse_retries + 2):
            prompt = build_llm_prompt(registry, shot, previous_shots, retry_feedback=last_error)
            raw_response, metadata = call_llm_decision(prompt, model, timeout_seconds)
            parsed = parse_and_validate_decision(raw_response, registry)
            attempts.append(
                {
                    "attempt": attempt_index,
                    "raw_response": raw_response,
                    "metadata": metadata,
                    "parse_error": parsed.error,
                }
            )
            if parsed.decision is not None:
                decision = parsed.decision
                break
            last_error = parsed.error

        shot_log: dict[str, Any] = {
            "shot": asdict(shot),
            "registry_before": [asdict(element) for element in registry.elements],
            "attempts": attempts,
        }
        if decision is None:
            shot_log["status"] = "failed"
            shot_log["error"] = f"模型输出连续 {max_parse_retries + 1} 次未通过校验: {last_error}"
            log["shots"].append(shot_log)
            log["final_registry"] = [asdict(element) for element in registry.elements]
            log["status"] = "failed"
            log["finished_at_unix"] = time.time()
            record_log(output_path, log)
            raise RuntimeError(shot_log["error"])

        inserted = apply_decision(registry, decision, shot)
        history_evaluation = evaluate_historical_frames(historical_annotations, decision)
        try:
            current_annotations = annotate_current_frames(
                shot=shot,
                frame_paths=frames_by_shot[shot.id],
                registry=registry,
                model=model,
                timeout_seconds=timeout_seconds,
                max_parse_retries=max_parse_retries,
            )
        except (LLMCallError, RuntimeError, ValueError) as exc:
            shot_log.update(
                {
                    "status": "failed",
                    "decision": asdict(decision),
                    "inserted_elements": [asdict(element) for element in inserted],
                    "historical_frame_evaluation": history_evaluation,
                    "error": str(exc),
                    "registry_after": [asdict(element) for element in registry.elements],
                }
            )
            log["shots"].append(shot_log)
            log["final_registry"] = [asdict(element) for element in registry.elements]
            log["status"] = "failed"
            log["finished_at_unix"] = time.time()
            record_log(output_path, log)
            raise

        shot_log.update(
            {
                "status": "completed",
                "decision": asdict(decision),
                "inserted_elements": [asdict(element) for element in inserted],
                "historical_frame_evaluation": history_evaluation,
                "current_frame_annotations": [asdict(annotation) for annotation in current_annotations],
                "registry_after": [asdict(element) for element in registry.elements],
            }
        )
        visualization_paths = write_shot_visualizations(
            shot_id=shot.id,
            current_annotations=current_annotations,
            historical_annotations=historical_annotations,
            decision=decision,
            visualization_root=visualization_root,
        )
        shot_log.update(visualization_paths)
        historical_annotations.extend(current_annotations)
        log["shots"].append(shot_log)
        log["final_registry"] = [asdict(element) for element in registry.elements]
        record_log(output_path, log)
        print(
            f"已完成 {shot.id}: 新增 {len(inserted)} 个元素，"
            f"标注 {len(current_annotations)} 张当前帧，集合现有 {len(registry.elements)} 个元素。"
        )

    log["status"] = "completed"
    log["finished_at_unix"] = time.time()
    log["visualization_root"] = str(visualization_root)
    record_log(output_path, log)
    return log


def _extract_json_object(raw_text: str) -> Any:
    """容忍模型偶发的前后说明文字，但只接受第一个完整 JSON 对象。"""
    decoder = json.JSONDecoder()
    start = raw_text.find("{")
    if start < 0:
        raise ValueError("模型输出中没有 JSON 对象")
    try:
        value, _ = decoder.raw_decode(raw_text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型 JSON 无法解析: {exc.msg}") from exc
    return value


def _find_keyframes(directory: Path) -> list[Path]:
    """返回一个 Attempt 中全部已保存的代表帧，按文件名稳定排序。"""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*keyframe*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _image_mime_type(frame_path: Path) -> str:
    """根据扩展名选择 data URL 的图片 MIME 类型。"""
    suffix = frame_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    raise ValueError(f"不支持的关键帧图片格式: {frame_path}")


def _is_valid_bbox(value: Any) -> bool:
    """校验 0--1000 范围内、从左上到右下的归一化边界框。"""
    if not isinstance(value, list) or len(value) != 4:
        return False
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return False
    x1, y1, x2, y2 = (float(item) for item in value)
    return (
        0 <= x1 <= NORMALIZED_COORDINATE_MAX
        and 0 <= y1 <= NORMALIZED_COORDINATE_MAX
        and 0 <= x2 <= NORMALIZED_COORDINATE_MAX
        and 0 <= y2 <= NORMALIZED_COORDINATE_MAX
        and x2 >= x1
        and y2 >= y1
    )


def _normalized_bbox_to_pixels(bbox_1000: list[float], width: int, height: int) -> list[int]:
    """将模型统一使用的 0--1000 坐标换算为原始图片像素坐标。"""
    x1, y1, x2, y2 = bbox_1000
    return [
        round(x1 / NORMALIZED_COORDINATE_MAX * width),
        round(y1 / NORMALIZED_COORDINATE_MAX * height),
        round(x2 / NORMALIZED_COORDINATE_MAX * width),
        round(y2 / NORMALIZED_COORDINATE_MAX * height),
    ]


def _normalize_name(name: str) -> str:
    """用于本地重复名称检查，不改变写入集合的原始名称。"""
    return "".join(name.strip().lower().split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="文本集合维护与关键帧闭集标注实验")
    parser.add_argument("--script", type=Path, default=Path("story/little_prince_cn.json"))
    parser.add_argument(
        "--keyframe-project",
        type=Path,
        default=Path(".runtime/web/projects/519f3635af3e43d8a112e27304b0e22a"),
        help="保存关键帧的 Web 项目目录",
    )
    parser.add_argument(
        "--web-database",
        type=Path,
        default=Path(".runtime/web/storymem_web.sqlite3"),
        help="用于定位当前 Attempt 的 Web SQLite 数据库",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".runtime/experiments/visual_element_memory_text_and_keyframes.json"),
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=None,
        help="每个 Shot 的当前标注图与历史评判图的输出根目录",
    )
    parser.add_argument(
        "--render-log",
        type=Path,
        default=None,
        help="仅根据既有实验日志重绘可视化，不调用 LLM 或 VLM",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=90, help="单次模型调用超时秒数")
    parser.add_argument("--max-parse-retries", type=int, default=1, help="JSON 解析失败后的额外调用次数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.render_log is not None:
            root = args.visualization_dir or args.render_log.parent / f"{args.render_log.stem}_visualizations"
            counts = render_log_visualizations(args.render_log, root)
            print(
                f"可视化完成：当前标注图 {counts['current_annotations']} 张，"
                f"历史评判图 {counts['historical_evaluations']} 张。目录: {root}"
            )
            return

        visualization_root = args.visualization_dir or args.output.parent / f"{args.output.stem}_visualizations"
        result = run_visual_element_memory(
            script_path=args.script,
            keyframe_project_dir=args.keyframe_project,
            web_database_path=args.web_database,
            output_path=args.output,
            visualization_root=visualization_root,
            model=args.model,
            timeout_seconds=args.timeout,
            max_parse_retries=args.max_parse_retries,
        )
    except (LLMCallError, RuntimeError, ValueError) as exc:
        print(f"实验失败: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(
        f"实验完成，共处理 {len(result['shots'])} 个 Shot。日志: {args.output}；"
        f"可视化: {visualization_root}"
    )


if __name__ == "__main__":
    main()
