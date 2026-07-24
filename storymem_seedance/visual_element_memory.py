from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import requests
from PIL import Image, ImageDraw, ImageFont

from .models import RunConfig, ShotSpec
from .prompting import escape_prompt_line, image_item_with_metadata, story_prompt_outline


ARK_CHAT_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
DEFAULT_VISUAL_ELEMENT_MODEL = "doubao-seed-2-1-turbo-260628"
VISUAL_ELEMENT_MAX_OUTPUT_TOKENS = 131_072
ELEMENT_TYPES = {"character", "scene", "object"}
ELEMENT_STATES = {"should_reference", "should_exclude", "optional_or_uncertain"}
NORMALIZED_COORDINATE_MAX = 1000
CHINESE_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

TYPE_WEIGHTS = {
    "character": 3.0,
    "scene": 2.0,
    "object": 1.5,
}
REFERENCE_WEIGHT_UNCOVERED = 1.0
REFERENCE_WEIGHT_COVERED = 0.2
OPTIONAL_WEIGHT = 0.1
EXCLUDE_WEIGHT = -0.1
REFERENCE_QUALITY_WEIGHTS = {
    "full": 1.0,
    "partial": 0.2,
    "weak": 0.1,
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
CURRENT_ANNOTATION_COLORS = {
    "character": (220, 70, 70),
    "scene": (150, 70, 210),
    "object": (235, 130, 30),
}
TYPE_ALIASES = {
    "environment": "scene",
    "location": "scene",
    "style": "scene",
    "action": "object",
    "other": "object",
}


class VisualElementMemoryError(RuntimeError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


@dataclass
class VisualElement:
    id: str
    name: str
    type: Literal["character", "scene", "object"]
    introduced_at: str
    notes: str = ""


@dataclass
class ElementRegistry:
    elements: list[VisualElement] = field(default_factory=list)

    def next_id(self) -> str:
        return f"element-{len(self.elements) + 1:04d}"

    def by_id(self) -> dict[str, VisualElement]:
        return {element.id: element for element in self.elements}

    def names(self) -> set[str]:
        return {_normalize_name(element.name) for element in self.elements}


@dataclass
class ExistingElementState:
    id: str
    state: Literal["should_reference", "should_exclude", "optional_or_uncertain"]
    reason: str


@dataclass
class NewElement:
    name: str
    type: Literal["character", "scene", "object"]
    notes: str


@dataclass
class ShotDecision:
    existing_element_states: list[ExistingElementState]
    new_elements: list[NewElement]
    shot_notes: list[str]


@dataclass
class AnnotatedElement:
    id: str
    name: str
    type: Literal["character", "scene", "object"]
    bbox_1000: list[float]
    bbox_pixels: list[int]
    reference_quality: Literal["full", "partial", "weak"] = "full"


@dataclass
class FrameAnnotation:
    source_shot_id: str
    source_scene_num: int
    source_shot_num: int
    frame_path: str
    width: int
    height: int
    elements: list[AnnotatedElement]
    holistic_description: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class VisualElementPlanResult:
    decision: ShotDecision
    inserted_elements: list[VisualElement]
    report_record: dict[str, Any]
    record: dict[str, Any]


@dataclass
class SelectionResult:
    selected_references: list[dict[str, Any]]
    prompt_context: str
    report_record: dict[str, Any]


class VisualElementMemory:
    def __init__(self, config: RunConfig, story_script: dict, shots: list[ShotSpec]) -> None:
        self.config = config
        _validate_visual_reference_budget(config)
        self.story_script = story_script
        self.shots = shots
        self.registry = ElementRegistry()
        self.annotations: list[FrameAnnotation] = []
        self.output_dir = Path(config.output_dir)
        self.state_path = self.output_dir / "visual_element_memory.json"
        self.visualization_root = self.output_dir / "visual_element_memory"
        self.records: list[dict[str, Any]] = []
        self.prompt_by_source = {
            (item.scene_num, item.shot_num): item.prompt
            for item in shots
        }

    def plan_visual_elements_for_shot(self, shot_index: int, shot: ShotSpec) -> VisualElementPlanResult:
        decision, inserted, attempts = self._decide_shot(shot_index, shot)
        plan = self._plan_for_report(decision, inserted)
        report_record = {
            "visual_element_enabled": True,
            "visual_element_selection_mode": self.config.visual_element_selection_mode,
            "visual_element_sink_frame_count": self.config.visual_element_sink_frame_count,
            "visual_element_max_retrieved_frames": self.config.visual_element_max_retrieved_frames,
            "visual_element_decision": asdict(decision),
            "visual_element_inserted": [asdict(element) for element in inserted],
            "visual_element_plan": plan,
        }
        record = {
            "shot": _shot_id(shot_index + 1),
            "scene_num": shot.scene_num,
            "shot_num": shot.shot_num,
            "decision": asdict(decision),
            "inserted": [asdict(element) for element in inserted],
            "registry_after": [asdict(element) for element in self.registry.elements],
            "llm_attempts": attempts,
        }
        self.records.append(record)
        self._write_state("running")
        return VisualElementPlanResult(
            decision=decision,
            inserted_elements=inserted,
            report_record=report_record,
            record=record,
        )

    def select_historical_references_for_shot(
        self,
        shot_index: int,
        shot: ShotSpec,
        plan: VisualElementPlanResult,
    ) -> SelectionResult:
        decision = plan.decision
        inserted = plan.inserted_elements
        selected = self._select_reference_frames(decision)
        retrieved_metadata = self._metadata_for_selected(shot, decision, selected)
        sink_metadata = self._metadata_for_sink(
            excluded_paths={item["source_path"] for item in retrieved_metadata}
        )
        metadata = _assign_reference_indices([*retrieved_metadata, *sink_metadata])
        references = [image_item_with_metadata(item["source_path"], item) for item in metadata]
        prompt_context = self._build_prompt_context(shot, decision, inserted, metadata)
        report_record = {
            **plan.report_record,
            "visual_element_sink_references": sink_metadata,
            "visual_element_retrieved_references": retrieved_metadata,
            "visual_element_selected_references": metadata,
        }
        record = {
            **plan.record,
            "sink_references": sink_metadata,
            "retrieved_references": retrieved_metadata,
            "selected_references": metadata,
        }
        if self.records:
            self.records[-1] = record
        else:
            self.records.append(record)
        self._write_state("running")
        return SelectionResult(
            selected_references=references,
            prompt_context=prompt_context,
            report_record=report_record,
        )

    def prepare_references_for_shot(self, shot_index: int, shot: ShotSpec) -> SelectionResult:
        plan = self.plan_visual_elements_for_shot(shot_index, shot)
        return self.select_historical_references_for_shot(shot_index, shot, plan)
        self._write_state("running")
        return SelectionResult(
            selected_references=references,
            prompt_context=prompt_context,
            report_record=report_record,
        )

    def annotate_completed_shot(
        self,
        shot_index: int,
        shot: ShotSpec,
        keyframe_paths: Iterable[str],
    ) -> list[dict[str, Any]]:
        annotations: list[FrameAnnotation] = []
        for keyframe_path in keyframe_paths:
            annotation = self._annotate_frame(
                shot_id=_shot_id(shot_index),
                shot=shot,
                frame_path=Path(keyframe_path),
            )
            annotations.append(annotation)
        self.annotations.extend(annotations)
        visualization_paths = self._write_current_annotation_visualizations(_shot_id(shot_index), annotations)
        if self.records:
            self.records[-1]["current_frame_annotations"] = [asdict(item) for item in annotations]
            self.records[-1]["current_annotation_visualizations"] = visualization_paths
        self._write_state("running")
        return [asdict(item) for item in annotations]

    def finish(self) -> None:
        self._write_state("completed")

    def snapshot(self) -> dict[str, Any]:
        return {
            "registry": [asdict(element) for element in self.registry.elements],
            "annotations": [asdict(annotation) for annotation in self.annotations],
            "records": self.records,
        }

    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        registry = snapshot.get("registry") or []
        annotations = snapshot.get("annotations") or []
        records = snapshot.get("records") or []
        self.registry = ElementRegistry(
            [
                VisualElement(
                    id=str(item["id"]),
                    name=str(item["name"]),
                    type=_normalize_element_type(item.get("type")),
                    introduced_at=str(item["introduced_at"]),
                    notes=str(item.get("notes") or ""),
                )
                for item in registry
            ]
        )
        self.annotations = [
            FrameAnnotation(
                source_shot_id=str(item["source_shot_id"]),
                source_scene_num=int(item["source_scene_num"]),
                source_shot_num=int(item["source_shot_num"]),
                frame_path=str(item["frame_path"]),
                width=int(item["width"]),
                height=int(item["height"]),
                elements=[
                    AnnotatedElement(
                        id=str(element["id"]),
                        name=str(element["name"]),
                        type=_normalize_element_type(element.get("type")),
                        bbox_1000=[float(value) for value in element["bbox_1000"]],
                        bbox_pixels=[int(value) for value in element["bbox_pixels"]],
                        reference_quality=_normalize_reference_quality(element.get("reference_quality")),
                    )
                    for element in (item.get("elements") or [])
                ],
                holistic_description=str(item.get("holistic_description") or ""),
                attempts=list(item.get("attempts") or []),
            )
            for item in annotations
        ]
        self.records = list(records) if isinstance(records, list) else []

    def _decide_shot(self, shot_index: int, shot: ShotSpec) -> tuple[ShotDecision, list[VisualElement], list[dict[str, Any]]]:
        attempts: list[dict[str, Any]] = []
        last_error: Optional[str] = None
        registry_before = [asdict(element) for element in self.registry.elements]
        previous_shots = self.shots[:shot_index]
        for attempt_index in range(1, self.config.visual_element_max_parse_retries + 2):
            prompt = self._build_decision_prompt(shot_index, shot, previous_shots, last_error)
            raw_text, metadata = _call_ark_chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.visual_element_model,
                max_tokens=VISUAL_ELEMENT_MAX_OUTPUT_TOKENS,
                timeout_seconds=self.config.visual_element_timeout,
            )
            parsed, error = self._parse_decision(raw_text)
            attempts.append(
                {
                    "attempt": attempt_index,
                    "prompt": prompt,
                    "metadata": metadata,
                    "parse_error": error,
                    "registry_before": registry_before,
                    "raw_response": raw_text,
                }
            )
            if parsed is not None:
                inserted = self._apply_decision(parsed, _shot_id(shot_index + 1))
                return parsed, inserted, attempts
            last_error = error
        raise VisualElementMemoryError(
            f"视觉元素决策连续解析失败: {last_error}",
            details={
                "failed_stage": "visual_elements_plan",
                "last_error": last_error,
                "llm_attempts": attempts,
                "registry_before": registry_before,
            },
        )

    def _build_decision_prompt(
        self,
        shot_index: int,
        shot: ShotSpec,
        previous_shots: Iterable[ShotSpec],
        retry_feedback: Optional[str],
    ) -> str:
        previous_prompt_text = "\n".join(
            f"- {_shot_id(index + 1)}: {item.prompt}" for index, item in enumerate(previous_shots)
        ) or "（当前是第一个 Shot，没有前序 Prompt。）"
        registry_text = json.dumps([asdict(element) for element in self.registry.elements], ensure_ascii=False, indent=2)
        retry_section = ""
        if retry_feedback:
            retry_section = f"\n上一次回答未通过本地校验：{retry_feedback}\n请修正格式或遗漏字段。\n"
        return f"""你是长视频生成项目中的视觉元素规划助手。请只根据文字剧本，维护一个跨 Shot 的视觉元素集合。

视觉元素仅允许三类：
- character：有姿态、表情、语言或动作变化的角色实体；
- scene：通过画面整体判断的宏观背景或地点；
- object：画面中可检测或分割的物体。

工作原则：
1. 元素名称必须具体，能够稳定指代特定角色、场景或物体，不要使用“男孩”“花”“桌子”这类泛称。
2. 只有当前 Prompt 明确引入新的、对画面叙事有意义的视觉实体时，才放入 new_elements。
3. 必须给每个已有元素分配状态：
   - should_reference：当前 Shot 需要它作为视觉一致性的参考；
   - should_exclude：当前 Prompt 有明确证据表明它不应出现在画面中；
   - optional_or_uncertain：可能有帮助，但既非必需也非明确禁止。
4. 没提到某元素不等于 should_exclude；请根据完整剧情上下文谨慎判断。
5. 姿态、动作、人物关系和镜头变化写入 shot_notes，不要作为全局元素。
6. 只返回 JSON 对象，不得使用 Markdown、解释文字或代码块。

现有元素集合：
{registry_text}

此前所有 Shot 的原始 video_prompt：
{previous_prompt_text}

当前待处理 Shot：
- id: {_shot_id(shot_index + 1)}
- scene: {shot.scene_num}
- shot: {shot.shot_num}
- cut: {shot.is_cut}
- video_prompt: {shot.prompt}

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

    def _parse_decision(self, raw_text: str) -> tuple[Optional[ShotDecision], Optional[str]]:
        try:
            data = _extract_json_object(raw_text)
        except ValueError as exc:
            return None, str(exc)
        if not isinstance(data, dict):
            return None, "模型输出不是 JSON 对象"
        raw_states = data.get("existing_element_states")
        raw_new = data.get("new_elements")
        raw_notes = data.get("shot_notes", [])
        if not isinstance(raw_states, list) or not isinstance(raw_new, list) or not isinstance(raw_notes, list):
            return None, "顶层字段类型不合法"

        expected_ids = {element.id for element in self.registry.elements}
        seen: set[str] = set()
        states: list[ExistingElementState] = []
        for item in raw_states:
            if not isinstance(item, dict):
                return None, "existing_element_states 包含非对象项"
            element_id = item.get("id")
            state = item.get("state")
            reason = item.get("reason")
            if not isinstance(element_id, str) or not isinstance(state, str) or not isinstance(reason, str):
                return None, "已有元素状态缺少字符串字段"
            if element_id in seen:
                return None, f"已有元素 {element_id} 被重复分配状态"
            if state not in ELEMENT_STATES:
                return None, f"已有元素 {element_id} 使用非法状态 {state}"
            seen.add(element_id)
            states.append(ExistingElementState(id=element_id, state=state, reason=reason.strip()))
        if seen != expected_ids:
            return None, f"已有元素状态必须完整覆盖集合；缺少={sorted(expected_ids - seen)}，未知={sorted(seen - expected_ids)}"

        known_names = self.registry.names()
        seen_names: set[str] = set()
        new_elements: list[NewElement] = []
        for item in raw_new:
            if not isinstance(item, dict):
                return None, "new_elements 包含非对象项"
            name = item.get("name")
            element_type = item.get("type")
            notes = item.get("notes", "")
            if not isinstance(name, str) or not isinstance(element_type, str) or not isinstance(notes, str):
                return None, "新元素缺少字符串字段"
            normalized = _normalize_name(name)
            if not normalized:
                return None, "新元素名称为空"
            if element_type not in ELEMENT_TYPES:
                return None, f"新元素 {name} 类型非法"
            if normalized in known_names or normalized in seen_names:
                return None, f"新元素 {name} 与已有或本轮新元素重名"
            seen_names.add(normalized)
            new_elements.append(NewElement(name=name.strip(), type=element_type, notes=notes.strip()))
        if not all(isinstance(note, str) for note in raw_notes):
            return None, "shot_notes 必须只包含字符串"
        return ShotDecision(states, new_elements, [note.strip() for note in raw_notes if note.strip()]), None

    def _apply_decision(self, decision: ShotDecision, shot_id: str) -> list[VisualElement]:
        inserted: list[VisualElement] = []
        for proposed in decision.new_elements:
            element = VisualElement(
                id=self.registry.next_id(),
                name=proposed.name,
                type=proposed.type,
                introduced_at=shot_id,
                notes=proposed.notes,
            )
            self.registry.elements.append(element)
            inserted.append(element)
        return inserted

    def _select_reference_frames(self, decision: ShotDecision) -> list[tuple[FrameAnnotation, dict[str, Any]]]:
        max_retrieved = int(self.config.visual_element_max_retrieved_frames)
        if not self.annotations or max_retrieved <= 0:
            return []
        selected: list[tuple[FrameAnnotation, dict[str, Any]]] = []
        covered: set[str] = set()
        remaining = list(self.annotations)
        while len(selected) < max_retrieved and remaining:
            best: Optional[FrameAnnotation] = None
            best_details: Optional[dict[str, Any]] = None
            best_score = float("-inf")
            for annotation in remaining:
                score, details = self._score_frame(annotation, decision, covered)
                if score > best_score:
                    best = annotation
                    best_details = details
                    best_score = score
            if best is None or best_details is None:
                break
            selected.append((best, best_details))
            covered.update(best_details["newly_covered_element_ids"])
            remaining.remove(best)
        return selected

    def _metadata_for_sink(self, *, excluded_paths: set[str]) -> list[dict[str, Any]]:
        count = int(self.config.visual_element_sink_frame_count)
        if count <= 0:
            return []
        metadata: list[dict[str, Any]] = []
        seen: set[str] = set(excluded_paths)
        for annotation in self.annotations:
            if len(metadata) >= count:
                break
            if annotation.frame_path in seen:
                continue
            seen.add(annotation.frame_path)
            metadata.append(
                {
                    "source_path": annotation.frame_path,
                    "file": Path(annotation.frame_path).name,
                    "label": "visual_sink_memory",
                    "roles": ["visual_sink_memory"],
                    "source_scene_num": annotation.source_scene_num,
                    "source_shot_num": annotation.source_shot_num,
                    "source_prompt": self.prompt_by_source.get(
                        (annotation.source_scene_num, annotation.source_shot_num),
                        "",
                    ),
                    "reference_intent": "早期参考锚点：用于稳定角色身份、整体风格和画面基调；不参与视觉元素覆盖检索。",
                }
            )
        return metadata

    def _score_frame(
        self,
        annotation: FrameAnnotation,
        decision: ShotDecision,
        covered: set[str],
    ) -> tuple[float, dict[str, Any]]:
        state_by_id = {item.id: item.state for item in decision.existing_element_states}
        visible_by_id = {element.id: element for element in annotation.elements}
        score = 0.0
        groups = {
            "should_reference": [],
            "should_exclude": [],
            "optional_or_uncertain": [],
        }
        newly_covered: list[str] = []
        already_covered: list[str] = []
        for element in visible_by_id.values():
            state = state_by_id.get(element.id)
            if state not in groups:
                continue
            quality = _visibility_quality(element, annotation.width, annotation.height)
            reference_quality_weight = self._reference_quality_weight(element)
            type_weight = self._type_weight(element.type)
            groups[state].append(_element_summary(element, quality, reference_quality_weight))
            if state == "should_reference":
                if self.config.visual_element_selection_mode == "greedy_coverage" and element.id in covered:
                    ref_weight = self._state_weight("reference_covered")
                else:
                    ref_weight = self._state_weight("reference_uncovered")
                if element.reference_quality == "full":
                    if self.config.visual_element_selection_mode == "greedy_coverage" and element.id in covered:
                        already_covered.append(element.id)
                    else:
                        newly_covered.append(element.id)
                score += ref_weight * type_weight * quality * reference_quality_weight
            elif state == "optional_or_uncertain":
                score += self._state_weight("optional") * type_weight * quality * reference_quality_weight
            elif state == "should_exclude":
                score += self._state_weight("exclude") * type_weight * quality * reference_quality_weight
        details = {
            "score": score,
            "newly_covered_element_ids": newly_covered,
            "already_covered_element_ids": already_covered,
            "should_reference": groups["should_reference"],
            "should_exclude": groups["should_exclude"],
            "optional_or_uncertain": groups["optional_or_uncertain"],
        }
        return score, details

    def _type_weight(self, element_type: str) -> float:
        defaults = TYPE_WEIGHTS
        values = {
            "character": getattr(self.config, "visual_element_weight_character", defaults["character"]),
            "scene": getattr(self.config, "visual_element_weight_scene", defaults["scene"]),
            "object": getattr(self.config, "visual_element_weight_object", defaults["object"]),
        }
        return float(values.get(element_type, values["object"]))

    def _state_weight(self, name: str) -> float:
        defaults = {
            "reference_uncovered": REFERENCE_WEIGHT_UNCOVERED,
            "reference_covered": REFERENCE_WEIGHT_COVERED,
            "optional": OPTIONAL_WEIGHT,
            "exclude": EXCLUDE_WEIGHT,
        }
        return float(getattr(self.config, f"visual_element_weight_{name}", defaults[name]))

    def _reference_quality_weight(self, element: AnnotatedElement) -> float:
        defaults = REFERENCE_QUALITY_WEIGHTS
        values = {
            "full": getattr(self.config, "visual_element_weight_quality_full", defaults["full"]),
            "partial": getattr(self.config, "visual_element_weight_quality_partial", defaults["partial"]),
            "weak": getattr(self.config, "visual_element_weight_quality_weak", defaults["weak"]),
        }
        return float(values.get(element.reference_quality, values["full"]))

    def _metadata_for_selected(
        self,
        shot: ShotSpec,
        decision: ShotDecision,
        selected: list[tuple[FrameAnnotation, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        state_by_id = {item.id: item.state for item in decision.existing_element_states}
        metadata: list[dict[str, Any]] = []
        round_dir = self.visualization_root / f"scene-{shot.scene_num:02d}_shot-{shot.shot_num:02d}"
        for index, (annotation, details) in enumerate(selected, start=1):
            visualization_path = round_dir / f"selected_ref_{index:02d}_{Path(annotation.frame_path).stem}.jpg"
            _draw_frame_visualization(
                source_path=Path(annotation.frame_path),
                elements=[asdict(element) for element in annotation.elements],
                output_path=visualization_path,
                color_for_element=lambda element: EVALUATION_COLORS[
                    state_by_id.get(element["id"], "optional_or_uncertain")
                ],
                label_for_element=lambda element: (
                    f"{EVALUATION_LABELS[state_by_id.get(element['id'], 'optional_or_uncertain')]}  "
                    f"{element['id']}  {element['name']}"
                ),
            )
            metadata.append(
                {
                    "source_path": annotation.frame_path,
                    "visualization_path": str(visualization_path),
                    "file": Path(annotation.frame_path).name,
                    "label": "visual_element_memory",
                    "roles": ["visual_element_memory"],
                    "source_scene_num": annotation.source_scene_num,
                    "source_shot_num": annotation.source_shot_num,
                    "source_prompt": self.prompt_by_source.get(
                        (annotation.source_scene_num, annotation.source_shot_num),
                        "",
                    ),
                    "score": details["score"],
                    "visual_element_selection": details,
                    "reference_intent": self._reference_intent_text(details),
                    "holistic_description": annotation.holistic_description,
                }
            )
        return metadata

    def _reference_intent_text(self, details: dict[str, Any]) -> str:
        parts = []
        if details["should_reference"]:
            parts.append("应参考：" + "、".join(item["name"] for item in details["should_reference"]))
        if details["should_exclude"]:
            parts.append("不应引入：" + "、".join(item["name"] for item in details["should_exclude"]))
        if details["optional_or_uncertain"]:
            parts.append("可选/不确定：" + "、".join(item["name"] for item in details["optional_or_uncertain"]))
        return "；".join(parts) if parts else "无可见已知元素"

    def _build_prompt_context(
        self,
        shot: ShotSpec,
        decision: ShotDecision,
        inserted: list[VisualElement],
        references: list[dict[str, Any]],
    ) -> str:
        plan = self._plan_for_report(decision, inserted)
        reference_lines = []
        if references:
            for ref in references:
                if "visual_sink_memory" in (ref.get("roles") or []):
                    reference_lines.extend(
                        [
                            f"参考图片 Image {ref['reference_index']} 来自 Scene {ref.get('source_scene_num')} / Shot {ref.get('source_shot_num')}：",
                            "这是一张早期参考锚点图，仅用于稳定整体角色身份、画面风格、色彩气质和长期视觉一致性。",
                            "不要因为这张图中的旧场景、旧动作或未在当前镜头要求中出现的物体，就把它们带入当前画面。",
                        ]
                    )
                    continue
                selection = ref["visual_element_selection"]
                holistic_description = str(ref.get("holistic_description") or "").strip()
                reference_lines.extend(
                    [
                        f"参考图片 Image {ref['reference_index']} 来自 Scene {ref.get('source_scene_num')} / Shot {ref.get('source_shot_num')}：",
                    ]
                )
                if holistic_description:
                    reference_lines.append(holistic_description)
                reference_lines.extend(
                    [
                        "# 客观参考约束",
                        "应参考其中的：",
                        _bullet_names(selection["should_reference"]),
                        "不应引入其中的：",
                        _bullet_names(selection["should_exclude"]),
                    ]
                )
        else:
            reference_lines.append("本镜头不提供历史参考图。")

        return "\n".join(
            [
                "[0. 完整剧本上下文]",
                "以下是完整视频剧本的 shot 级描述，仅用于理解剧情上下文、人物关系、场景切换，以及理解哪些历史元素不应出现在当前镜头中。",
                "当前任务只生成指定 shot。不要生成其他 shot 的内容，也不要提前引入未来 shot 的人物、场景或物体。",
                story_prompt_outline(self.story_script),
                "",
                "[1. 当前 shot 的生成任务]",
                f"现在请只生成 Scene {shot.scene_num} / Shot {shot.shot_num}：",
                escape_prompt_line(shot.prompt),
                "如果本次输入媒体包含参考视频，则请生成参考视频的延长视频：生成的视频应从参考视频的最后一帧之后的下一时刻开始，自然延续其中的人物动作、物体运动、镜头运动方向和速度，并保持整体画面风格、色彩气质和视觉元素的一致性。",
                "",
                "[2. 本 shot 的视觉元素计划]",
                "本镜头应参考并保持一致的历史元素：",
                _bullet_names(plan["should_reference"]),
                "本镜头不应引入的历史元素：",
                _bullet_names(plan["should_exclude"]),
                "不确定的视觉元素：",
                _bullet_names(plan["optional_or_uncertain"]),
                "本镜头新引入的视觉元素：",
                _bullet_names(plan["new_elements"]),
                "",
                "[3. 参考图使用说明]",
                "以下参考图片仅用于保持指定视觉元素的一致性，不代表整张图都应被复刻。",
                "下文的 Image 1、Image 2 等参考图片的顺序编号。可能存在参考视频，用于维持镜头的连续性，但与下列描述无关，下列描述针对的是上传的参考图片。",
                *reference_lines,
                "",
                "[4. 总体约束]",
                "当前 shot 的生成任务优先级最高。",
                "完整剧本仅用于上下文理解，不要生成其他镜头的故事内容。对于本镜头shot中未被详细描述的元素，可参考剧本其他镜头中的描述及对应的参考图。",
                "请只参考视觉元素计划中明确指定的视觉元素，不要将参考图中的与本镜头无关的元素加入当前画面。",
                "如果参考图说明与当前 shot 描述冲突，以当前 shot 描述和本 shot 的视觉元素计划为准。",
                "除非剧本明确要求，不要在视频结束时添加镜头远离、淡出、变暗等转场效果。",
                "Maintain a stable shot scale through the end of the clip unless the script explicitly asks otherwise.",
                "Do NOT zoom out, pull the camera back, fade out, dim the image, or add an ending transition at the end of video, unless the script explicitly asks for it, as next shot may continue the same scene."
            ]
        )

    def _plan_for_report(self, decision: ShotDecision, inserted: list[VisualElement]) -> dict[str, list[dict[str, str]]]:
        elements = self.registry.by_id()
        groups = {
            "should_reference": [],
            "should_exclude": [],
            "optional_or_uncertain": [],
        }
        for item in decision.existing_element_states:
            element = elements.get(item.id)
            if element is not None:
                groups[item.state].append(
                    {
                        "id": element.id,
                        "name": element.name,
                        "type": element.type,
                        "reason": item.reason,
                    }
                )
        groups["new_elements"] = [
            {"id": item.id, "name": item.name, "type": item.type, "reason": item.notes}
            for item in inserted
        ]
        return groups

    def _annotate_frame(self, shot_id: str, shot: ShotSpec, frame_path: Path) -> FrameAnnotation:
        if not frame_path.is_file():
            raise VisualElementMemoryError(f"关键帧不存在: {frame_path}")
        with Image.open(frame_path) as opened:
            width, height = opened.size
        attempts: list[dict[str, Any]] = []
        last_error: Optional[str] = None
        for attempt_index in range(1, self.config.visual_element_max_parse_retries + 2):
            raw_text, metadata = self._call_vlm_annotation(shot, frame_path, last_error)
            parsed, error = self._parse_annotation(raw_text, width, height)
            attempts.append({"attempt": attempt_index, "metadata": metadata, "parse_error": error, "raw_response": raw_text})
            if parsed is not None:
                elements, holistic_description = parsed
                return FrameAnnotation(
                    source_shot_id=shot_id,
                    source_scene_num=shot.scene_num,
                    source_shot_num=shot.shot_num,
                    frame_path=str(frame_path),
                    width=width,
                    height=height,
                    elements=elements,
                    holistic_description=holistic_description,
                    attempts=attempts,
                )
            last_error = error
        raise VisualElementMemoryError(
            f"关键帧 {frame_path} VLM 标注连续解析失败: {last_error}",
            details={
                "failed_stage": "keyframe_annotation",
                "frame_path": str(frame_path),
                "last_error": last_error,
                "vlm_attempts": attempts,
            },
        )

    def _call_vlm_annotation(self, shot: ShotSpec, frame_path: Path, retry_feedback: Optional[str]) -> tuple[str, dict[str, Any]]:
        mime_type = _image_mime_type(frame_path)
        image_url = f"data:{mime_type};base64," + base64.b64encode(frame_path.read_bytes()).decode("ascii")
        element_text = json.dumps(
            [{"id": item.id, "name": item.name, "type": item.type, "notes": item.notes} for item in self.registry.elements],
            ensure_ascii=False,
            indent=2,
        )
        story_context = json.dumps(_shot_context_until(self.shots, shot), ensure_ascii=False, indent=2)
        retry_section = f"\n上一次输出未通过校验：{retry_feedback}\n请仅修正格式或坐标问题。\n" if retry_feedback else ""
        prompt = f"""你是长视频项目中的关键帧视觉标注助手。请检查输入图片，但只能从给定视觉元素集合中选择可见元素进行标注，并为关键帧补充整体描述。

严格规则：
1. 不要识别、命名或新增集合以外的元素。
2. 仅当元素在图片中可见且可辨认时才输出。
3. 坐标使用 0 到 1000 的归一化 xyxy 格式 [x1, y1, x2, y2]。
4. 如果 type 是 scene，且该场景可见，bbox_1000 必须严格写为 [0, 0, 1000, 1000]。
5. character 和 object 使用覆盖该元素主体的紧致边界框；一个元素在一张图中最多输出一次。
6. 对 type 为 character 的元素，必须输出 reference_quality，取值只能是 full、partial、weak：
   - full：人物完整、清晰，且人脸清晰可见，可作为完整角色参考；
   - partial：人物部分可见，或脸/服装/主体信息不完整，只能作为局部参考；
   - weak：背影、远景、严重遮挡、模糊、小尺寸等，不能作为完整角色参考。
   非 character 元素可以省略 reference_quality。
7. holistic_description 用中文输出，描述该关键帧的整体画面，不要编造画面中不可见的信息；剧情关系可基于下方剧情上下文判断。
8. 只返回 JSON 对象，不得使用 Markdown、解释文字或代码块。

截至当前 shot 的剧情上下文：
{story_context}

当前视觉元素集合：
{element_text}

请返回：
{{
  "holistic_description": "一句到三句，包含主要人物/物体/环境构成、画面氛围/构图/动作/镜头状态，以及该画面与当前剧情的关系；无法可靠描述则为空字符串。",
  "visible_elements": [
    {{"id": "element-0001", "bbox_1000": [0, 0, 1000, 1000], "reference_quality": "full"}}
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
            model=self.config.visual_element_model,
            max_tokens=VISUAL_ELEMENT_MAX_OUTPUT_TOKENS,
            timeout_seconds=self.config.visual_element_timeout,
        )
        metadata = {**metadata, "prompt": prompt, "frame_path": str(frame_path)}
        return raw_text, metadata

    def _parse_annotation(self, raw_text: str, width: int, height: int) -> tuple[Optional[tuple[list[AnnotatedElement], str]], Optional[str]]:
        try:
            data = _extract_json_object(raw_text)
        except ValueError as exc:
            return None, str(exc)
        if not isinstance(data, dict) or not isinstance(data.get("visible_elements"), list):
            return None, "标注输出缺少 visible_elements 列表"
        holistic_description = str(data.get("holistic_description") or "").strip()
        known = self.registry.by_id()
        seen: set[str] = set()
        annotated: list[AnnotatedElement] = []
        for item in data["visible_elements"]:
            if not isinstance(item, dict):
                return None, "visible_elements 包含非对象项"
            element_id = item.get("id")
            bbox = item.get("bbox_1000")
            if not isinstance(element_id, str) or element_id not in known:
                return None, f"标注引用未知元素 {element_id}"
            if element_id in seen:
                return None, f"元素 {element_id} 重复标注"
            if not _is_valid_bbox(bbox):
                return None, f"元素 {element_id} 的 bbox_1000 非法"
            element = known[element_id]
            reference_quality = (
                _normalize_reference_quality(item.get("reference_quality"))
                if element.type == "character"
                else "full"
            )
            normalized = [float(value) for value in bbox]
            if element.type == "scene":
                normalized = [0.0, 0.0, 1000.0, 1000.0]
            pixel_bbox = _normalized_bbox_to_pixels(normalized, width, height)
            if element.type != "scene" and (pixel_bbox[2] <= pixel_bbox[0] or pixel_bbox[3] <= pixel_bbox[1]):
                return None, f"元素 {element_id} 的边界框没有有效面积"
            seen.add(element_id)
            annotated.append(
                AnnotatedElement(
                    id=element.id,
                    name=element.name,
                    type=element.type,
                    bbox_1000=normalized,
                    bbox_pixels=pixel_bbox,
                    reference_quality=reference_quality,
                )
            )
        return (annotated, holistic_description), None

    def _write_current_annotation_visualizations(
        self,
        shot_id: str,
        annotations: Iterable[FrameAnnotation],
    ) -> list[str]:
        output_paths: list[str] = []
        round_dir = self.visualization_root / shot_id
        for annotation in annotations:
            output_path = round_dir / f"current_{Path(annotation.frame_path).stem}_annotation.jpg"
            _draw_frame_visualization(
                source_path=Path(annotation.frame_path),
                elements=[asdict(element) for element in annotation.elements],
                output_path=output_path,
                color_for_element=lambda element: CURRENT_ANNOTATION_COLORS[
                    _normalize_element_type(element.get("type"))
                ],
                label_for_element=lambda element: f"{element['id']}  {element['name']}",
            )
            output_paths.append(str(output_path))
        return output_paths

    def _write_state(self, status: str) -> None:
        payload = {
            "status": status,
            "updated_at_unix": time.time(),
            "model": self.config.visual_element_model,
            "selection_mode": self.config.visual_element_selection_mode,
            "registry": [asdict(element) for element in self.registry.elements],
            "annotations": [asdict(annotation) for annotation in self.annotations],
            "records": self.records,
        }
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.state_path)


def _call_ark_chat(
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    timeout_seconds: int,
) -> tuple[str, dict[str, Any]]:
    api_key = os.getenv("SEEDANCE_API_KEY") or os.getenv("ARK_API_KEY")
    if not api_key:
        raise VisualElementMemoryError("缺少 SEEDANCE_API_KEY 或 ARK_API_KEY")
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    try:
        response = requests.post(
            ARK_CHAT_URL,
            headers={"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise VisualElementMemoryError(f"调用 Seed2.1 Turbo 失败: {exc}") from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise VisualElementMemoryError(f"模型返回非 JSON HTTP {response.status_code}: {response.text[:500]}") from exc
    if response.status_code >= 400:
        raise VisualElementMemoryError(f"模型调用失败 HTTP {response.status_code}: {json.dumps(data, ensure_ascii=False)}")
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VisualElementMemoryError(f"模型响应缺少文本内容: {json.dumps(data, ensure_ascii=False)}") from exc
    if not isinstance(content, str):
        raise VisualElementMemoryError("模型响应 content 不是字符串")
    return content, {
        "model": data.get("model", model),
        "usage": data.get("usage", {}),
        "request_id": response.headers.get("x-request-id"),
    }


def _validate_visual_reference_budget(config: RunConfig) -> None:
    sink_count = int(config.visual_element_sink_frame_count)
    retrieved_count = int(config.visual_element_max_retrieved_frames)
    if sink_count < 0:
        raise VisualElementMemoryError("visual_element_sink_frame_count must be >= 0")
    if retrieved_count < 0:
        raise VisualElementMemoryError("visual_element_max_retrieved_frames must be >= 0")
    if sink_count + retrieved_count > 9:
        raise VisualElementMemoryError(
            "visual-element static references exceed Seedance's 9-image limit: "
            f"sink={sink_count}, retrieved={retrieved_count}"
        )


def _assign_reference_indices(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        updated = dict(item)
        updated["reference_index"] = index
        indexed.append(updated)
    return indexed


def _shot_context_until(shots: list[ShotSpec], current: ShotSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in shots:
        rows.append(
            {
                "scene_num": item.scene_num,
                "shot_num": item.shot_num,
                "video_prompt": item.prompt,
            }
        )
        if (
            item is current
            or (
                item.scene_num == current.scene_num
                and item.shot_num == current.shot_num
                and item.prompt == current.prompt
            )
        ):
            break
    return rows


def _extract_json_object(raw_text: str) -> Any:
    decoder = json.JSONDecoder()
    start = raw_text.find("{")
    if start < 0:
        raise ValueError("模型输出中没有 JSON 对象")
    try:
        value, _ = decoder.raw_decode(raw_text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型 JSON 无法解析: {exc.msg}") from exc
    return value


def _visibility_quality(element: AnnotatedElement, width: int, height: int) -> float:
    if element.type == "scene":
        return 1.0
    x1, y1, x2, y2 = element.bbox_pixels
    area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / max(1, width * height)
    if area_ratio < 0.01:
        return 0.3
    if area_ratio < 0.10:
        return 0.6
    return 1.0


def _normalize_reference_quality(value: object) -> Literal["full", "partial", "weak"]:
    raw = str(value or "full").strip().lower()
    if raw in REFERENCE_QUALITY_WEIGHTS:
        return raw  # type: ignore[return-value]
    return "full"


def _element_summary(element: AnnotatedElement, quality: float, reference_quality_weight: float) -> dict[str, Any]:
    return {
        "id": element.id,
        "name": element.name,
        "type": element.type,
        "quality": quality,
        "reference_quality": element.reference_quality,
        "reference_quality_weight": reference_quality_weight,
        "bbox_pixels": element.bbox_pixels,
    }


def _draw_frame_visualization(
    source_path: Path,
    elements: Iterable[dict[str, Any]],
    output_path: Path,
    color_for_element: Any,
    label_for_element: Any,
) -> None:
    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font(image.height)
    line_width = max(3, image.height // 180)
    for element in elements:
        bbox = [int(value) for value in element["bbox_pixels"]]
        color = color_for_element(element)
        draw.rectangle(bbox, outline=color, width=line_width)
        _draw_label(draw, bbox, label_for_element(element), color, font, image.width, image.height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def _load_font(image_height: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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


def _image_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    raise VisualElementMemoryError(f"不支持的图片格式: {path}")


def _normalize_element_type(value: Any) -> Literal["character", "scene", "object"]:
    raw = str(value or "object").strip()
    normalized = TYPE_ALIASES.get(raw, raw)
    if normalized in ELEMENT_TYPES:
        return normalized  # type: ignore[return-value]
    return "object"


def _is_valid_bbox(value: Any) -> bool:
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
    x1, y1, x2, y2 = bbox_1000
    return [
        round(x1 / NORMALIZED_COORDINATE_MAX * width),
        round(y1 / NORMALIZED_COORDINATE_MAX * height),
        round(x2 / NORMALIZED_COORDINATE_MAX * width),
        round(y2 / NORMALIZED_COORDINATE_MAX * height),
    ]


def _normalize_name(name: str) -> str:
    return "".join(name.strip().lower().split())


def _shot_id(index: int) -> str:
    return f"shot-{index:04d}"


def _bullet_names(items: list[dict[str, Any]]) -> str:
    if not items:
        return "- 无"
    return "\n".join(f"- {item.get('name', '')}" for item in items)
