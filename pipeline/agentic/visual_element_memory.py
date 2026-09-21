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

from .language import detect_story_language, output_language_instruction
from ..models import RunConfig, ShotSpec
from ..prompting import escape_prompt_line, image_item_with_metadata, story_prompt_outline


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
    "should_reference": "reference",
    "should_exclude": "exclude",
    "optional_or_uncertain": "optional",
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
            f"Visual element decision parsing failed repeatedly: {last_error}",
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
        previous_shots = list(previous_shots)
        output_language = detect_story_language(
            [item.prompt for item in previous_shots] + [shot.prompt]
        )
        previous_prompt_text = "\n".join(
            f"- {_shot_id(index + 1)}: {item.prompt}" for index, item in enumerate(previous_shots)
        ) or "(This is the first shot; no prior prompt is available.)"
        registry_text = json.dumps([asdict(element) for element in self.registry.elements], ensure_ascii=False, indent=2)
        retry_section = ""
        if retry_feedback:
            retry_section = f"\nThe previous response failed local validation: {retry_feedback}\nFix only the format or missing fields.\n"
        return f"""You are the visual-element planning assistant for a long-form video generation project. Use only the written script to maintain a visual-element registry across shots.

Only these visual element types are allowed:
- character: a character entity with pose, expression, speech, or action changes;
- scene: a macro background, environment, or location recognized as a whole;
- object: a visible object that can be detected or segmented.

Rules:
1. Element names must be specific and stably identify a particular character, scene, or object. Do not use generic names such as "boy", "flower", or "table". {output_language_instruction(output_language)}
2. Add an item to new_elements only when the current prompt explicitly introduces a new visual entity that matters to the visual narrative.
3. Assign every existing element one state:
   - should_reference: the current shot needs it as a visual-consistency reference;
   - should_exclude: the current prompt provides clear evidence that it should not appear;
   - optional_or_uncertain: it may help, but is neither required nor clearly prohibited.
4. An element not mentioned is not automatically should_exclude; judge conservatively from the full narrative context.
5. Put poses, actions, relationships, and camera changes in shot_notes rather than the global registry.
6. Return only a JSON object. Do not use Markdown, explanations, or code fences.

Existing element registry:
{registry_text}

Original video_prompt values for all preceding shots:
{previous_prompt_text}

Current shot:
- id: {_shot_id(shot_index + 1)}
- scene: {shot.scene_num}
- shot: {shot.shot_num}
- cut: {shot.is_cut}
- video_prompt: {shot.prompt}

Return exactly this schema. existing_element_states must cover every existing registry id exactly once:
{{
  "existing_element_states": [
    {{"id": "existing element id", "state": "should_reference|should_exclude|optional_or_uncertain", "reason": "brief reason in the detected script language"}}
  ],
  "new_elements": [
    {{"name": "specific, detailed visual-element name in the detected script language", "type": "character|scene|object", "notes": "brief notes in the detected script language"}}
  ],
  "shot_notes": ["action, pose, relationship, or camera notes in the detected script language that apply only to the current shot"]
}}
{retry_section}"""

    def _parse_decision(self, raw_text: str) -> tuple[Optional[ShotDecision], Optional[str]]:
        try:
            data = _extract_json_object(raw_text)
        except ValueError as exc:
            return None, str(exc)
        if not isinstance(data, dict):
            return None, "Model output is not a JSON object"
        raw_states = data.get("existing_element_states")
        raw_new = data.get("new_elements")
        raw_notes = data.get("shot_notes", [])
        if not isinstance(raw_states, list) or not isinstance(raw_new, list) or not isinstance(raw_notes, list):
            return None, "Top-level field has an invalid type"

        expected_ids = {element.id for element in self.registry.elements}
        seen: set[str] = set()
        states: list[ExistingElementState] = []
        for item in raw_states:
            if not isinstance(item, dict):
                return None, "existing_element_states contains a non-object item"
            element_id = item.get("id")
            state = item.get("state")
            reason = item.get("reason")
            if not isinstance(element_id, str) or not isinstance(state, str) or not isinstance(reason, str):
                return None, "Existing element state is missing a string field"
            if element_id in seen:
                return None, f"Existing element {element_id} is assigned a state more than once"
            if state not in ELEMENT_STATES:
                return None, f"Existing element {element_id} uses an invalid state {state}"
            seen.add(element_id)
            states.append(ExistingElementState(id=element_id, state=state, reason=reason.strip()))
        if seen != expected_ids:
                return None, f"Existing element states must fully cover the registry; missing={sorted(expected_ids - seen)}, unknown={sorted(seen - expected_ids)}"

        known_names = self.registry.names()
        seen_names: set[str] = set()
        new_elements: list[NewElement] = []
        for item in raw_new:
            if not isinstance(item, dict):
                return None, "new_elements contains a non-object item"
            name = item.get("name")
            element_type = item.get("type")
            notes = item.get("notes", "")
            if not isinstance(name, str) or not isinstance(element_type, str) or not isinstance(notes, str):
                return None, "New element is missing a string field"
            normalized = _normalize_name(name)
            if not normalized:
                return None, "New element name is empty"
            if element_type not in ELEMENT_TYPES:
                return None, f"New element {name} has an invalid type"
            if normalized in known_names or normalized in seen_names:
                return None, f"New element {name} duplicates an existing or current new element"
            seen_names.add(normalized)
            new_elements.append(NewElement(name=name.strip(), type=element_type, notes=notes.strip()))
        if not all(isinstance(note, str) for note in raw_notes):
            return None, "shot_notes must contain only strings"
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
                    "reference_intent": "Early reference anchor: stabilizes character identity, overall style, and visual tone; excluded from visual-element coverage retrieval.",
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
                    f"{EVALUATION_LABELS[state_by_id.get(element['id'], 'optional_or_uncertain')]} "
                    f"{element['name']}"
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
            parts.append("should reference: " + ", ".join(item["name"] for item in details["should_reference"]))
        if details["should_exclude"]:
            parts.append("should exclude: " + ", ".join(item["name"] for item in details["should_exclude"]))
        if details["optional_or_uncertain"]:
            parts.append("optional/uncertain: " + ", ".join(item["name"] for item in details["optional_or_uncertain"]))
        return "; ".join(parts) if parts else "no known tracked elements visible"

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
                            f"Reference image {ref['reference_index']} is from Scene {ref.get('source_scene_num')} / Shot {ref.get('source_shot_num')}:",
                            "This is an early reference anchor. Use it only to stabilize overall character identity, visual style, color tone, and long-range consistency.",
                            "Do not carry old scenes, old actions, or objects not required by the current shot into the new frame.",
                        ]
                    )
                    continue
                selection = ref["visual_element_selection"]
                holistic_description = str(ref.get("holistic_description") or "").strip()
                reference_lines.extend(
                    [
                        f"Reference image {ref['reference_index']} is from Scene {ref.get('source_scene_num')} / Shot {ref.get('source_shot_num')}:",
                    ]
                )
                if holistic_description:
                    reference_lines.append(holistic_description)
                reference_lines.extend(
                    [
                        "# Objective reference constraints",
                        "Elements to reference:",
                        _bullet_names(selection["should_reference"]),
                        "Elements not to introduce:",
                        _bullet_names(selection["should_exclude"]),
                    ]
                )
        else:
            reference_lines.append("No historical reference images are provided for this shot.")

        return "\n".join(
            [
                "[Full Script Context]",
                "The following shot-level descriptions are provided only to understand the narrative context, relationships, scene changes, and historical elements that should not appear in the current shot.",
                "Generate only the specified shot. Do not generate other shots or introduce characters, scenes, or objects from future shots early.",
                story_prompt_outline(self.story_script),
                "",
                "[Current Shot Generation Task]",
                f"Generate only Scene {shot.scene_num} / Shot {shot.shot_num}:",
                escape_prompt_line(shot.prompt),
                "When the input media includes a reference video, generate a continuation of it: start immediately after its final frame and naturally continue character actions, object motion, camera direction, and camera speed while preserving overall visual style, color tone, and visual elements.",
                "",
                "[Visual Element Plan for This Shot]",
                "Historical elements to reference and keep consistent:",
                _bullet_names(plan["should_reference"]),
                "Historical elements not to introduce:",
                _bullet_names(plan["should_exclude"]),
                "Optional or uncertain visual elements:",
                _bullet_names(plan["optional_or_uncertain"]),
                "New visual elements introduced in this shot:",
                _bullet_names(plan["new_elements"]),
                "",
                "[Reference Image Instructions]",
                "The following images are used only to preserve specified visual elements; they do not mean that the entire image should be reproduced.",
                "Image 1, Image 2, and so on refer to the order of the reference images below. A reference video may be present for shot continuity, but these instructions refer only to uploaded reference images.",
                *reference_lines,
                "",
                "[Overall Constraints]",
                "The current shot generation task has the highest priority.",
                "The full script is provided only for context. Do not generate story content from other shots. For elements not fully described in this shot, consult their descriptions and reference images in other shots.",
                "Reference only visual elements explicitly specified by the visual plan. Do not add elements from reference images that are unrelated to this shot.",
                "If a reference-image instruction conflicts with the current shot description, follow the current shot description and this shot's visual-element plan.",
                "Unless explicitly required by the script, do not add pull-backs, fade-outs, dimming, or other transitions at the end of the video.",
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
            raise VisualElementMemoryError(f"Keyframe does not exist: {frame_path}")
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
            f"VLM annotation parsing failed repeatedly for keyframe {frame_path}: {last_error}",
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
        story_context_rows = _shot_context_until(self.shots, shot)
        story_context = json.dumps(story_context_rows, ensure_ascii=False, indent=2)
        output_language = detect_story_language(item["video_prompt"] for item in story_context_rows)
        retry_section = f"\nThe previous output failed validation: {retry_feedback}\nFix only the format or coordinates.\n" if retry_feedback else ""
        prompt = f"""You are the keyframe visual-annotation assistant for a long-form video project. Inspect the input image, annotate only visible elements from the supplied visual-element registry, and provide a holistic keyframe description.

Strict rules:
1. Do not identify, name, or add elements outside the registry.
2. Output an element only when it is visible and recognizable in the image.
3. Use normalized xyxy coordinates from 0 to 1000: [x1, y1, x2, y2].
4. If a visible element has type scene, bbox_1000 must be exactly [0, 0, 1000, 1000].
5. For character and object, use a tight bounding box around the main body; output each element at most once per image.
6. For every character, output reference_quality as full, partial, or weak:
   - full: the character is complete and clear, with a clearly visible face; suitable as a full character reference;
   - partial: only part of the character is visible, or the face, clothing, or subject information is incomplete; suitable only as a partial reference;
   - weak: back view, distant view, severe occlusion, blur, or very small scale; not suitable as a full character reference.
   reference_quality may be omitted for non-character elements.
7. {output_language_instruction(output_language)} holistic_description must be one to three sentences describing the visible composition and, when justified by the story context, the frame's narrative role. Do not invent invisible details.
8. Return only a JSON object. Do not use Markdown, explanations, or code fences.

Story context through the current shot:
{story_context}

Current visual-element registry:
{element_text}

Return:
{{
  "holistic_description": "One to three sentences in the detected script language covering the main characters/objects/environment, composition, mood, action or camera state, and the frame's relation to the current story; use an empty string when it cannot be described reliably.",
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
            return None, "Annotation output is missing the visible_elements list"
        holistic_description = str(data.get("holistic_description") or "").strip()
        known = self.registry.by_id()
        seen: set[str] = set()
        annotated: list[AnnotatedElement] = []
        for item in data["visible_elements"]:
            if not isinstance(item, dict):
                return None, "visible_elements contains a non-object item"
            element_id = item.get("id")
            bbox = item.get("bbox_1000")
            if not isinstance(element_id, str) or element_id not in known:
                return None, f"Annotation references an unknown element {element_id}"
            if element_id in seen:
                return None, f"Element {element_id} is annotated more than once"
            if not _is_valid_bbox(bbox):
                return None, f"Element {element_id} has an invalid bbox_1000"
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
                return None, f"Element {element_id} has a bounding box with no valid area"
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
                label_for_element=lambda element: f"{element['name']} {element['type']}",
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
        raise VisualElementMemoryError("Missing SEEDANCE_API_KEY or ARK_API_KEY")
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
        raise VisualElementMemoryError(f"Seed2.1 Turbo call failed: {exc}") from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise VisualElementMemoryError(f"Model returned non-JSON HTTP {response.status_code}: {response.text[:500]}") from exc
    if response.status_code >= 400:
        raise VisualElementMemoryError(f"Model call failed HTTP {response.status_code}: {json.dumps(data, ensure_ascii=False)}")
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VisualElementMemoryError(f"Model response is missing text content: {json.dumps(data, ensure_ascii=False)}") from exc
    if not isinstance(content, str):
        raise VisualElementMemoryError("Model response content is not a string")
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
        raise ValueError("Model output contains no JSON object")
    try:
        value, _ = decoder.raw_decode(raw_text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model JSON could not be parsed: {exc.msg}") from exc
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
    raise VisualElementMemoryError(f"Unsupported image format: {path}")


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
        return "- none"
    return "\n".join(f"- {item.get('name', '')}" for item in items)
