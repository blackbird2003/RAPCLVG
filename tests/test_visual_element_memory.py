import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from storymem_seedance.models import RunConfig, ShotSpec
from storymem_seedance.visual_element_memory import (
    AnnotatedElement,
    ExistingElementState,
    FrameAnnotation,
    ShotDecision,
    VISUAL_ELEMENT_MAX_OUTPUT_TOKENS,
    VisualElement,
    VisualElementMemory,
    VisualElementMemoryError,
)


class VisualElementMemoryTests(unittest.TestCase):
    def test_visual_retrieval_precedes_sink_and_prompt_indices_match(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            sink_frame = root / "01_01_keyframe0.jpg"
            retrieved_frame = root / "01_02_keyframe0.jpg"
            for path in (sink_frame, retrieved_frame):
                Image.new("RGB", (32, 32), color=(120, 150, 180)).save(path)

            config = RunConfig(
                output_dir=tempdir,
                visual_element_sink_frame_count=1,
                visual_element_max_retrieved_frames=1,
            )
            story = {"scenes": [{"scene_num": 1, "video_prompts": ["first", "second"]}]}
            shot = ShotSpec(
                scene=story["scenes"][0],
                scene_num=1,
                shot_num=2,
                prompt="second",
                is_cut=True,
            )
            memory = VisualElementMemory(config, story, [shot])
            memory.registry.elements.append(
                VisualElement(
                    id="element-0001",
                    name="green-coated traveler",
                    type="character",
                    introduced_at="shot-0001",
                )
            )
            memory.annotations = [
                FrameAnnotation(
                    source_shot_id="shot-0001",
                    source_scene_num=1,
                    source_shot_num=1,
                    frame_path=str(sink_frame),
                    width=32,
                    height=32,
                    elements=[],
                ),
                FrameAnnotation(
                    source_shot_id="shot-0001",
                    source_scene_num=1,
                    source_shot_num=1,
                    frame_path=str(retrieved_frame),
                    width=32,
                    height=32,
                    elements=[
                        AnnotatedElement(
                            id="element-0001",
                            name="green-coated traveler",
                            type="character",
                            bbox_1000=[0, 0, 1000, 1000],
                            bbox_pixels=[0, 0, 32, 32],
                        )
                    ],
                ),
            ]
            decision = ShotDecision(
                existing_element_states=[
                    ExistingElementState(
                        id="element-0001",
                        state="should_reference",
                        reason="The same traveler appears again.",
                    )
                ],
                new_elements=[],
                shot_notes=[],
            )
            memory._decide_shot = types.MethodType(
                lambda self, shot_index, shot: (decision, [], [{"attempt": 1}]),
                memory,
            )

            result = memory.prepare_references_for_shot(1, shot)
            metadata = [item["metadata"] for item in result.selected_references]

            self.assertEqual(
                [item["roles"] for item in metadata],
                [["visual_element_memory"], ["visual_sink_memory"]],
            )
            self.assertEqual([item["reference_index"] for item in metadata], [1, 2])
            self.assertIn("参考图片 Image 1", result.prompt_context)
            self.assertIn("green-coated traveler", result.prompt_context)
            self.assertIn("参考图片 Image 2", result.prompt_context)
            self.assertIn("早期参考锚点", result.prompt_context)

    def test_visual_reference_budget_is_limited_to_nine_images(self):
        config = RunConfig(
            output_dir="/tmp",
            visual_element_sink_frame_count=6,
            visual_element_max_retrieved_frames=4,
        )
        with self.assertRaises(VisualElementMemoryError):
            VisualElementMemory(config, {"scenes": []}, [])

    def test_visual_plan_uses_128k_output_limit(self):
        config = RunConfig(output_dir="/tmp")
        self.assertEqual(config.visual_element_max_parse_retries, 2)
        shot = ShotSpec(scene={}, scene_num=1, shot_num=1, prompt="first", is_cut=True)
        memory = VisualElementMemory(config, {"scenes": []}, [shot])

        with patch(
            "storymem_seedance.visual_element_memory._call_ark_chat",
            return_value=('{"existing_element_states":[],"new_elements":[],"shot_notes":[]}', {"model": "fake"}),
        ) as call:
            memory.plan_visual_elements_for_shot(0, shot)

        self.assertEqual(call.call_args.kwargs["max_tokens"], VISUAL_ELEMENT_MAX_OUTPUT_TOKENS)
        self.assertEqual(VISUAL_ELEMENT_MAX_OUTPUT_TOKENS, 131_072)

    def test_vlm_annotation_records_prompt_and_attempts(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            frame = root / "frame.jpg"
            Image.new("RGB", (32, 32), color=(120, 150, 180)).save(frame)
            config = RunConfig(output_dir=tempdir)
            shot = ShotSpec(scene={}, scene_num=1, shot_num=1, prompt="first", is_cut=True)
            memory = VisualElementMemory(config, {"scenes": []}, [shot])

            with patch(
                "storymem_seedance.visual_element_memory._call_ark_chat",
                return_value=('{"holistic_description":"A calm empty frame introduces the opening mood.","visible_elements":[]}', {"model": "fake", "usage": {}}),
            ):
                annotations = memory.annotate_completed_shot(1, shot, [str(frame)])

            attempt = annotations[0]["attempts"][0]
            self.assertIn("关键帧视觉标注助手", attempt["metadata"]["prompt"])
            self.assertIn("截至当前 shot 的剧情上下文", attempt["metadata"]["prompt"])
            self.assertEqual(attempt["raw_response"], '{"holistic_description":"A calm empty frame introduces the opening mood.","visible_elements":[]}')
            self.assertIsNone(attempt["parse_error"])
            self.assertEqual(annotations[0]["holistic_description"], "A calm empty frame introduces the opening mood.")

    def test_vlm_annotation_keeps_character_reference_quality_and_defaults_to_full(self):
        config = RunConfig(output_dir="/tmp")
        shot = ShotSpec(scene={}, scene_num=1, shot_num=1, prompt="first", is_cut=True)
        memory = VisualElementMemory(config, {"scenes": []}, [shot])
        memory.registry.elements.extend(
            [
                VisualElement(
                    id="element-0001",
                    name="Pilot",
                    type="character",
                    introduced_at="shot-0001",
                ),
                VisualElement(
                    id="element-0002",
                    name="Console",
                    type="object",
                    introduced_at="shot-0001",
                ),
                VisualElement(
                    id="element-0003",
                    name="Navigator",
                    type="character",
                    introduced_at="shot-0001",
                ),
            ]
        )

        parsed, error = memory._parse_annotation(
            """
            {
              "holistic_description": "Pilot at a console.",
              "visible_elements": [
                {"id": "element-0001", "bbox_1000": [0, 0, 500, 1000], "reference_quality": "partial"},
                {"id": "element-0002", "bbox_1000": [500, 0, 1000, 500], "reference_quality": "weak"},
                {"id": "element-0003", "bbox_1000": [500, 500, 1000, 1000]}
              ]
            }
            """,
            width=100,
            height=100,
        )

        self.assertIsNone(error)
        self.assertIsNotNone(parsed)
        elements, holistic = parsed
        self.assertEqual(holistic, "Pilot at a console.")
        self.assertEqual([item.reference_quality for item in elements], ["partial", "full", "full"])

    def test_reference_quality_weights_score_but_only_full_covers_reference_elements(self):
        config = RunConfig(output_dir="/tmp")
        shot = ShotSpec(scene={}, scene_num=1, shot_num=1, prompt="second", is_cut=True)
        memory = VisualElementMemory(config, {"scenes": []}, [shot])
        decision = ShotDecision(
            existing_element_states=[
                ExistingElementState(
                    id="element-0001",
                    state="should_reference",
                    reason="The pilot must stay consistent.",
                )
            ],
            new_elements=[],
            shot_notes=[],
        )
        partial_annotation = FrameAnnotation(
            source_shot_id="shot-0001",
            source_scene_num=1,
            source_shot_num=1,
            frame_path="/tmp/partial.jpg",
            width=100,
            height=100,
            elements=[
                AnnotatedElement(
                    id="element-0001",
                    name="Pilot",
                    type="character",
                    bbox_1000=[0, 0, 1000, 1000],
                    bbox_pixels=[0, 0, 100, 100],
                    reference_quality="partial",
                )
            ],
        )
        full_annotation = FrameAnnotation(
            source_shot_id="shot-0001",
            source_scene_num=1,
            source_shot_num=1,
            frame_path="/tmp/full.jpg",
            width=100,
            height=100,
            elements=[
                AnnotatedElement(
                    id="element-0001",
                    name="Pilot",
                    type="character",
                    bbox_1000=[0, 0, 1000, 1000],
                    bbox_pixels=[0, 0, 100, 100],
                )
            ],
        )

        partial_score, partial_details = memory._score_frame(partial_annotation, decision, covered=set())
        full_score, full_details = memory._score_frame(full_annotation, decision, covered=set())

        self.assertAlmostEqual(partial_score, 0.6)
        self.assertAlmostEqual(full_score, 3.0)
        self.assertEqual(partial_details["newly_covered_element_ids"], [])
        self.assertEqual(full_details["newly_covered_element_ids"], ["element-0001"])
        self.assertEqual(partial_details["should_reference"][0]["reference_quality"], "partial")

    def test_greedy_selection_keeps_negative_scored_frame_when_it_is_best_available(self):
        config = RunConfig(output_dir="/tmp", visual_element_weight_exclude=-10.0)
        shot = ShotSpec(scene={}, scene_num=1, shot_num=1, prompt="second", is_cut=True)
        memory = VisualElementMemory(config, {"scenes": []}, [shot])
        decision = ShotDecision(
            existing_element_states=[
                ExistingElementState(id="element-0001", state="should_reference", reason="Need the pilot."),
                ExistingElementState(id="element-0002", state="should_exclude", reason="Exclude the old room."),
            ],
            new_elements=[],
            shot_notes=[],
        )
        annotation = FrameAnnotation(
            source_shot_id="shot-0001",
            source_scene_num=1,
            source_shot_num=1,
            frame_path="/tmp/negative.jpg",
            width=100,
            height=100,
            elements=[
                AnnotatedElement(
                    id="element-0001",
                    name="Pilot",
                    type="character",
                    bbox_1000=[0, 0, 1000, 1000],
                    bbox_pixels=[0, 0, 100, 100],
                ),
                AnnotatedElement(
                    id="element-0002",
                    name="Old room",
                    type="scene",
                    bbox_1000=[0, 0, 1000, 1000],
                    bbox_pixels=[0, 0, 100, 100],
                ),
            ],
        )
        score, _details = memory._score_frame(annotation, decision, covered=set())
        memory.annotations = [annotation]

        selected = memory._select_reference_frames(decision)

        self.assertLess(score, 0)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][0].frame_path, "/tmp/negative.jpg")

    def test_greedy_selection_does_not_stop_after_required_elements_are_covered(self):
        config = RunConfig(output_dir="/tmp", visual_element_max_retrieved_frames=2)
        shot = ShotSpec(scene={}, scene_num=1, shot_num=1, prompt="second", is_cut=True)
        memory = VisualElementMemory(config, {"scenes": []}, [shot])
        decision = ShotDecision(
            existing_element_states=[
                ExistingElementState(id="element-0001", state="should_reference", reason="Need the pilot."),
                ExistingElementState(id="element-0002", state="optional_or_uncertain", reason="Style cue."),
            ],
            new_elements=[],
            shot_notes=[],
        )
        memory.annotations = [
            FrameAnnotation(
                source_shot_id="shot-0001",
                source_scene_num=1,
                source_shot_num=1,
                frame_path="/tmp/pilot.jpg",
                width=100,
                height=100,
                elements=[
                    AnnotatedElement(
                        id="element-0001",
                        name="Pilot",
                        type="character",
                        bbox_1000=[0, 0, 1000, 1000],
                        bbox_pixels=[0, 0, 100, 100],
                    )
                ],
            ),
            FrameAnnotation(
                source_shot_id="shot-0001",
                source_scene_num=1,
                source_shot_num=1,
                frame_path="/tmp/style.jpg",
                width=100,
                height=100,
                elements=[
                    AnnotatedElement(
                        id="element-0002",
                        name="Style cue",
                        type="scene",
                        bbox_1000=[0, 0, 1000, 1000],
                        bbox_pixels=[0, 0, 100, 100],
                    )
                ],
            ),
        ]

        selected = memory._select_reference_frames(decision)

        self.assertEqual([item[0].frame_path for item in selected], ["/tmp/pilot.jpg", "/tmp/style.jpg"])


if __name__ == "__main__":
    unittest.main()
