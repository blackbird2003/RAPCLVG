from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from seedance_client import SeedanceError
from storymem_seedance.visual_element_memory import VisualElementMemoryError
from storymem_seedance.visual_plan_reflection import VisualPlanReflectionResult
from videogen_notebook.app import _element_names, _step_elapsed, create_app
from videogen_notebook.gpu_lock import KeyframeGpuLock
from videogen_notebook.jobs import BackgroundJobManager
from videogen_notebook.json_store import read_json, write_json_atomic
from videogen_notebook.runner import (
    FORCE_ANIMATION_PROMPT_PREFIX,
    FakeExecutionBackend,
    FakeNotebookRunner,
    NotebookVisualElementMemoryAdapter,
    NotebookRunner,
    PlaceholderVideoPostprocessor,
    PlaceholderVisualMemoryPlanner,
    ReferenceSelectionContext,
    VideoPostprocessContext,
    VideoPostprocessResult,
    VisualMemoryPlanResult,
    _add_reference_guidance,
    _media_debug_for_submission,
    _prompt_for_generation_mode,
    STYLE_REFERENCE_GUIDANCE,
    _run_config_for_bundle,
    _visual_reference_prompt_context,
    create_runner,
)
from videogen_notebook.project_store import ProjectStore


class FakeSubmitClient:
    def __init__(self) -> None:
        self.created_calls = []
        self.waited_task_ids = []
        self.downloads = []

    def create_task(self, content, **kwargs):
        self.created_calls.append({"content": content, "options": kwargs})
        return {"id": f"task-{len(self.created_calls)}"}

    def wait_task(self, task_id, **kwargs):
        self.waited_task_ids.append(task_id)
        return {
            "id": task_id,
            "status": "succeeded",
            "content": {"video_url": f"https://example.test/{task_id}.mp4"},
            "usage": {"total_tokens": 12},
        }

    def get_task(self, task_id):
        return {
            "id": task_id,
            "status": "succeeded",
            "content": {"last_frame_url": f"https://example.test/{task_id}-last-frame.jpg"},
        }

    def download_video(self, url, output_path):
        self.downloads.append({"url": url, "output_path": output_path})
        Path(output_path).write_bytes(b"submitted video")


class SensitiveImageOnceFakeSubmitClient(FakeSubmitClient):
    def __init__(self, rejected_content_index: int) -> None:
        super().__init__()
        self.rejected_content_index = rejected_content_index
        self.rejected = False

    def create_task(self, content, **kwargs):
        self.created_calls.append({"content": content, "options": kwargs})
        should_reject = (
            not self.rejected
            and self.rejected_content_index < len(content)
            and content[self.rejected_content_index].get("type") == "image_url"
        )
        if should_reject:
            self.rejected = True
            payload = {
                "error": {
                    "code": "InputImageSensitiveContentDetected.PolicyViolation",
                    "message": (
                        "The request failed because the input image "
                        f"'content[{self.rejected_content_index}]' may be related to copyright restrictions."
                    ),
                    "param": f"content[{self.rejected_content_index}]",
                    "type": "BadRequest",
                }
            }
            raise SeedanceError(f"create_task returned HTTP 400: {json.dumps(payload)}")
        return {"id": f"task-{len(self.created_calls)}"}


class FakeMediaPreflightResponse:
    status_code = 200
    url = "https://cdn.example.test/tail.mp4"
    headers = {"content-type": "video/mp4", "content-length": "12345"}
    history = []

    def __init__(self) -> None:
        self.closed = False

    def iter_content(self, chunk_size):
        yield b"\x00\x00\x00\x18ftypisom"

    def close(self):
        self.closed = True


class FakePostprocessor:
    name = "fake_postprocessor"

    def __init__(self) -> None:
        self.calls = []

    def process(self, context):
        self.calls.append(context)
        project_dir = context.video_path.parents[2]
        asset_id = f"img_{context.shot['shot_id']}_{context.attempt_id}_fakepp"
        image_path = project_dir / "assets" / "images" / f"{asset_id}.jpg"
        image_path.write_bytes(b"fake postprocessed image")
        return VideoPostprocessResult(
            assets={
                asset_id: {
                    "kind": "image",
                    "path": str(image_path.relative_to(project_dir)),
                    "created_at": "test-time",
                    "source": {
                        "type": "fake_postprocessed_keyframe",
                        "shot_id": context.shot["shot_id"],
                        "attempt_id": context.attempt_id,
                    },
                    "metadata": {"postprocessor": self.name},
                }
            },
            produced_visual_memory=[
                {
                    "asset_id": asset_id,
                    "rank": 1,
                    "visible_elements": ["fake-element"],
                    "annotation": "Fake postprocessor annotation.",
                    "active_in_memory_pool": True,
                }
            ],
            postprocess={"postprocessor": self.name, "fake_postprocess": True},
            logs=[{"time": "test-time", "step": "fake_postprocess", "message": "fake postprocessor ran"}],
        )


class FakeVisualPlanner:
    name = "fake_visual_planner"

    def __init__(self) -> None:
        self.calls = []

    def plan(self, context):
        self.calls.append(context)
        return VisualMemoryPlanResult(
            visual_element_status=[
                {
                    "element_id": "element-fake-planner",
                    "name": "Planner Element",
                    "type": "object",
                    "introduced_at": context.shot["shot_id"],
                    "notes": "Planner injected row.",
                    "status": "should_reference",
                    "reason": "Planner test reason.",
                }
            ],
            selected_references=[],
            prompt_context="FULL PROMPT FROM FAKE VISUAL PLANNER",
            logs=[{"time": "test-time", "step": "fake_visual_plan", "message": "fake planner ran"}],
        )


class FailingVisualPlanner:
    name = "failing_visual_planner"

    def plan_visual_elements(self, context):
        raise VisualElementMemoryError(
            "视觉元素决策连续解析失败: 模型 JSON 无法解析",
            details={
                "failed_stage": "visual_elements_plan",
                "last_error": "Unterminated string starting at",
                "llm_attempts": [
                    {
                        "attempt": 1,
                        "metadata": {"prompt": "请为老公和咖啡店杯型生成视觉元素计划"},
                        "raw_response": '{"annotation":"Visible: 老公站在咖啡店柜台前',
                        "parse_error": "Unterminated string starting at",
                    }
                ],
            },
        )


class FlakyKeyframeBackend(FakeExecutionBackend):
    def __init__(self, store: ProjectStore, failures_before_success: int) -> None:
        super().__init__(store)
        self.failures_before_success = failures_before_success
        self.maintain_calls = 0

    def maintain_keyframes(self, context):
        self.maintain_calls += 1
        if self.maintain_calls <= self.failures_before_success:
            raise RuntimeError(f"transient keyframe failure {self.maintain_calls}")
        return super().maintain_keyframes(context)


class FlakyAlgorithmStepBackend(FakeExecutionBackend):
    def __init__(self, store: ProjectStore, failures_before_success: dict[str, int]) -> None:
        super().__init__(store)
        self.failures_before_success = dict(failures_before_success)
        self.calls = {"visual_plan": 0, "reference_selection": 0, "seedance_generation": 0}

    def _maybe_fail(self, step: str) -> None:
        self.calls[step] += 1
        if self.calls[step] <= self.failures_before_success.get(step, 0):
            raise RuntimeError(f"transient {step} failure {self.calls[step]}")

    def plan_visual_elements(self, context):
        self._maybe_fail("visual_plan")
        return super().plan_visual_elements(context)

    def select_historical_references(self, context):
        self._maybe_fail("reference_selection")
        return super().select_historical_references(context)

    def generate_seedance_video(self, context):
        self._maybe_fail("seedance_generation")
        return super().generate_seedance_video(context)


class BlockingKeyframeBackend(FakeExecutionBackend):
    def __init__(self, store: ProjectStore) -> None:
        super().__init__(store)
        self.entered: dict[str, threading.Event] = {}
        self.release: dict[str, threading.Event] = {}
        self.order: list[str] = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def prepare_project(self, project_id: str) -> None:
        self.entered[project_id] = threading.Event()
        self.release[project_id] = threading.Event()

    def maintain_keyframes(self, context):
        project_id = context.project_id
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.order.append(project_id)
        self.entered[project_id].set()
        self.release[project_id].wait(timeout=5)
        try:
            return super().maintain_keyframes(context)
        finally:
            with self.lock:
                self.active -= 1


class BlockingRunner:
    def __init__(self) -> None:
        self.started: dict[str, threading.Event] = {}
        self.release = threading.Event()

    def run_all(self, project_id: str, *, apply_review_delay: bool = False):
        self.started.setdefault(project_id, threading.Event()).set()
        self.release.wait(timeout=5)

    def run_shot(self, project_id: str, shot_id: str, *, apply_review_delay: bool = False):
        self.run_all(project_id, apply_review_delay=apply_review_delay)

    def run_step(self, project_id: str, shot_id: str, step: str):
        self.run_all(project_id)

    def reflect_visual_plan(self, project_id: str, shot_id: str):
        self.run_all(project_id)

    def rerun_assembly(self, project_id: str):
        self.run_all(project_id)

    def interrupt_project(self, project_id: str):
        self.release.set()


class VideogenNotebookStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name)
        self.store = ProjectStore(self.workspace)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_import_story_creates_self_contained_bundle(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        project_dir = self.store.project_dir(project_id)

        self.assertTrue((project_dir / "project.json").exists())
        self.assertTrue((project_dir / "settings.json").exists())
        self.assertTrue((project_dir / "assets" / "manifest.json").exists())
        self.assertTrue((project_dir / "memory" / "visual_state.json").exists())
        self.assertEqual(project["project"]["pipeline"], "visual_element_memory_v1")
        self.assertEqual(project["settings"]["generation"]["default_duration_seconds"], -1)
        self.assertEqual(project["settings"]["generation"]["algorithm_step_max_attempts"], 5)
        self.assertEqual(project["settings"]["generation"]["smooth_reference_seconds"], 2.0)
        self.assertEqual(project["settings"]["generation"]["force_animation_style"], True)
        self.assertEqual(project["settings"]["seedance"]["resolution"], "720p")
        self.assertEqual(project["settings"]["seedance"]["ratio"], "16:9")
        self.assertEqual(project["settings"]["visual_element_memory"]["sink_frame_count"], 0)
        self.assertEqual(project["settings"]["visual_element_memory"]["max_retrieved_frames"], 4)
        self.assertEqual(project["settings"]["visual_element_memory"]["selection_mode"], "greedy_coverage")
        self.assertEqual(project["settings"]["visual_element_memory"]["scoring"]["exclude_weight"], -0.1)
        self.assertEqual(project["settings"]["visual_element_memory"]["scoring"]["quality_partial_weight"], 0.2)
        self.assertEqual(project["settings"]["evaluation"]["auto_submit_eval"], False)
        self.assertEqual(project["shots"][0]["inputs"]["generation_mode"], "default")
        self.assertEqual(project["shots"][1]["inputs"]["generation_mode"], "smooth")

    def test_list_projects_sorts_by_created_at_descending(self):
        old_project = self.store.import_story(_story(), name="Older but recently updated")
        new_project = self.store.import_story(_story(), name="Newer but stale")
        old_path = self.store.project_dir(old_project["project"]["project_id"]) / "project.json"
        new_path = self.store.project_dir(new_project["project"]["project_id"]) / "project.json"
        old_json = read_json(old_path)
        new_json = read_json(new_path)
        old_json["created_at"] = "2026-07-01T00:00:00Z"
        old_json["updated_at"] = "2026-07-14T00:00:00Z"
        new_json["created_at"] = "2026-07-13T00:00:00Z"
        new_json["updated_at"] = "2026-07-13T00:00:00Z"
        write_json_atomic(old_path, old_json)
        write_json_atomic(new_path, new_json)

        names = [item["name"] for item in self.store.list_projects()]

        self.assertLess(names.index("Newer but stale"), names.index("Older but recently updated"))

    def test_import_story_invalid_durations_fall_back_to_auto(self):
        story = _story()
        story["scenes"][0]["durations"] = [3, 16, "bad"]

        project = self.store.import_story(story)

        self.assertEqual([shot["inputs"]["duration_seconds"] for shot in project["shots"]], [-1, -1, -1])

    def test_import_predefined_references_copies_assets_and_exports_design(self):
        source = self.workspace / "role-ref.jpg"
        source.write_bytes(b"fake role reference")
        story = _story()
        story["scenes"][0]["predefined_references"] = [
            [{"image_path": str(source), "label": "Prince", "guidance": "Use as character identity."}],
            [],
            [],
        ]

        project = self.store.import_story(story)
        project_id = project["project"]["project_id"]
        bundle = self.store.get_project(project_id)
        reference = bundle["shots"][0]["inputs"]["predefined_references"][0]

        self.assertEqual(reference["label"], "Prince")
        self.assertEqual(reference["guidance"], "Use as character identity.")
        self.assertTrue(reference["media_url"].startswith(f"/projects/{project_id}/assets/"))
        self.assertTrue((self.store.project_dir(project_id) / reference["image_path"]).is_file())
        self.assertIn(reference["asset_id"], bundle["assets"])

        exported = self.store.export_story_design(project_id)
        exported_ref = exported["scenes"][0]["predefined_references"][0][0]
        self.assertEqual(exported_ref["label"], "Prince")
        self.assertTrue(Path(exported_ref["image_path"]).is_absolute())

        saved = self.store.save_project(
            project_id,
            shot_updates=[
                {
                    "shot_id": "0001",
                    "video_prompt": "Prompt 1",
                    "is_cut": True,
                    "generation_mode": "default",
                    "duration_seconds": 8,
                    "predefined_references": [
                        {
                            "id": reference["id"],
                            "label": "Edited Prince",
                            "guidance": "Edited character identity guidance.",
                        }
                    ],
                }
            ],
        )
        edited = saved["shots"][0]["inputs"]["predefined_references"][0]
        self.assertEqual(edited["label"], "Edited Prince")
        self.assertEqual(edited["guidance"], "Edited character identity guidance.")

        removed = self.store.save_project(
            project_id,
            shot_updates=[
                {
                    "shot_id": "0001",
                    "video_prompt": "Prompt 1",
                    "is_cut": True,
                    "generation_mode": "default",
                    "duration_seconds": 8,
                    "predefined_references": [],
                }
            ],
        )
        self.assertEqual(removed["shots"][0]["inputs"]["predefined_references"], [])

    def test_completed_shot_save_ignores_textarea_crlf_normalization(self):
        story = {
            "story_name": "Multiline",
            "scenes": [{"scene_num": 1, "video_prompts": ["Line one\nLine two"], "cut": [True]}],
        }
        project = self.store.import_story(story)
        project_id = project["project"]["project_id"]
        shot = project["shots"][0]
        shot["state"]["status"] = "completed"
        self.store.write_shot(project_id, shot)

        saved = self.store.save_project(
            project_id,
            shot_updates=[
                {
                    "shot_id": "0001",
                    "video_prompt": "Line one\r\nLine two",
                    "is_cut": True,
                    "generation_mode": "default",
                    "duration_seconds": -1,
                }
            ],
        )

        self.assertEqual(saved["shots"][0]["inputs"]["video_prompt"], "Line one\nLine two")

    def test_background_jobs_allow_different_projects_but_reject_duplicate_project(self):
        runner = BlockingRunner()
        jobs = BackgroundJobManager(runner, max_workers=2)  # type: ignore[arg-type]
        try:
            jobs.start_run_all("project-a")
            self.assertTrue(runner.started.setdefault("project-a", threading.Event()).wait(timeout=2))

            jobs.start_run_all("project-b")
            self.assertTrue(runner.started.setdefault("project-b", threading.Event()).wait(timeout=2))

            with self.assertRaisesRegex(Exception, "already running"):
                jobs.start_run_all("project-a")
        finally:
            runner.release.set()
            jobs.executor.shutdown(wait=True)

    def test_visual_memory_selection_mode_and_scoring_settings_feed_run_config(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        saved = self.store.save_project(
            project_id,
            settings_update={
                "visual_element_memory": {
                    "selection_mode": "static_top_k",
                    "scoring": {
                        "character_weight": 4.0,
                        "exclude_weight": -0.25,
                        "quality_partial_weight": 0.3,
                    },
                },
                "generation": {"smooth_reference_seconds": 3.5},
                "seedance": {"resolution": "1080p", "ratio": "9:16"},
            },
        )

        config = _run_config_for_bundle(saved, self.store.project_dir(project_id) / "memory")

        self.assertEqual(saved["settings"]["visual_element_memory"]["selection_mode"], "static_top_k")
        self.assertEqual(config.visual_element_selection_mode, "static_top_k")
        self.assertEqual(config.visual_element_weight_character, 4.0)
        self.assertEqual(config.visual_element_weight_exclude, -0.25)
        self.assertEqual(config.visual_element_weight_quality_partial, 0.3)
        self.assertEqual(config.smooth_reference_seconds, 3.5)
        self.assertEqual(config.resolution, "1080p")
        self.assertEqual(config.ratio, "9:16")

        no_refs = self.store.save_project(
            project_id,
            settings_update={"visual_element_memory": {"selection_mode": "none"}},
        )
        no_refs_config = _run_config_for_bundle(no_refs, self.store.project_dir(project_id) / "memory")
        self.assertEqual(no_refs["settings"]["visual_element_memory"]["selection_mode"], "none")
        self.assertEqual(no_refs_config.visual_element_selection_mode, "none")

        naive = self.store.save_project(
            project_id,
            settings_update={"visual_element_memory": {"selection_mode": "naive_top_k"}},
        )
        naive_config = _run_config_for_bundle(naive, self.store.project_dir(project_id) / "memory")
        self.assertEqual(naive["settings"]["visual_element_memory"]["selection_mode"], "naive_top_k")
        self.assertEqual(naive_config.visual_element_selection_mode, "naive_top_k")

        storymem = self.store.save_project(
            project_id,
            settings_update={"visual_element_memory": {"selection_mode": "storymem_memory"}},
        )
        storymem_config = _run_config_for_bundle(storymem, self.store.project_dir(project_id) / "memory")
        self.assertEqual(storymem["settings"]["visual_element_memory"]["selection_mode"], "storymem_memory")
        self.assertEqual(storymem_config.visual_element_selection_mode, "storymem_memory")

    def test_no_history_reference_mode_skips_reference_selection_backend(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        self.store.save_project(
            project_id,
            settings_update={"visual_element_memory": {"selection_mode": "none"}},
        )
        backend = FlakyAlgorithmStepBackend(self.store, {"reference_selection": 99})
        runner = NotebookRunner(self.store, backend)

        runner.run_step(project_id, "0001", "visual_plan")
        runner.run_step(project_id, "0001", "reference_selection")

        bundle = self.store.get_project(project_id)
        first = bundle["shots"][0]
        attempt_dir = self.store.attempt_dir(project_id, "0001", first["state"]["current_attempt_id"])
        selected = read_json(attempt_dir / "reference_selection" / "selected_references.json")
        details = read_json(attempt_dir / "reference_selection" / "details.json")
        logs = [log["message"] for log in first["attempt"]["logs"]]

        self.assertEqual(backend.calls["reference_selection"], 0)
        self.assertEqual(first["state"]["status"], "reference_selection_completed")
        self.assertEqual(selected, [])
        self.assertEqual(details["selection_mode"], "none")
        self.assertTrue(details["skipped"])
        self.assertTrue(any("Reference Selection skipped" in item for item in logs))

    def test_naive_top_k_skips_visual_planner_and_selects_prior_keyframes(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        FakeNotebookRunner(self.store).run_shot(project_id, "0001")
        self.store.save_project(
            project_id,
            settings_update={"visual_element_memory": {"selection_mode": "naive_top_k", "max_retrieved_frames": 1}},
        )
        backend = FlakyAlgorithmStepBackend(self.store, {"visual_plan": 99, "reference_selection": 99})
        runner = NotebookRunner(self.store, backend)

        with patch("videogen_notebook.runner._clip_text_embedding", return_value=[1.0, 0.0]), patch(
            "videogen_notebook.runner._clip_image_embedding_for_asset",
            return_value=[0.75, 0.25],
        ) as image_embedding:
            runner.run_step(project_id, "0002", "visual_plan")
            runner.run_step(project_id, "0002", "reference_selection")

        bundle = self.store.get_project(project_id)
        second = bundle["shots"][1]
        attempt_dir = self.store.attempt_dir(project_id, "0002", second["state"]["current_attempt_id"])
        visual_status = read_json(attempt_dir / "visual_plan" / "visual_element_status.json")
        visual_details = read_json(attempt_dir / "visual_plan" / "details.json")
        selected = read_json(attempt_dir / "reference_selection" / "selected_references.json")
        reference_details = read_json(attempt_dir / "reference_selection" / "details.json")

        self.assertEqual(backend.calls["visual_plan"], 0)
        self.assertEqual(backend.calls["reference_selection"], 0)
        self.assertEqual(visual_status, [])
        self.assertEqual(visual_details["selection_mode"], "naive_top_k")
        self.assertEqual(reference_details["selection_mode"], "naive_top_k")
        self.assertEqual(reference_details["score"]["method"], "clip_text_image")
        self.assertEqual(image_embedding.call_count, 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["source_shot_id"], "0001")
        self.assertEqual(selected[0]["covered_elements"], [])
        self.assertEqual(selected[0]["holistic_description"], "")

    def test_storymem_memory_mode_skips_visual_planner_and_uses_sink_recent_memory(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        fake_runner = FakeNotebookRunner(self.store)
        fake_runner.run_shot(project_id, "0001")
        fake_runner.run_shot(project_id, "0002")
        self.store.save_project(
            project_id,
            settings_update={"visual_element_memory": {"selection_mode": "storymem_memory"}},
        )
        backend = FlakyAlgorithmStepBackend(self.store, {"visual_plan": 99, "reference_selection": 99})
        runner = NotebookRunner(self.store, backend)

        runner.run_step(project_id, "0003", "visual_plan")
        runner.run_step(project_id, "0003", "reference_selection")

        bundle = self.store.get_project(project_id)
        third = bundle["shots"][2]
        attempt_dir = self.store.attempt_dir(project_id, "0003", third["state"]["current_attempt_id"])
        visual_status = read_json(attempt_dir / "visual_plan" / "visual_element_status.json")
        visual_details = read_json(attempt_dir / "visual_plan" / "details.json")
        selected = read_json(attempt_dir / "reference_selection" / "selected_references.json")
        reference_details = read_json(attempt_dir / "reference_selection" / "details.json")

        self.assertEqual(backend.calls["visual_plan"], 0)
        self.assertEqual(backend.calls["reference_selection"], 0)
        self.assertEqual(visual_status, [])
        self.assertEqual(visual_details["selection_mode"], "storymem_memory")
        self.assertEqual(reference_details["selection_mode"], "storymem_memory")
        self.assertEqual(reference_details["storymem_memory"]["max_memory_size"], 10)
        self.assertEqual(reference_details["storymem_memory"]["fix"], 3)
        self.assertEqual([item["source_shot_id"] for item in selected], ["0001", "0002"])
        self.assertEqual(selected[0]["roles"], ["early_sink_memory"])
        self.assertEqual(selected[1]["roles"], ["early_sink_memory"])

    def test_naive_top_k_keyframe_maintaining_skips_vlm_annotation(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        self.store.save_project(
            project_id,
            settings_update={"visual_element_memory": {"selection_mode": "naive_top_k"}},
        )
        bundle = self.store.get_project(project_id)
        project_dir = self.store.project_dir(project_id)
        attempt_dir = self.store.attempt_dir(project_id, "0001", "a001")
        attempt_dir.mkdir(parents=True, exist_ok=True)
        video_path = project_dir / "assets" / "videos" / "vid_0001_a001.mp4"
        video_path.write_bytes(b"fake video")
        self.store.update_asset_manifest(
            project_id,
            {
                "vid_0001_a001": {
                    "kind": "video",
                    "path": "assets/videos/vid_0001_a001.mp4",
                    "created_at": "test-time",
                    "source": {"type": "fake_seedance_output", "shot_id": "0001", "attempt_id": "a001"},
                    "metadata": {},
                }
            },
        )
        extracted_keyframe = self.workspace / "source-keyframe.jpg"
        extracted_last = self.workspace / "source-last.jpg"
        extracted_keyframe.write_bytes(b"fake keyframe")
        extracted_last.write_bytes(b"fake last frame")

        class ExplodingMemory:
            records = []

            def annotate_completed_shot(self, **kwargs):
                raise AssertionError("VLM annotation should be skipped")

        adapter = NotebookVisualElementMemoryAdapter(self.store)
        context = VideoPostprocessContext(
            project_id=project_id,
            bundle=self.store.get_project(project_id),
            shot=bundle["shots"][0],
            attempt_id="a001",
            attempt_dir=attempt_dir,
            video_asset_id="vid_0001_a001",
            video_path=video_path,
            visual_element_status=[],
            dry_run=False,
        )
        with patch("storymem_seedance.executor.extract_shot_memory") as extract, patch.object(
            adapter,
            "_memory_for_current_postprocess",
            return_value=ExplodingMemory(),
        ):
            extract.return_value = {
                "keyframe_paths": [str(extracted_keyframe)],
                "last_frame_path": str(extracted_last),
            }
            result = adapter.process(context)

        self.assertEqual(len(result.produced_visual_memory), 1)
        self.assertEqual(result.produced_visual_memory[0]["visible_elements"], [])
        self.assertEqual(result.produced_visual_memory[0]["annotation"], "")
        self.assertNotIn("frame_annotation", result.produced_visual_memory[0])
        self.assertTrue(result.postprocess["vlm_annotation_skipped"])
        self.assertTrue(read_json(attempt_dir / "visual_element_details.json")["naive_top_k"]["vlm_annotation_skipped"])

    def test_storymem_memory_keyframe_maintaining_uses_original_profile_and_skips_vlm_annotation(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        self.store.save_project(
            project_id,
            settings_update={"visual_element_memory": {"selection_mode": "storymem_memory"}},
        )
        bundle = self.store.get_project(project_id)
        project_dir = self.store.project_dir(project_id)
        attempt_dir = self.store.attempt_dir(project_id, "0001", "a001")
        attempt_dir.mkdir(parents=True, exist_ok=True)
        video_path = project_dir / "assets" / "videos" / "vid_0001_a001.mp4"
        video_path.write_bytes(b"fake video")
        self.store.update_asset_manifest(
            project_id,
            {
                "vid_0001_a001": {
                    "kind": "video",
                    "path": "assets/videos/vid_0001_a001.mp4",
                    "created_at": "test-time",
                    "source": {"type": "fake_seedance_output", "shot_id": "0001", "attempt_id": "a001"},
                    "metadata": {},
                }
            },
        )
        extracted_keyframe = self.workspace / "source-keyframe.jpg"
        extracted_last = self.workspace / "source-last.jpg"
        extracted_keyframe.write_bytes(b"fake keyframe")
        extracted_last.write_bytes(b"fake last frame")

        class ExplodingMemory:
            records = []

            def annotate_completed_shot(self, **kwargs):
                raise AssertionError("VLM annotation should be skipped")

        adapter = NotebookVisualElementMemoryAdapter(self.store)
        context = VideoPostprocessContext(
            project_id=project_id,
            bundle=self.store.get_project(project_id),
            shot=bundle["shots"][0],
            attempt_id="a001",
            attempt_dir=attempt_dir,
            video_asset_id="vid_0001_a001",
            video_path=video_path,
            visual_element_status=[],
            dry_run=False,
        )
        with patch("storymem_seedance.executor.extract_shot_memory") as extract, patch.object(
            adapter,
            "_memory_for_current_postprocess",
            return_value=ExplodingMemory(),
        ):
            extract.return_value = {
                "keyframe_paths": [str(extracted_keyframe)],
                "last_frame_path": str(extracted_last),
            }
            result = adapter.process(context)

        self.assertEqual(extract.call_args.kwargs["keyframe_profile"], "storymem_original")
        self.assertEqual(result.produced_visual_memory[0]["visible_elements"], [])
        self.assertTrue(result.postprocess["vlm_annotation_skipped"])
        self.assertEqual(result.postprocess["keyframe_profile"], "storymem_original")
        self.assertTrue(read_json(attempt_dir / "visual_element_details.json")["storymem_memory"]["vlm_annotation_skipped"])

    def test_storymem_keyframe_dedup_compares_selected_memory_bank_only(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        self.store.save_project(
            project_id,
            settings_update={"visual_element_memory": {"selection_mode": "storymem_memory"}},
        )
        project_dir = self.store.project_dir(project_id)
        image_dir = project_dir / "assets" / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        assets = {}
        produced = []
        for index in range(12):
            asset_id = f"img_0001_a001_{index + 1:03d}"
            image_path = image_dir / f"{asset_id}.jpg"
            image_path.write_bytes(f"history {index}".encode("utf-8"))
            assets[asset_id] = {
                "kind": "image",
                "path": f"assets/images/{asset_id}.jpg",
                "created_at": "test-time",
                "source": {"type": "seedance_produced_keyframe", "shot_id": "0001", "attempt_id": "a001"},
                "metadata": {},
            }
            produced.append(
                {
                    "asset_id": asset_id,
                    "source_shot_id": "0001",
                    "attempt_id": "a001",
                    "rank": index + 1,
                    "active_in_memory_pool": True,
                }
            )
        self.store.update_asset_manifest(project_id, assets)
        first = self.store.get_project(project_id)["shots"][0]
        first["state"]["status"] = "completed"
        first["state"]["current_attempt_id"] = "a001"
        self.store.write_shot(project_id, first)
        first_attempt_dir = self.store.attempt_dir(project_id, "0001", "a001")
        write_json_atomic(
            first_attempt_dir / "attempt.json",
            {
                "attempt_id": "a001",
                "shot_id": "0001",
                "status": "completed",
                "selected_references": [],
                "produced_visual_memory": produced,
            },
        )
        write_json_atomic(first_attempt_dir / "produced_visual_memory.json", produced)

        selected_asset_ids = [
            "img_0001_a001_001",
            "img_0001_a001_002",
            "img_0001_a001_003",
            "img_0001_a001_007",
            "img_0001_a001_008",
            "img_0001_a001_009",
            "img_0001_a001_010",
            "img_0001_a001_011",
            "img_0001_a001_012",
        ]
        selected = [
            {
                "reference_id": f"storymem-0001-a001-{asset_id}",
                "asset_id": asset_id,
                "source_shot_id": "0001",
                "roles": ["early_sink_memory"] if index < 3 else ["recent_window_memory"],
            }
            for index, asset_id in enumerate(selected_asset_ids)
        ]
        attempt_dir = self.store.attempt_dir(project_id, "0002", "a001")
        write_json_atomic(attempt_dir / "reference_selection" / "selected_references.json", selected)
        video_path = project_dir / "assets" / "videos" / "vid_0002_a001.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"fake video")
        extracted_keyframe = self.workspace / "source-storymem-keyframe.jpg"
        extracted_last = self.workspace / "source-storymem-last.jpg"
        extracted_keyframe.write_bytes(b"fake keyframe")
        extracted_last.write_bytes(b"fake last frame")

        class ExplodingMemory:
            records = []

            def annotate_completed_shot(self, **kwargs):
                raise AssertionError("VLM annotation should be skipped")

        bundle = self.store.get_project(project_id)
        adapter = NotebookVisualElementMemoryAdapter(self.store)
        context = VideoPostprocessContext(
            project_id=project_id,
            bundle=bundle,
            shot=bundle["shots"][1],
            attempt_id="a001",
            attempt_dir=attempt_dir,
            video_asset_id="vid_0002_a001",
            video_path=video_path,
            visual_element_status=[],
            dry_run=False,
        )
        with patch("storymem_seedance.executor.extract_shot_memory") as extract, patch.object(
            adapter,
            "_memory_for_current_postprocess",
            return_value=ExplodingMemory(),
        ):
            extract.return_value = {
                "keyframe_paths": [str(extracted_keyframe)],
                "last_frame_path": str(extracted_last),
            }
            result = adapter.process(context)

        expected_paths = [
            str(self.store.asset_path(project_id, asset_id))
            for asset_id in selected_asset_ids
        ]
        self.assertEqual(extract.call_args.kwargs["existing_memory_paths"], expected_paths)
        self.assertLess(len(extract.call_args.kwargs["existing_memory_paths"]), len(produced))
        self.assertEqual(result.postprocess["history_compare_scope"], "selected_storymem_memory_bank")
        self.assertEqual(result.postprocess["history_compare_path_count"], len(expected_paths))

    def test_duplicate_project_is_independent_after_original_delete(self):
        project = self.store.import_story(_story())
        original_id = project["project"]["project_id"]

        duplicate = self.store.duplicate_project(original_id, name="copy")
        duplicate_id = duplicate["project"]["project_id"]
        self.assertNotEqual(original_id, duplicate_id)
        self.assertEqual(duplicate["project"]["name"], "copy")

        self.store.delete_project(original_id)
        duplicate_after_delete = self.store.get_project(duplicate_id)
        self.assertEqual(duplicate_after_delete["project"]["project_id"], duplicate_id)
        self.assertEqual(len(duplicate_after_delete["shots"]), 3)

    def test_duplicate_running_project_interrupts_copied_runtime_state(self):
        project = self.store.import_story(_story())
        original_id = project["project"]["project_id"]
        shot = project["shots"][0]
        shot["state"]["status"] = "running"
        shot["state"]["current_attempt_id"] = "a001"
        self.store.write_shot(original_id, shot)
        self.store.update_project_json(original_id, {"status": "running", "active_shot_id": "0001"})

        duplicate = self.store.duplicate_project(original_id, name="copy")
        copied_first = duplicate["shots"][0]

        self.assertEqual(duplicate["project"]["status"], "interrupted")
        self.assertEqual(duplicate["project"]["active_shot_id"], None)
        self.assertEqual(copied_first["state"]["status"], "interrupted")
        self.assertEqual(copied_first["state"]["current_attempt_id"], None)
        self.assertEqual(copied_first["archived_attempt_ids"], ["a001"])

    def test_recover_interrupted_runs_clears_orphan_running_state(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        shot = project["shots"][0]
        shot["state"]["status"] = "running"
        shot["state"]["current_attempt_id"] = "a001"
        self.store.write_shot(project_id, shot)
        self.store.update_project_json(project_id, {"status": "running", "active_shot_id": "0001"})

        recovered = self.store.recover_interrupted_runs()
        bundle = self.store.get_project(project_id)
        first = bundle["shots"][0]

        self.assertEqual(recovered, [project_id])
        self.assertEqual(bundle["project"]["status"], "interrupted")
        self.assertEqual(bundle["project"]["active_shot_id"], None)
        self.assertEqual(first["state"]["status"], "interrupted")
        self.assertEqual(first["state"]["current_attempt_id"], None)
        self.assertEqual(first["archived_attempt_ids"], ["a001"])

    def test_recover_interrupted_runs_clears_orphan_keyframe_queue_state(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        shot = project["shots"][0]
        shot["state"]["status"] = "keyframe_maintaining_queued"
        shot["state"]["current_attempt_id"] = "a001"
        shot["steps"]["keyframe_maintaining"]["status"] = "keyframe_maintaining_queued"
        self.store.write_shot(project_id, shot)
        self.store.update_project_json(project_id, {"status": "running", "active_shot_id": "0001"})

        recovered = self.store.recover_interrupted_runs()
        first = self.store.get_project(project_id)["shots"][0]

        self.assertEqual(recovered, [project_id])
        self.assertEqual(first["state"]["status"], "interrupted")
        self.assertEqual(first["state"]["current_attempt_id"], None)
        self.assertEqual(first["archived_attempt_ids"], ["a001"])

    def test_fork_after_shot_keeps_prefix_and_resets_later_shots(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        self._mark_completed(project_id, "0001", "a001")
        self._mark_completed(project_id, "0002", "a001")
        self._mark_completed(project_id, "0003", "a001")
        project_file = self.store.project_dir(project_id) / "project.json"
        project_json = read_json(project_file)
        project_json["completed_prefix"] = 3
        project_json["current_final_video_asset_id"] = "vid_final_full"
        write_json_atomic(project_file, project_json)

        fork = self.store.fork_after_shot(project_id, "0001")
        fork_id = fork["project"]["project_id"]
        shots = {shot["shot_id"]: shot for shot in fork["shots"]}

        self.assertEqual(fork["project"]["completed_prefix"], 1)
        self.assertEqual(shots["0001"]["state"]["status"], "completed")
        self.assertEqual(shots["0002"]["state"]["status"], "draft")
        self.assertEqual(shots["0003"]["state"]["status"], "draft")
        self.assertEqual(shots["0002"]["state"]["current_attempt_id"], None)
        self.assertEqual(shots["0002"]["archived_attempt_ids"], ["a001"])
        self.assertEqual(shots["0002"]["inputs"]["video_prompt"], "Prompt 2")
        self.assertEqual(shots["0002"]["inputs"]["duration_seconds"], 9)
        self.assertTrue((self.store.shot_dir(fork_id, "0002") / "attempts" / "a001").exists())

    def test_fake_runner_writes_attempt_and_project_status(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        runner = FakeNotebookRunner(self.store)

        runner.run_shot(project_id, "0001")
        bundle = self.store.get_project(project_id)
        first = bundle["shots"][0]

        self.assertEqual(bundle["project"]["status"], "partial")
        self.assertEqual(bundle["project"]["completed_prefix"], 1)
        self.assertEqual(first["state"]["status"], "completed")
        self.assertEqual(first["state"]["current_attempt_id"], "a001")
        self.assertEqual(first["attempt"]["status"], "completed")
        self.assertEqual(first["attempt"]["runner"]["backend"], "fake")
        self.assertIn("Visual element guidance", first["attempt"]["prompt"]["submitted_prompt"])
        self.assertEqual(first["steps"]["seedance_prompt"]["status"], "seedance_prompt_completed")
        self.assertTrue((self.store.attempt_dir(project_id, "0001", "a001") / "seedance_prompt" / "prompt.txt").exists())
        self.assertEqual(len(first["attempt"]["visual_element_status"]), 1)
        self.assertEqual(len(first["attempt"]["produced_visual_memory"]), 1)
        self.assertEqual(first["attempt"]["logs"][0]["step"], "planning_visual_elements")
        self.assertIn("vid_0001_a001", bundle["assets"])
        self.assertEqual(bundle["project"]["current_final_video_asset_id"], "vid_final_0001_a001")
        self.assertEqual(first["state"]["current_final_video_asset_id"], "vid_final_0001_a001")
        self.assertEqual(first["attempt"]["outputs"]["current_final_video_asset_id"], "vid_final_0001_a001")
        self.assertEqual(bundle["project"]["thumbnail_asset_id"], "img_0001_a001_001")
        self.assertIn("vid_final_0001_a001", bundle["assets"])
        self.assertEqual(
            first["attempt"]["produced_visual_memory"][0]["holistic_description"],
            "Fake holistic keyframe description with composition and story context.",
        )

    def test_run_all_completes_remaining_shots(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]

        FakeNotebookRunner(self.store).run_all(project_id)
        bundle = self.store.get_project(project_id)

        self.assertEqual(bundle["project"]["status"], "completed")
        self.assertEqual(bundle["project"]["completed_prefix"], 3)
        self.assertEqual([shot["state"]["status"] for shot in bundle["shots"]], ["completed", "completed", "completed"])
        self.assertEqual(bundle["shots"][1]["attempt"]["selected_references"][0]["source_shot_id"], "0001")
        self.assertEqual(bundle["project"]["current_final_video_asset_id"], "vid_final_0003_a001")
        self.assertEqual(bundle["shots"][2]["state"]["current_final_video_asset_id"], "vid_final_0003_a001")
        final_asset = bundle["assets"]["vid_final_0003_a001"]
        self.assertEqual(final_asset["metadata"]["assembly_mode"], "placeholder")
        assembly = read_json(self.store.project_dir(project_id) / "final" / "assembly.json")
        self.assertEqual(assembly["prefix_final_video_asset_ids"]["0003"], "vid_final_0003_a001")
        self.assertEqual(bundle["assembly"]["prefix_final_video_asset_ids"]["0003"], "vid_final_0003_a001")
        self.assertIsNotNone(bundle["shots"][0]["steps"]["visual_plan"]["started_at"])
        self.assertIsNotNone(bundle["shots"][0]["steps"]["keyframe_maintaining"]["started_at"])

    def test_completed_project_auto_submits_eval_when_enabled(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        self.store.save_project(project_id, settings_update={"evaluation": {"auto_submit_eval": True}})

        with patch("videogen_notebook.runner.subprocess.Popen") as popen:
            FakeNotebookRunner(self.store).run_all(project_id)

        self.assertEqual(popen.call_count, 1)
        command = popen.call_args.args[0]
        self.assertIn("submit_videogen_eval_to_a6000.py", command[1])
        self.assertEqual(command[2], "--project-dir")
        self.assertEqual(command[3], str(self.store.project_dir(project_id)))
        self.assertEqual(popen.call_args.kwargs["cwd"], "/home/wxh/world_model_projects/StoryMem")

    def test_step_elapsed_formats_completed_step_durations(self):
        steps = {
            "visual_plan": {
                "status": "visual_plan_completed",
                "started_at": "2026-07-07T00:00:00+00:00",
                "updated_at": "2026-07-07T00:01:05+00:00",
            },
            "reference_selection": {
                "status": "reference_selection_completed",
                "started_at": "2026-07-07T00:01:05+00:00",
                "updated_at": "2026-07-07T00:01:07+00:00",
            },
            "seedance_prompt": {
                "status": "seedance_prompt_completed",
                "started_at": "2026-07-07T00:01:07+00:00",
                "updated_at": "2026-07-07T00:01:07+00:00",
            },
            "seedance_generation": {"status": "draft", "started_at": None, "updated_at": None},
        }

        self.assertEqual(_step_elapsed(steps, "visual_plan"), "in 1min 5s")
        self.assertEqual(_step_elapsed(steps, "reference_selection"), "in 2s")
        self.assertEqual(_step_elapsed(steps, "seedance_prompt"), "in 0s")
        self.assertEqual(_step_elapsed(steps), "in 1min 7s")

    def test_keyframe_maintaining_retries_full_step_twice_before_success(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        backend = FlakyKeyframeBackend(self.store, failures_before_success=2)
        runner = NotebookRunner(self.store, backend)

        runner.run_shot(project_id, "0001")

        bundle = self.store.get_project(project_id)
        first = bundle["shots"][0]
        logs = [
            log["message"]
            for log in first["attempt"]["logs"]
            if log["step"] == "annotating_visual_memory"
        ]
        self.assertEqual(backend.maintain_calls, 3)
        self.assertEqual(first["state"]["status"], "completed")
        self.assertTrue(any("full attempt 1/5 failed" in item for item in logs))
        self.assertTrue(any("full attempt 2/5 failed" in item for item in logs))
        self.assertTrue(any("waiting 60s before retrying" in item for item in logs))
        self.assertTrue(any("full attempt 3/5 succeeded" in item for item in logs))

    def test_seedance_prompt_composition_respects_force_animation_setting(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        runner = FakeNotebookRunner(self.store)
        for step in ("visual_plan", "reference_selection", "seedance_prompt"):
            runner.run_step(project_id, "0001", step)
        prompt = self.store.get_project(project_id)["shots"][0]["attempt"]["prompt"]["submitted_prompt"]
        self.assertTrue(prompt.startswith(FORCE_ANIMATION_PROMPT_PREFIX))
        structured_prompt = _visual_reference_prompt_context(
            self.store.get_project(project_id),
            self.store.get_project(project_id)["shots"][0],
            self.store.get_project(project_id)["shots"][0]["inputs"],
            [],
            [],
        )
        self.assertLess(structured_prompt.index("[当前 shot 的生成任务]"), structured_prompt.index("[前序完整剧本]"))
        self.assertEqual(structured_prompt.count(FORCE_ANIMATION_PROMPT_PREFIX), 2)
        current_task = structured_prompt.split("[前序完整剧本]", 1)[0]
        overall_constraints = structured_prompt.split("[总体约束]", 1)[1]
        self.assertIn(FORCE_ANIMATION_PROMPT_PREFIX, current_task)
        self.assertIn(FORCE_ANIMATION_PROMPT_PREFIX, overall_constraints)
        second_shot = self.store.get_project(project_id)["shots"][1]
        second_prompt = _visual_reference_prompt_context(
            self.store.get_project(project_id),
            second_shot,
            second_shot["inputs"],
            [],
            [],
        )
        previous_script = second_prompt.split("[前序完整剧本]", 1)[1].split("[历史参考图说明]", 1)[0]
        self.assertIn("Prompt 1", previous_script)
        self.assertIn("Prompt 2", previous_script)
        self.assertNotIn("Prompt 3", previous_script)
        edited_prompt = prompt.replace(FORCE_ANIMATION_PROMPT_PREFIX + "\n\n", "", 1)
        self.store.save_seedance_prompt(project_id, "0001", edited_prompt)
        runner.run_step(project_id, "0001", "seedance_generation")
        submitted = self.store.get_project(project_id)["shots"][0]["attempt"]["prompt"]["submitted_prompt"]
        self.assertEqual(submitted, edited_prompt)
        self.assertNotIn(FORCE_ANIMATION_PROMPT_PREFIX, submitted)

        disabled = self.store.import_story(_story(), name="No animation prefix")
        disabled_id = disabled["project"]["project_id"]
        self.store.save_project(
            disabled_id,
            settings_update={"generation": {"force_animation_style": False}},
        )
        disabled_runner = FakeNotebookRunner(self.store)
        for step in ("visual_plan", "reference_selection", "seedance_prompt"):
            disabled_runner.run_step(disabled_id, "0001", step)
        disabled_prompt = self.store.get_project(disabled_id)["shots"][0]["attempt"]["prompt"]["submitted_prompt"]
        self.assertNotIn(FORCE_ANIMATION_PROMPT_PREFIX, disabled_prompt)

    def test_seedance_prompt_module_settings_can_disable_ablation_sections(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        self.store.save_project(
            project_id,
            settings_update={
                "prompt_modules": {
                    "full_script_context": False,
                    "visual_element_plan": False,
                    "holistic_guidance": False,
                    "should_reference": False,
                    "should_exclude": False,
                }
            },
        )
        bundle = self.store.get_project(project_id)
        shot = bundle["shots"][0]
        input_snapshot = {**shot["inputs"], "project_settings": bundle["settings"]}
        visual_status = [
            {"element_id": "e-character", "name": "Pilot", "status": "should_reference"},
            {"element_id": "e-object", "name": "Wrong prop", "status": "should_exclude"},
        ]
        references = [
            {
                "reference_index": 1,
                "source_scene_num": 1,
                "source_shot_num": 1,
                "holistic_description": "A cockpit frame with strong rim light.",
                "reference_guidance": "Use the cockpit rhythm.",
                "visual_element_selection": {
                    "should_reference": [{"id": "e-character", "name": "Pilot"}],
                    "should_exclude": [{"id": "e-object", "name": "Wrong prop"}],
                },
            }
        ]

        prompt = _visual_reference_prompt_context(bundle, shot, input_snapshot, visual_status, references)

        self.assertIn("[当前 shot 的生成任务]", prompt)
        self.assertIn("[历史参考图说明]", prompt)
        self.assertIn("[总体约束]", prompt)
        self.assertNotIn("[前序完整剧本]", prompt)
        self.assertNotIn("[本 shot 的视觉元素计划]", prompt)
        self.assertNotIn("A cockpit frame with strong rim light.", prompt)
        self.assertNotIn("Use the cockpit rhythm.", prompt)
        self.assertNotIn("应参考其中的", prompt)
        self.assertNotIn("不应引入其中的", prompt)
        self.assertNotIn("Pilot", prompt)
        self.assertNotIn("Wrong prop", prompt)

    def test_predefined_references_appear_before_historical_references_in_prompt(self):
        bundle = self.store.import_story(_story())
        shot = bundle["shots"][1]
        input_snapshot = {
            **shot["inputs"],
            "predefined_references": [
                {
                    "id": "pref-0001",
                    "asset_id": "pref_0002_pref_0001",
                    "label": "Main character sheet",
                    "guidance": "Keep the animated character design consistent.",
                }
            ],
        }
        references = [
            {
                "reference_index": 1,
                "source_scene_num": 1,
                "source_shot_num": 1,
                "covered_elements": ["Pilot"],
                "visual_element_selection": {"should_reference": [{"name": "Pilot"}]},
            }
        ]

        prompt = _visual_reference_prompt_context(bundle, shot, input_snapshot, [], references)

        self.assertLess(prompt.index("[预定义参考图说明]"), prompt.index("[历史参考图说明]"))
        self.assertIn("参考图片 Image 1: Main character sheet", prompt)
        self.assertIn("参考图片 Image 2 来自 Scene 1 / Shot 1", prompt)

    def test_structured_prompt_only_mentions_reference_video_when_present(self):
        bundle = self.store.import_story(_story())
        first = bundle["shots"][0]
        second = bundle["shots"][1]
        first_prompt = _visual_reference_prompt_context(bundle, first, first["inputs"], [], [])
        second_inputs = {**second["inputs"], "generation_mode": "smooth", "is_cut": False}
        second_prompt = _visual_reference_prompt_context(bundle, second, second_inputs, [], [])

        old_conditional = "如果本次输入媒体包含参考视频，则请生成参考视频的延长视频"
        new_instruction = "本次输入媒体包含参考视频，请生成参考视频的延长视频"
        self.assertNotIn(old_conditional, first_prompt)
        self.assertNotIn(new_instruction, first_prompt)
        self.assertNotIn(old_conditional, second_prompt)
        self.assertIn(new_instruction, second_prompt)

    def test_keyframe_maintaining_uses_single_gpu_queue_across_projects(self):
        first = self.store.import_story(_story())
        second = self.store.import_story(_story(), name="Second project")
        first_id = first["project"]["project_id"]
        second_id = second["project"]["project_id"]
        backend = BlockingKeyframeBackend(self.store)
        backend.prepare_project(first_id)
        backend.prepare_project(second_id)
        runner = NotebookRunner(self.store, backend, gpu_lock=KeyframeGpuLock())

        for project_id in (first_id, second_id):
            for step in ("visual_plan", "reference_selection", "seedance_prompt", "seedance_generation"):
                runner.run_step(project_id, "0001", step)

        first_thread = threading.Thread(target=runner.run_step, args=(first_id, "0001", "keyframe_maintaining"))
        second_thread = threading.Thread(target=runner.run_step, args=(second_id, "0001", "keyframe_maintaining"))
        first_thread.start()
        self.assertTrue(backend.entered[first_id].wait(timeout=2))
        second_thread.start()
        time.sleep(0.2)

        second_bundle = self.store.get_project(second_id)
        second_shot = second_bundle["shots"][0]
        self.assertEqual(second_shot["state"]["status"], "keyframe_maintaining_queued")
        self.assertEqual(second_bundle["project"]["run_state"]["status"], "keyframe_maintaining_queued")
        self.assertEqual(second_bundle["project"]["run_state"]["step"], "keyframe_maintaining")
        self.assertEqual(backend.max_active, 1)

        backend.release[first_id].set()
        first_thread.join(timeout=5)
        self.assertFalse(first_thread.is_alive())
        self.assertTrue(backend.entered[second_id].wait(timeout=2))
        second_running = self.store.get_project(second_id)
        self.assertEqual(second_running["project"]["run_state"]["status"], "keyframe_maintaining_running")
        self.assertEqual(backend.max_active, 1)
        backend.release[second_id].set()
        second_thread.join(timeout=5)
        self.assertFalse(second_thread.is_alive())

        first_after = self.store.get_project(first_id)["shots"][0]
        second_after = self.store.get_project(second_id)["shots"][0]
        self.assertEqual(first_after["state"]["status"], "completed")
        self.assertEqual(second_after["state"]["status"], "completed")
        self.assertEqual(backend.order, [first_id, second_id])
        logs = [log["step"] for log in second_after["attempt"]["logs"]]
        self.assertIn("waiting_for_gpu_keyframe_slot", logs)

    def test_visual_plan_retries_full_step_before_success(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        backend = FlakyAlgorithmStepBackend(self.store, {"visual_plan": 2})
        runner = NotebookRunner(self.store, backend)

        runner.run_step(project_id, "0001", "visual_plan")

        bundle = self.store.get_project(project_id)
        first = bundle["shots"][0]
        logs = [
            log["message"]
            for log in first["attempt"]["logs"]
            if log["step"] == "planning_visual_elements"
        ]
        self.assertEqual(backend.calls["visual_plan"], 3)
        self.assertEqual(first["state"]["status"], "visual_plan_completed")
        self.assertTrue(any("Visual Elements Plan attempt 1/5 failed" in item for item in logs))
        self.assertTrue(any("Visual Elements Plan attempt 2/5 failed" in item for item in logs))
        self.assertTrue(any("waiting 60s before retrying" in item for item in logs))
        self.assertTrue(any("Visual Elements Plan attempt 3/5 succeeded" in item for item in logs))

    def test_reference_and_seedance_generation_use_global_step_attempts(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        self.store.save_project(
            project_id,
            settings_update={"generation": {"algorithm_step_max_attempts": 2}},
        )
        backend = FlakyAlgorithmStepBackend(
            self.store,
            {"reference_selection": 1, "seedance_generation": 1},
        )
        runner = NotebookRunner(self.store, backend)

        runner.run_shot(project_id, "0001")

        bundle = self.store.get_project(project_id)
        first = bundle["shots"][0]
        logs = [log["message"] for log in first["attempt"]["logs"]]
        self.assertEqual(backend.calls["reference_selection"], 2)
        self.assertEqual(backend.calls["seedance_generation"], 2)
        self.assertEqual(first["state"]["status"], "completed")
        self.assertTrue(any("Historical Reference Selection attempt 1/2 failed" in item for item in logs))
        self.assertTrue(any("Seedance Video Generation attempt 1/2 failed" in item for item in logs))

    def test_reference_guidance_llm_patch_and_prompt_context_include_text_memory(self):
        project = self.store.import_story(_story())
        bundle = self.store.get_project(project["project"]["project_id"])
        shot = bundle["shots"][1]
        context = ReferenceSelectionContext(
            project_id=bundle["project"]["project_id"],
            bundle=bundle,
            shot=shot,
            previous_shots=bundle["shots"][:1],
            attempt_id="a001",
            attempt_dir=self.store.project_dir(bundle["project"]["project_id"]),
            input_snapshot=shot["inputs"],
            visual_element_status=[
                {
                    "element_id": "element-0001",
                    "name": "Pilot",
                    "type": "character",
                    "status": "should_reference",
                    "reason": "Keep the pilot consistent.",
                }
            ],
            visual_element_details={},
        )
        references = [
            {
                "reference_id": "ref-001",
                "reference_index": 1,
                "source_scene_num": 1,
                "source_shot_num": 1,
                "source_shot_id": "0001",
                "covered_elements": ["element-0001"],
                "conflict_elements": [],
                "visual_element_selection": {
                    "should_reference": [{"id": "element-0001", "name": "Pilot"}],
                    "should_exclude": [],
                    "optional_or_uncertain": [],
                },
                "holistic_description": "Pilot stands near a glowing console in a tense cockpit.",
            },
            {
                "reference_id": "ref-002",
                "reference_index": 2,
                "source_scene_num": 1,
                "source_shot_num": 1,
                "source_shot_id": "0001",
                "covered_elements": [],
                "conflict_elements": [],
                "visual_element_selection": {
                    "should_reference": [],
                    "should_exclude": [],
                    "optional_or_uncertain": [],
                },
                "holistic_description": "A broad animated color and lighting reference.",
            }
        ]

        with patch("videogen_notebook.runner._call_ark_chat") as call:
            call.return_value = (
                '{"references":[{"reference_id":"ref-001","reference_guidance":"Use the cockpit layout and tense posture as the overall reference."}]}',
                {"usage": {"total_tokens": 8}},
            )
            enriched, details, logs = _add_reference_guidance(context, references)

        self.assertEqual(details["status"], "completed")
        self.assertIn("reference guidance generated", logs[0]["message"])
        self.assertEqual(
            enriched[0]["reference_guidance"],
            "Use the cockpit layout and tense posture as the overall reference.",
        )
        self.assertEqual(enriched[1]["reference_guidance"], STYLE_REFERENCE_GUIDANCE)
        prompt = _visual_reference_prompt_context(bundle, shot, shot["inputs"], context.visual_element_status, enriched)
        self.assertIn("Pilot stands near a glowing console", prompt)
        self.assertIn("Use the cockpit layout", prompt)
        self.assertIn(STYLE_REFERENCE_GUIDANCE, prompt)
        self.assertIn("针对视觉计划中各元素的客观参考约束如下:", prompt)

    def test_rerunning_completed_prefix_writes_new_current_output_for_shot(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        runner = FakeNotebookRunner(self.store)
        runner.run_all(project_id)

        self.store.reset_from_shot(project_id, "0002")
        runner.run_shot(project_id, "0002")
        bundle = self.store.get_project(project_id)
        second = bundle["shots"][1]
        assembly = bundle["assembly"]

        self.assertEqual(second["state"]["current_attempt_id"], "a002")
        self.assertEqual(second["state"]["current_final_video_asset_id"], "vid_final_0002_a002")
        self.assertEqual(second["attempt"]["outputs"]["current_final_video_asset_id"], "vid_final_0002_a002")
        self.assertEqual(bundle["project"]["current_final_video_asset_id"], "vid_final_0002_a002")
        self.assertEqual(assembly["prefix_final_video_asset_ids"]["0002"], "vid_final_0002_a002")
        self.assertIn("vid_final_0002_a002", bundle["assets"])

    def test_run_shot_can_pause_before_seedance_when_confirmation_is_required(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        self.store.save_project(
            project_id,
            settings_update={"generation": {"require_human_confirmation_before_seedance": True}},
        )

        FakeNotebookRunner(self.store).run_shot(project_id, "0001", apply_review_delay=True)
        bundle = self.store.get_project(project_id)
        first = bundle["shots"][0]

        self.assertEqual(bundle["project"]["run_state"]["status"], "paused_for_confirmation")
        self.assertEqual(first["state"]["status"], "seedance_prompt_completed")
        self.assertEqual(first["steps"]["seedance_prompt"]["status"], "seedance_prompt_completed")
        self.assertEqual(first["steps"]["seedance_generation"]["status"], "draft")

    def test_resume_interrupted_shot_inherits_completed_step_artifacts(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        runner = FakeNotebookRunner(self.store)
        for step in ("visual_plan", "reference_selection", "seedance_prompt"):
            runner.run_step(project_id, "0001", step)

        first = self.store.get_project(project_id)["shots"][0]
        self.assertEqual(first["state"]["current_attempt_id"], "a001")
        first["state"]["status"] = "interrupted"
        first["state"]["current_attempt_id"] = None
        first["archived_attempt_ids"] = ["a001"]
        first["steps"]["seedance_generation"]["status"] = "seedance_generation_running"
        self.store.write_shot(project_id, first)
        self.store.update_project_json(project_id, {"status": "interrupted", "active_shot_id": None})

        runner.run_shot(project_id, "0001")

        resumed = self.store.get_project(project_id)["shots"][0]
        attempt_id = resumed["state"]["current_attempt_id"]
        attempt_dir = self.store.attempt_dir(project_id, "0001", attempt_id)
        visual_details = read_json(attempt_dir / "visual_plan" / "details.json")
        attempt = read_json(attempt_dir / "attempt.json")

        self.assertEqual(attempt_id, "a002")
        self.assertEqual(resumed["state"]["status"], "completed")
        self.assertTrue((attempt_dir / "visual_plan" / "visual_element_status.json").exists())
        self.assertTrue((attempt_dir / "reference_selection" / "prompt_context.json").exists())
        self.assertEqual(len(visual_details["visual_element_record"]["registry_after"]), 1)
        self.assertEqual(len(attempt["visual_element_status"]), 1)
        self.assertEqual(attempt["outputs"]["raw_video_asset_id"], "vid_0001_a002")

    def test_resume_interrupted_keyframe_step_inherits_seedance_output(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        runner = FakeNotebookRunner(self.store)
        for step in ("visual_plan", "reference_selection", "seedance_prompt", "seedance_generation"):
            runner.run_step(project_id, "0001", step)

        first = self.store.get_project(project_id)["shots"][0]
        first["state"]["status"] = "interrupted"
        first["state"]["current_attempt_id"] = None
        first["archived_attempt_ids"] = ["a001"]
        first["steps"]["keyframe_maintaining"]["status"] = "keyframe_maintaining_running"
        self.store.write_shot(project_id, first)
        self.store.update_project_json(project_id, {"status": "interrupted", "active_shot_id": None})

        runner.run_shot(project_id, "0001")

        resumed = self.store.get_project(project_id)["shots"][0]
        attempt_id = resumed["state"]["current_attempt_id"]
        attempt = read_json(self.store.attempt_dir(project_id, "0001", attempt_id) / "attempt.json")

        self.assertEqual(attempt_id, "a002")
        self.assertEqual(resumed["state"]["status"], "completed")
        self.assertEqual(attempt["outputs"]["raw_video_asset_id"], "vid_0001_a001")
        self.assertEqual(len(attempt["produced_visual_memory"]), 1)

    def test_visual_plan_reflect_updates_rows_without_extra_step_state(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        runner = FakeNotebookRunner(self.store)
        runner.run_step(project_id, "0001", "visual_plan")

        with patch("videogen_notebook.runner.reflect_visual_plan_with_llm", side_effect=_fake_reflection):
            runner.reflect_visual_plan(project_id, "0001")

        bundle = self.store.get_project(project_id)
        first = bundle["shots"][0]
        details = first["attempt"]["visual_element_details"]

        self.assertEqual(first["state"]["status"], "visual_plan_completed")
        self.assertEqual(first["steps"]["visual_plan"]["status"], "visual_plan_completed")
        self.assertEqual(first["steps"]["reference_selection"]["status"], "draft")
        self.assertEqual(first["attempt"]["visual_element_status"][0]["name"], "Reflected element")
        self.assertEqual(details["reflection"]["summary"], "fake reflection applied")
        self.assertIn(
            "visual plan reflection applied",
            [log["message"] for log in first["attempt"]["logs"] if log["step"] == "planning_visual_elements"][-1],
        )

    def test_auto_reflect_visual_plan_runs_during_automatic_shot_run(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        self.store.save_project(
            project_id,
            settings_update={"generation": {"auto_reflect_visual_plan": True}},
        )
        runner = FakeNotebookRunner(self.store)

        with patch("videogen_notebook.runner.reflect_visual_plan_with_llm", side_effect=_fake_reflection) as reflected:
            runner.run_shot(project_id, "0001", apply_review_delay=True)

        bundle = self.store.get_project(project_id)
        first = bundle["shots"][0]

        self.assertEqual(reflected.call_count, 1)
        self.assertEqual(first["state"]["status"], "completed")
        self.assertEqual(first["attempt"]["visual_element_status"][0]["name"], "Reflected element")
        self.assertEqual(first["attempt"]["visual_element_details"]["reflection"]["summary"], "fake reflection applied")

    def test_runner_refreshes_project_memory_state(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]

        FakeNotebookRunner(self.store).run_all(project_id)
        visual_state = read_json(self.store.project_dir(project_id) / "memory" / "visual_state.json")
        lineage = read_json(self.store.project_dir(project_id) / "memory" / "lineage.json")

        self.assertEqual(visual_state["completed_prefix"], 3)
        self.assertEqual(len(visual_state["elements"]), 3)
        self.assertEqual(visual_state["produced_visual_memory"][-1]["source_shot_id"], "0003")
        self.assertEqual([shot["shot_id"] for shot in lineage["shots"]], ["0001", "0002", "0003"])

        self.store.reset_from_shot(project_id, "0002")
        visual_state = read_json(self.store.project_dir(project_id) / "memory" / "visual_state.json")
        lineage = read_json(self.store.project_dir(project_id) / "memory" / "lineage.json")

        self.assertEqual(visual_state["completed_prefix"], 1)
        self.assertEqual([item["source_shot_id"] for item in visual_state["produced_visual_memory"]], ["0001"])
        self.assertEqual([shot["shot_id"] for shot in lineage["shots"]], ["0001"])
        self.assertEqual(lineage["reset_from_shot_id"], "0002")

    def test_reset_from_shot_archives_current_attempts_and_keeps_inputs(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        FakeNotebookRunner(self.store).run_all(project_id)

        reset = self.store.reset_from_shot(project_id, "0002")
        shots = {shot["shot_id"]: shot for shot in reset["shots"]}

        self.assertEqual(reset["project"]["completed_prefix"], 1)
        self.assertEqual(reset["project"]["current_final_video_asset_id"], "vid_final_0001_a001")
        self.assertEqual(shots["0001"]["state"]["status"], "completed")
        self.assertEqual(shots["0002"]["state"]["status"], "draft")
        self.assertEqual(shots["0003"]["state"]["status"], "draft")
        self.assertEqual(shots["0002"]["state"]["current_attempt_id"], None)
        self.assertEqual(shots["0002"]["archived_attempt_ids"], ["a001"])
        self.assertEqual(shots["0002"]["inputs"]["video_prompt"], "Prompt 2")
        self.assertEqual(shots["0002"]["inputs"]["duration_seconds"], 9)
        self.assertTrue((self.store.shot_dir(project_id, "0002") / "attempts" / "a001").exists())

    def test_save_all_allows_unchanged_completed_shots(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        FakeNotebookRunner(self.store).run_shot(project_id, "0001")

        bundle = self.store.get_project(project_id)
        updates = []
        for shot in bundle["shots"]:
            inputs = dict(shot["inputs"])
            if shot["shot_id"] == "0002":
                inputs["video_prompt"] = "Changed prompt 2"
            updates.append({"shot_id": shot["shot_id"], **inputs})

        saved = self.store.save_project(project_id, shot_updates=updates)
        self.assertEqual(saved["shots"][1]["inputs"]["video_prompt"], "Changed prompt 2")

    def test_real_backend_dry_run_records_request_shape(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        runner = create_runner(self.store, "real", real_submit=False)

        runner.run_shot(project_id, "0001")
        bundle = self.store.get_project(project_id)
        first = bundle["shots"][0]

        self.assertEqual(bundle["project"]["status"], "partial")
        self.assertEqual(first["state"]["status"], "completed")
        self.assertEqual(first["attempt"]["status"], "completed")
        self.assertEqual(first["attempt"]["runner"]["backend"], "real")
        self.assertTrue(first["attempt"]["request"]["dry_run"])
        self.assertEqual(first["attempt"]["request"]["seedance"]["duration"], 8)
        self.assertIn("CURRENT GENERATION TASK", first["attempt"]["prompt"]["submitted_prompt"])
        self.assertEqual(first["attempt"]["logs"][0]["step"], "planning_visual_elements")
        self.assertIn("vid_0001_a001_dryrun", bundle["assets"])

    def test_real_backend_dry_run_uses_injected_visual_planner(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        planner = FakeVisualPlanner()
        runner = create_runner(self.store, "real", real_visual_planner=planner, real_submit=False)

        runner.run_shot(project_id, "0001")
        bundle = self.store.get_project(project_id)
        first = bundle["shots"][0]

        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(planner.calls[0].shot["shot_id"], "0001")
        self.assertEqual(first["attempt"]["prompt"]["submitted_prompt"], "FULL PROMPT FROM FAKE VISUAL PLANNER")
        self.assertEqual(first["attempt"]["request"]["content"][0]["text"], "FULL PROMPT FROM FAKE VISUAL PLANNER")
        self.assertEqual(first["attempt"]["visual_element_status"][0]["element_id"], "element-fake-planner")
        self.assertEqual(first["attempt"]["logs"][0]["step"], "fake_visual_plan")

    def test_real_backend_submit_uses_injected_client_without_network(self):
        story = _story()
        story["scenes"][0]["cut"] = [True, True, False]
        project = self.store.import_story(story)
        project_id = project["project"]["project_id"]
        client = FakeSubmitClient()
        runner = create_runner(
            self.store,
            "real",
            real_client=client,
            real_visual_planner=PlaceholderVisualMemoryPlanner(),
            real_postprocessor=PlaceholderVideoPostprocessor(self.store),
            real_submit=True,
        )

        runner.run_shot(project_id, "0001")
        runner.run_shot(project_id, "0002")
        bundle = self.store.get_project(project_id)
        second = bundle["shots"][1]

        self.assertEqual(len(client.created_calls), 2)
        self.assertEqual(client.created_calls[1]["options"]["duration"], 9)
        self.assertEqual(client.waited_task_ids, ["task-1", "task-2"])
        self.assertEqual(client.downloads[1]["url"], "https://example.test/task-2.mp4")
        self.assertEqual(client.created_calls[1]["content"][0]["type"], "text")
        self.assertEqual(client.created_calls[1]["content"][1]["type"], "image_url")
        self.assertNotIn("previous ending frame", client.created_calls[1]["content"][0]["text"])
        self.assertFalse(second["attempt"]["request"]["dry_run"])
        self.assertEqual(second["attempt"]["seedance"]["task_id"], "task-2")
        self.assertEqual(bundle["assets"]["vid_0002_a001"]["metadata"]["dry_run"], False)
        stored_image_items = [
            item for item in second["attempt"]["request"]["content"] if item.get("type") == "image_url"
        ]
        self.assertTrue(stored_image_items)
        self.assertFalse(stored_image_items[0]["image_url"]["url"].startswith("data:"))
        self.assertTrue(stored_image_items[0]["image_url"].get("embedded_data_url"))

    def test_seedance_generation_drops_sensitive_image_for_submission_retry_only(self):
        predefined = self.workspace / "predefined-ref.jpg"
        predefined.write_bytes(b"fake predefined reference")
        story = _story()
        story["scenes"][0]["cut"] = [True, True, False]
        story["scenes"][0]["predefined_references"] = [
            [],
            [
                {
                    "image_path": str(predefined),
                    "label": "Sensitive candidate",
                    "guidance": "Use as a test reference.",
                }
            ],
            [],
        ]
        project = self.store.import_story(story)
        project_id = project["project"]["project_id"]
        client = SensitiveImageOnceFakeSubmitClient(rejected_content_index=1)
        runner = create_runner(
            self.store,
            "real",
            real_client=client,
            real_visual_planner=PlaceholderVisualMemoryPlanner(),
            real_postprocessor=PlaceholderVideoPostprocessor(self.store),
            real_submit=True,
        )

        runner.run_shot(project_id, "0001")
        runner.run_step(project_id, "0002", "visual_plan")
        runner.run_step(project_id, "0002", "reference_selection")
        runner.run_step(project_id, "0002", "seedance_prompt")
        runner.run_step(project_id, "0002", "seedance_generation")
        bundle = self.store.get_project(project_id)
        second = bundle["shots"][1]

        self.assertEqual(len(client.created_calls), 3)
        retried_content = client.created_calls[2]["content"]
        self.assertEqual([item["type"] for item in retried_content], ["text", "image_url"])
        self.assertEqual(len(second["attempt"]["selected_references"]), 1)
        self.assertEqual(len(second["inputs"]["predefined_references"]), 1)
        self.assertEqual(second["state"]["status"], "seedance_generation_completed")
        request = second["attempt"]["request"]
        dropped = request["sensitive_content_filter"]["dropped"]
        self.assertEqual(dropped[0]["content_index"], 1)
        self.assertEqual(dropped[0]["type"], "image_url")
        self.assertIn("pref_0002_pref_0001", dropped[0]["local_source"])
        self.assertEqual([item["type"] for item in request["content"]], ["text", "image_url"])
        messages = [log["message"] for log in second["attempt"]["logs"]]
        self.assertTrue(any("InputImageSensitiveContentDetected" in message for message in messages))
        persisted = read_json(
            self.store.attempt_dir(project_id, "0002", "a001") / "seedance_generation" / "request.json"
        )
        self.assertEqual(persisted["sensitive_content_filter"]["dropped"][0]["content_index"], 1)

    def test_seedance_media_debug_records_reference_video_preflight(self):
        with patch("videogen_notebook.runner.requests.get", return_value=FakeMediaPreflightResponse()) as get:
            debug = _media_debug_for_submission(
                [
                    {"type": "text", "text": "prompt"},
                    {
                        "type": "video_url",
                        "role": "reference_video",
                        "video_url": {"url": "https://tmpfiles.example.test/download/tail.mp4"},
                        "metadata": {"source_shot_id": "0008"},
                    },
                ]
            )

        reference = debug["reference_videos"][0]
        self.assertEqual(reference["url"], "https://tmpfiles.example.test/download/tail.mp4")
        self.assertEqual(reference["metadata"]["source_shot_id"], "0008")
        self.assertEqual(reference["preflight"]["final_url"], "https://cdn.example.test/tail.mp4")
        self.assertEqual(reference["preflight"]["content_type"], "video/mp4")
        self.assertTrue(reference["preflight"]["is_probable_mp4"])
        get.assert_called_once()

    def test_default_non_cut_submits_previous_last_frame_after_memory_references(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        self.store.save_project(
            project_id,
            shot_updates=[
                {
                    "shot_id": "0002",
                    "video_prompt": "Prompt 2",
                    "is_cut": False,
                    "generation_mode": "default",
                    "duration_seconds": 9,
                }
            ],
        )
        client = FakeSubmitClient()
        runner = create_runner(
            self.store,
            "real",
            real_client=client,
            real_visual_planner=PlaceholderVisualMemoryPlanner(),
            real_postprocessor=PlaceholderVideoPostprocessor(self.store),
            real_submit=True,
        )

        runner.run_shot(project_id, "0001")
        runner.run_shot(project_id, "0002")

        second_content = client.created_calls[1]["content"]
        self.assertEqual([item["type"] for item in second_content], ["text", "image_url", "image_url"])
        self.assertEqual(second_content[1].get("role"), "reference_image")
        self.assertEqual(second_content[2].get("role"), "reference_image")
        self.assertEqual(second_content[2]["metadata"]["roles"], ["previous_last_frame"])
        self.assertIn("final reference image", second_content[0]["text"])
        self.assertIn("first-frame continuity constraint", second_content[0]["text"])

    def test_real_backend_submits_predefined_references_before_historical_references(self):
        source = self.workspace / "shot-two-predefined.jpg"
        source.write_bytes(b"fake predefined reference")
        story = _story()
        story["scenes"][0]["cut"] = [True, True, False]
        story["scenes"][0]["predefined_references"] = [
            [],
            [{"image_path": str(source), "label": "Character sheet", "guidance": "Use for identity."}],
            [],
        ]
        project = self.store.import_story(story)
        project_id = project["project"]["project_id"]
        client = FakeSubmitClient()
        runner = create_runner(
            self.store,
            "real",
            real_client=client,
            real_visual_planner=PlaceholderVisualMemoryPlanner(),
            real_postprocessor=PlaceholderVideoPostprocessor(self.store),
            real_submit=True,
        )

        runner.run_shot(project_id, "0001")
        runner.run_shot(project_id, "0002")

        second_content = client.created_calls[1]["content"]
        image_items = [item for item in second_content if item.get("type") == "image_url"]
        self.assertGreaterEqual(len(image_items), 2)
        self.assertEqual(image_items[0]["metadata"]["source_type"], "predefined_reference")
        self.assertEqual(image_items[0]["metadata"]["label"], "Character sheet")
        self.assertEqual(image_items[1]["metadata"]["source_shot_id"], "0001")

    def test_default_prompt_without_last_frame_context_is_plain(self):
        self.assertEqual(_prompt_for_generation_mode("plain prompt", "default"), "plain prompt")

    def test_real_backend_defaults_to_submit_when_client_is_injected(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        client = FakeSubmitClient()
        runner = create_runner(
            self.store,
            "real",
            real_client=client,
            real_visual_planner=PlaceholderVisualMemoryPlanner(),
            real_postprocessor=PlaceholderVideoPostprocessor(self.store),
        )

        runner.run_shot(project_id, "0001")
        bundle = self.store.get_project(project_id)
        first = bundle["shots"][0]

        self.assertEqual(len(client.created_calls), 1)
        self.assertFalse(first["attempt"]["request"]["dry_run"])
        self.assertEqual(first["attempt"]["seedance"]["task_id"], "task-1")

    def test_real_backend_submit_uses_injected_postprocessor(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        client = FakeSubmitClient()
        postprocessor = FakePostprocessor()
        runner = create_runner(
            self.store,
            "real",
            real_client=client,
            real_visual_planner=FakeVisualPlanner(),
            real_postprocessor=postprocessor,
            real_submit=True,
        )

        runner.run_shot(project_id, "0001")
        bundle = self.store.get_project(project_id)
        first = bundle["shots"][0]

        self.assertEqual(len(postprocessor.calls), 1)
        self.assertFalse(postprocessor.calls[0].dry_run)
        self.assertEqual(postprocessor.calls[0].video_asset_id, "vid_0001_a001")
        self.assertEqual(first["attempt"]["postprocess"]["postprocessor"], "fake_postprocessor")
        self.assertEqual(first["attempt"]["produced_visual_memory"][0]["visible_elements"], ["fake-element"])
        self.assertIn("img_0001_a001_fakepp", bundle["assets"])
        self.assertIn("fake_postprocess", [log["step"] for log in first["attempt"]["logs"]])

    def test_project_load_hydrates_visual_status_metadata(self):
        project = self.store.import_story(_story())
        project_id = project["project"]["project_id"]
        self._mark_completed(project_id, "0001", "a001")
        self._mark_completed(project_id, "0002", "a001")

        first_attempt = self.store.attempt_dir(project_id, "0001", "a001")
        second_attempt = self.store.attempt_dir(project_id, "0002", "a001")
        write_json_atomic(
            first_attempt / "visual_element_status.json",
            [
                {
                    "element_id": "element-0001",
                    "name": "Pilot",
                    "type": "character",
                    "introduced_at": "shot-0001",
                    "notes": "Blue jacket and silver helmet.",
                    "status": "new",
                    "reason": "Introduced in shot 1.",
                }
            ],
        )
        write_json_atomic(
            second_attempt / "visual_element_status.json",
            [
                {
                    "element_id": "element-0001",
                    "name": "Pilot",
                    "type": "character",
                    "introduced_at": "element-0001",
                    "notes": "",
                    "status": "should_reference",
                    "reason": "Still visible.",
                }
            ],
        )

        bundle = self.store.get_project(project_id)
        row = bundle["shots"][1]["attempt"]["visual_element_status"][0]

        self.assertEqual(row["introduced_at"], "shot-0001")
        self.assertEqual(row["notes"], "Blue jacket and silver helmet.")

    def _mark_completed(self, project_id: str, shot_id: str, attempt_id: str) -> None:
        shot_path = self.store.shot_dir(project_id, shot_id) / "shot.json"
        shot = read_json(shot_path)
        shot["state"]["status"] = "completed"
        shot["state"]["current_attempt_id"] = attempt_id
        write_json_atomic(shot_path, shot)
        attempt_dir = self.store.shot_dir(project_id, shot_id) / "attempts" / attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(attempt_dir / "attempt.json", {"attempt_id": attempt_id, "status": "completed"})


class VideogenNotebookAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.app = create_app(self.tempdir.name)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_homepage_and_create_project(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("videogen_notebook", response.text)

        created = self.client.post("/projects", data={"name": "Demo"}, follow_redirects=False)
        self.assertEqual(created.status_code, 303)
        project_page = self.client.get(created.headers["location"])
        self.assertEqual(project_page.status_code, 200)
        self.assertIn("Demo", project_page.text)
        self.assertIn("Visual Elements Plan", project_page.text)
        self.assertIn("Historical Reference Selection", project_page.text)
        self.assertIn("Add Shot", project_page.text)
        self.assertIn("Export JSON", project_page.text)
        self.assertIn("Step max attempts", project_page.text)
        self.assertIn("Seedance settings", project_page.text)

    def test_project_page_places_status_actions_before_step_one(self):
        project = self.app.state.store.import_story(_story())
        project_page = self.client.get(f"/projects/{project['project']['project_id']}")

        self.assertEqual(project_page.status_code, 200)
        self.assertLess(project_page.text.index("Run Shot"), project_page.text.index("Step 1 · Shot Design"))

    def test_predefined_reference_upload_and_save_controls(self):
        project = self.app.state.store.import_story(_story())
        project_id = project["project"]["project_id"]
        response = self.client.post(
            f"/projects/{project_id}/shots/0001/predefined-references/add",
            data={"label": "Hero sheet", "guidance": "Use as a stable animated character sheet."},
            files={"image_file": ("hero.jpg", b"fake hero image", "image/jpeg")},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        bundle = self.app.state.store.get_project(project_id)
        reference = bundle["shots"][0]["inputs"]["predefined_references"][0]
        page = self.client.get(f"/projects/{project_id}")
        self.assertIn("Predefined References", page.text)
        self.assertIn("Hero sheet", page.text)
        self.assertTrue(reference["media_url"].startswith(f"/projects/{project_id}/assets/"))

        edited = self.client.post(
            f"/projects/{project_id}/save",
            data={
                "name": "Demo Story",
                "project-name": "Demo Story",
                "shot-0001-prompt": "Prompt 1",
                "shot-0001-is-cut": "on",
                "shot-0001-generation-mode": "default",
                "shot-0001-duration": "8",
                "shot-0001-predefined-id": reference["id"],
                "shot-0001-predefined-keep": reference["id"],
                "shot-0001-predefined-label": "Edited hero sheet",
                "shot-0001-predefined-guidance": "Edited guidance.",
                "shot-0002-prompt": "Prompt 2",
                "shot-0002-generation-mode": "smooth",
                "shot-0002-duration": "9",
                "shot-0003-prompt": "Prompt 3",
                "shot-0003-generation-mode": "smooth",
                "shot-0003-duration": "10",
            },
            follow_redirects=False,
        )
        self.assertEqual(edited.status_code, 303)
        saved = self.app.state.store.get_project(project_id)["shots"][0]["inputs"]["predefined_references"][0]
        self.assertEqual(saved["label"], "Edited hero sheet")
        self.assertEqual(saved["guidance"], "Edited guidance.")

    def test_visual_plan_failure_page_shows_llm_error_context_in_chinese(self):
        project = self.app.state.store.import_story(_story())
        project_id = project["project"]["project_id"]
        runner = create_runner(
            self.app.state.store,
            "real",
            real_visual_planner=FailingVisualPlanner(),
            real_submit=False,
        )

        with self.assertRaises(Exception):
            runner.run_step(project_id, "0001", "visual_plan")

        project_page = self.client.get(f"/projects/{project_id}")
        self.assertEqual(project_page.status_code, 200)
        self.assertIn("visual_plan_failed", project_page.text)
        self.assertIn("LLM attempt 1 submitted prompt", project_page.text)
        self.assertIn("LLM attempt 1 raw response", project_page.text)
        self.assertIn("LLM attempt 1 parse error", project_page.text)
        self.assertIn("Error context", project_page.text)
        self.assertIn("请为老公和咖啡店杯型生成视觉元素计划", project_page.text)
        self.assertIn("Visible: 老公站在咖啡店柜台前", project_page.text)
        self.assertIn("Unterminated string starting at", project_page.text)
        self.assertNotIn("\\u8001\\u516c", project_page.text)

    def test_visual_plan_can_be_edited_after_ai_completion(self):
        project = self.app.state.store.import_story(_story())
        project_id = project["project"]["project_id"]
        self.app.state.runner.run_step(project_id, "0001", "visual_plan")
        self.app.state.runner.run_step(project_id, "0001", "reference_selection")

        edit_page = self.client.get(f"/projects/{project_id}?edit_visual_plan=0001")
        self.assertEqual(edit_page.status_code, 200)
        self.assertIn("Add element", edit_page.text)
        self.assertIn("Save", edit_page.text)

        response = self.client.post(
            f"/projects/{project_id}/shots/0001/visual-plan/save",
            data={
                "element-token": ["row-1", "row-2"],
                "element-id": ["element-0001-001", ""],
                "element-name": ["Edited Pilot", "Red Mug"],
                "element-type": ["character", "object"],
                "element-status": ["should_reference", "new"],
                "element-introduced": ["0001", "0001"],
                "element-notes": ["Edited note", "Manual object."],
                "element-reason": ["Keep the pilot consistent.", "New element from review."],
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        bundle = self.app.state.store.get_project(project_id)
        shot = bundle["shots"][0]
        rows = shot["attempt"]["visual_element_status"]
        self.assertEqual(shot["state"]["status"], "visual_plan_completed")
        self.assertEqual(shot["steps"]["reference_selection"]["status"], "draft")
        self.assertEqual(shot["attempt"]["selected_references"], [])
        self.assertEqual(rows[0]["name"], "Edited Pilot")
        self.assertEqual(rows[0]["status"], "should_reference")
        self.assertEqual(rows[1]["name"], "Red Mug")
        self.assertTrue(rows[1]["element_id"].startswith("element-0001-manual-"))
        self.assertIn("manual visual element plan saved", [log["message"] for log in shot["attempt"]["logs"] if log["step"] == "planning_visual_elements"][-1])

    def test_visual_plan_edit_normalizes_non_core_types(self):
        project = self.app.state.store.import_story(_story())
        project_id = project["project"]["project_id"]
        self.app.state.runner.run_step(project_id, "0001", "visual_plan")

        response = self.client.post(
            f"/projects/{project_id}/shots/0001/visual-plan/save",
            data={
                "element-token": ["row-1"],
                "element-id": ["element-0001-001"],
                "element-name": ["Edited room"],
                "element-type": ["location"],
                "element-status": ["should_reference"],
                "element-introduced": ["0001"],
                "element-notes": ["Edited note"],
                "element-reason": ["Keep the room consistent."],
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        bundle = self.app.state.store.get_project(project_id)
        rows = bundle["shots"][0]["attempt"]["visual_element_status"]
        self.assertEqual(rows[0]["type"], "scene")

    def test_seedance_prompt_can_be_edited_and_resets_downstream_steps(self):
        project = self.app.state.store.import_story(_story())
        project_id = project["project"]["project_id"]
        runner = self.app.state.runner
        runner.run_step(project_id, "0001", "visual_plan")
        runner.run_step(project_id, "0001", "reference_selection")
        runner.run_step(project_id, "0001", "seedance_prompt")
        runner.run_step(project_id, "0001", "seedance_generation")

        response = self.client.post(
            f"/projects/{project_id}/shots/0001/seedance-prompt/save",
            data={"seedance-prompt": "Edited Seedance prompt for manual review."},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        bundle = self.app.state.store.get_project(project_id)
        shot = bundle["shots"][0]
        self.assertEqual(shot["state"]["status"], "seedance_prompt_completed")
        self.assertEqual(shot["steps"]["seedance_prompt"]["status"], "seedance_prompt_completed")
        self.assertEqual(shot["steps"]["seedance_generation"]["status"], "draft")
        self.assertEqual(shot["steps"]["keyframe_maintaining"]["status"], "draft")
        self.assertEqual(shot["attempt"]["prompt"]["submitted_prompt"], "Edited Seedance prompt for manual review.")
        self.assertEqual(shot["attempt"]["outputs"], {})
        self.assertFalse((self.app.state.store.attempt_dir(project_id, "0001", "a001") / "seedance_generation" / "request.json").exists())

    def test_reference_guidance_can_be_edited_and_resets_downstream_steps(self):
        project = self.app.state.store.import_story(_story())
        project_id = project["project"]["project_id"]
        runner = self.app.state.runner
        runner.run_shot(project_id, "0001")
        runner.run_step(project_id, "0002", "visual_plan")
        runner.run_step(project_id, "0002", "reference_selection")
        runner.run_step(project_id, "0002", "seedance_prompt")

        bundle = self.app.state.store.get_project(project_id)
        reference = bundle["shots"][1]["attempt"]["selected_references"][0]
        edit_page = self.client.get(f"/projects/{project_id}?edit_references=0002")
        self.assertEqual(edit_page.status_code, 200)
        self.assertIn("Reference guidance", edit_page.text)

        response = self.client.post(
            f"/projects/{project_id}/shots/0002/references/save",
            data={
                "reference-token": ["row-1"],
                "reference-id": [reference["reference_id"]],
                "reference-keep": ["row-1"],
                "reference-guidance": ["Manual guidance for the selected cockpit frame."],
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        bundle = self.app.state.store.get_project(project_id)
        shot = bundle["shots"][1]
        self.assertEqual(shot["state"]["status"], "reference_selection_completed")
        self.assertEqual(shot["steps"]["seedance_prompt"]["status"], "draft")
        self.assertEqual(shot["attempt"]["selected_references"][0]["reference_guidance"], "Manual guidance for the selected cockpit frame.")
        self.assertEqual(shot["attempt"]["prompt"]["submitted_prompt"], "")

    def test_produced_memory_holistic_description_can_be_edited_and_resets_later_shots(self):
        project = self.app.state.store.import_story(_story())
        project_id = project["project"]["project_id"]
        self.app.state.runner.run_shot(project_id, "0001")
        bundle = self.app.state.store.get_project(project_id)
        memory = bundle["shots"][0]["attempt"]["produced_visual_memory"][0]

        edit_page = self.client.get(f"/projects/{project_id}?edit_produced_memory=0001")
        self.assertEqual(edit_page.status_code, 200)
        self.assertIn("Holistic description", edit_page.text)

        response = self.client.post(
            f"/projects/{project_id}/shots/0001/produced-memory/save",
            data={
                "memory-token": ["row-1"],
                "memory-asset-id": [memory["asset_id"]],
                "memory-keep": ["row-1"],
                "memory-holistic-description": ["Manual holistic frame description."],
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        bundle = self.app.state.store.get_project(project_id)
        first = bundle["shots"][0]
        self.assertEqual(first["state"]["status"], "completed")
        self.assertEqual(first["attempt"]["produced_visual_memory"][0]["holistic_description"], "Manual holistic frame description.")
        self.assertEqual(bundle["shots"][1]["state"]["status"], "draft")

    def test_visual_plan_reflect_route_runs_background_job(self):
        project = self.app.state.store.import_story(_story())
        project_id = project["project"]["project_id"]
        self.app.state.runner.run_step(project_id, "0001", "visual_plan")

        with patch("videogen_notebook.runner.reflect_visual_plan_with_llm", side_effect=_fake_reflection):
            response = self.client.post(
                f"/projects/{project_id}/shots/0001/visual-plan/reflect",
                follow_redirects=False,
            )
            self.assertEqual(response.status_code, 303)
            _wait_for_visual_reflection(self.app.state.store, project_id, "0001")

        bundle = self.app.state.store.get_project(project_id)
        first = bundle["shots"][0]
        page = self.client.get(f"/projects/{project_id}")

        self.assertEqual(first["attempt"]["visual_element_status"][0]["name"], "Reflected element")
        self.assertIn("Reflection result", page.text)
        self.assertIn("fake reflection applied", page.text)

    def test_element_names_filter_uses_selection_labels(self):
        selection = {
            "should_reference": [
                {"id": "element-0001", "name": "Pilot"},
                {"id": "element-0002", "name": "Control panel"},
            ],
            "optional_or_uncertain": [
                {"id": "element-0003", "name": "Window"},
            ],
        }

        self.assertEqual(
            _element_names(["element-0002", "element-0003"], selection),
            "Control panel, Window",
        )
        self.assertEqual(_element_names(["element-missing"], selection), "element-missing")
        self.assertEqual(_element_names([], selection), "none")

    def test_project_page_shows_shot_stop_action_when_running(self):
        project = self.app.state.store.import_story(_story())
        project_id = project["project"]["project_id"]
        shot = project["shots"][0]
        shot["state"]["status"] = "visual_plan_running"
        self.app.state.store.write_shot(project_id, shot)
        self.app.state.store.update_project_json(project_id, {"status": "running", "active_shot_id": shot["shot_id"]})

        project_page = self.client.get(f"/projects/{project_id}")

        self.assertEqual(project_page.status_code, 200)
        self.assertIn("Stop Shot", project_page.text)
        home = self.client.get("/")
        self.assertIn("Active shot 0001", home.text)

    def test_home_import_accepts_multiple_json_files_and_stays_on_home(self):
        story_a = _story()
        story_a["story_name"] = "Batch A"
        story_b = _story()
        story_b["story_name"] = "Batch B"

        response = self.client.post(
            "/projects/import",
            files=[
                ("story_file", ("batch-a.json", json.dumps(story_a), "application/json")),
                ("story_file", ("batch-b.json", json.dumps(story_b), "application/json")),
            ],
            follow_redirects=False,
        )
        home = self.client.get("/")

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/")
        self.assertIn("Batch A", home.text)
        self.assertIn("Batch B", home.text)

    def test_home_groups_projects_by_created_date_open_by_default(self):
        older = self.app.state.store.import_story(_story(), name="Older home project")
        newer = self.app.state.store.import_story(_story(), name="Newer home project")
        older_path = self.app.state.store.project_dir(older["project"]["project_id"]) / "project.json"
        newer_path = self.app.state.store.project_dir(newer["project"]["project_id"]) / "project.json"
        older_json = read_json(older_path)
        newer_json = read_json(newer_path)
        older_json["created_at"] = "2026-07-01T00:00:00Z"
        older_json["updated_at"] = "2026-07-14T00:00:00Z"
        newer_json["created_at"] = "2026-07-13T00:00:00Z"
        newer_json["updated_at"] = "2026-07-13T00:00:00Z"
        write_json_atomic(older_path, older_json)
        write_json_atomic(newer_path, newer_json)

        home = self.client.get("/")

        self.assertIn('<details class="project-date-group" open>', home.text)
        self.assertLess(home.text.index("2026-07-13"), home.text.index("2026-07-01"))
        self.assertLess(home.text.index("Newer home project"), home.text.index("Older home project"))

    def test_home_default_settings_apply_to_future_imports(self):
        response = self.client.post(
            "/settings/defaults",
            data={
                "sink-frame-count": "1",
                "max-retrieved-frames": "3",
                "selection-mode": "static_top_k",
                "default-duration": "12",
                "default-non-cut-mode": "default",
                "seedance-resolution": "1080p",
                "seedance-ratio": "9:16",
                "audio": "on",
                "auto-run-step-review-delay": "30",
                "algorithm-step-max-attempts": "4",
                "smooth-reference-seconds": "3",
                "auto-reflect-visual-plan": "on",
                "force-animation-style": "on",
                "auto-submit-eval": "on",
            },
            follow_redirects=False,
        )
        story = {
            "story_name": "Uses Defaults",
            "scenes": [{"scene_num": 1, "video_prompts": ["Prompt 1", "Prompt 2"], "cut": [True, False]}],
        }
        imported = self.client.post(
            "/projects/import",
            files=[("story_file", ("story.json", json.dumps(story), "application/json"))],
            follow_redirects=False,
        )
        project = next(item for item in self.app.state.store.list_projects() if item["name"] == "Uses Defaults")
        bundle = self.app.state.store.get_project(project["project_id"])

        self.assertEqual(response.status_code, 303)
        self.assertEqual(imported.status_code, 303)
        self.assertEqual(bundle["settings"]["generation"]["default_duration_seconds"], 12)
        self.assertEqual(bundle["settings"]["generation"]["default_non_cut_mode"], "default")
        self.assertEqual(bundle["settings"]["generation"]["force_animation_style"], True)
        self.assertEqual(bundle["settings"]["visual_element_memory"]["sink_frame_count"], 1)
        self.assertEqual(bundle["settings"]["visual_element_memory"]["selection_mode"], "static_top_k")
        self.assertEqual(bundle["settings"]["seedance"]["resolution"], "1080p")
        self.assertEqual(bundle["settings"]["seedance"]["ratio"], "9:16")
        self.assertEqual(bundle["settings"]["evaluation"]["auto_submit_eval"], True)
        self.assertEqual(bundle["shots"][0]["inputs"]["duration_seconds"], 12)
        self.assertEqual(bundle["shots"][1]["inputs"]["generation_mode"], "default")

    def test_home_project_row_has_run_action(self):
        project = self.app.state.store.import_story(_story())
        project_id = project["project"]["project_id"]
        self.app.state.store.save_project(
            project_id,
            settings_update={"generation": {"auto_run_step_review_delay_seconds": 0, "auto_reflect_visual_plan": False}},
        )

        home = self.client.get("/")
        self.assertIn(f'action="/projects/{project_id}/run-all"', home.text)
        response = self.client.post(f"/projects/{project_id}/run-all", follow_redirects=False)

        self.assertEqual(response.status_code, 303)
        _wait_for_project_status(self.app.state.store, f"/projects/{project_id}", "completed")

    def test_create_app_recovers_orphan_running_project(self):
        workspace = Path(self.tempdir.name) / "recover"
        store = ProjectStore(workspace)
        project = store.import_story(_story())
        project_id = project["project"]["project_id"]
        shot = project["shots"][0]
        shot["state"]["status"] = "running"
        shot["state"]["current_attempt_id"] = "a001"
        store.write_shot(project_id, shot)
        store.update_project_json(project_id, {"status": "running", "active_shot_id": "0001"})

        app = create_app(str(workspace))
        recovered = app.state.store.get_project(project_id)

        self.assertEqual(recovered["project"]["status"], "interrupted")
        self.assertEqual(recovered["shots"][0]["state"]["status"], "interrupted")

    def test_run_all_route_renders_attempt_tables(self):
        created = self.client.post("/projects", data={"name": "Demo"}, follow_redirects=False)
        project_path = created.headers["location"]
        project_id = project_path.rstrip("/").split("/")[-1]
        self.app.state.store.save_project(
            project_id,
            settings_update={"generation": {"auto_run_step_review_delay_seconds": 0, "auto_reflect_visual_plan": False}},
        )
        run = self.client.post(f"{project_path}/run-all", follow_redirects=False)
        self.assertEqual(run.status_code, 303)
        _wait_for_project_status(self.app.state.store, project_path, "completed")

        project_page = self.client.get(project_path)
        self.assertEqual(project_page.status_code, 200)
        self.assertIn("completed", project_page.text)
        self.assertIn("element-0001-001", project_page.text)
        self.assertIn("vid_0001_a001", project_page.text)
        self.assertIn("vid_final_0001", project_page.text)
        self.assertIn("Backend", project_page.text)
        self.assertIn("fake", project_page.text)
        self.assertIn("Visual Elements Plan Details", project_page.text)
        self.assertIn("Historical Reference Selection Details", project_page.text)
        self.assertIn("Seedance Attempt Details", project_page.text)
        self.assertIn("Keyframe And VLM Details", project_page.text)
        self.assertIn("fake visual element status generated", project_page.text)
        self.assertIn("thumb-img", project_page.text)
        self.assertIn("video-preview", project_page.text)
        self.assertIn("Assembly logs", project_page.text)
        home = self.client.get("/")
        self.assertIn("project-thumb", home.text)
        self.assertIn("img_0001_a001_001", home.text)

    def test_asset_and_reset_routes(self):
        project = self.app.state.store.import_story(_story())
        project_path = f"/projects/{project['project']['project_id']}"
        self.app.state.store.save_project(
            project["project"]["project_id"],
            settings_update={"generation": {"auto_run_step_review_delay_seconds": 0, "auto_reflect_visual_plan": False}},
        )
        self.client.post(f"{project_path}/run-all", follow_redirects=False)
        _wait_for_project_status(self.app.state.store, project_path, "completed")

        asset = self.client.get(f"{project_path}/assets/vid_0001_a001")
        self.assertEqual(asset.status_code, 200)
        self.assertIn(b"fake video placeholder", asset.content)

        project_dir = self.app.state.store.project_dir(project["project"]["project_id"])
        visual_file = project_dir / "memory" / "visual_element_memory" / "probe.jpg"
        visual_file.parent.mkdir(parents=True, exist_ok=True)
        visual_file.write_bytes(b"visual probe")
        visual = self.client.get(f"{project_path}/files/memory/visual_element_memory/probe.jpg")
        self.assertEqual(visual.status_code, 200)
        self.assertEqual(visual.content, b"visual probe")

        reset = self.client.post(f"{project_path}/shots/0002/reset-from", follow_redirects=False)
        self.assertEqual(reset.status_code, 303)
        project_page = self.client.get(project_path)
        self.assertEqual(project_page.status_code, 200)
        self.assertIn("1 / 3 completed", project_page.text)
        self.assertIn("vid_0001_a001", project_page.text)

    def test_real_backend_app_records_visible_dry_run(self):
        with patch.dict("os.environ", {"VIDEOGEN_NOTEBOOK_REAL_SUBMIT": "0"}):
            app = create_app(self.tempdir.name, runner_backend="real")
        client = TestClient(app)
        try:
            created = client.post("/projects", data={"name": "Demo"}, follow_redirects=False)
            project_path = created.headers["location"]
            project_id = project_path.rstrip("/").split("/")[-1]
            app.state.store.save_project(
                project_id,
                settings_update={"generation": {"auto_run_step_review_delay_seconds": 0, "auto_reflect_visual_plan": False}},
            )

            run = client.post(f"{project_path}/shots/0001/run", follow_redirects=False)
            self.assertEqual(run.status_code, 303)
            _wait_for_project_status(app.state.store, project_path, "completed")
            project_page = client.get(project_path)
            self.assertEqual(project_page.status_code, 200)
            self.assertIn("completed", project_page.text)
            self.assertIn("Backend", project_page.text)
            self.assertIn("real", project_page.text)
            self.assertIn("dry-run mode skipped Seedance task creation", project_page.text)
        finally:
            app.state.jobs.executor.shutdown(wait=True, cancel_futures=True)


def _story():
    return {
        "story_name": "Demo Story",
        "scenes": [
            {
                "scene_num": 1,
                "video_prompts": ["Prompt 1", "Prompt 2", "Prompt 3"],
                "cut": [True, False, False],
                "durations": [8, 9, 10],
            }
        ],
    }


def _fake_reflection(request):
    rows = []
    for index, row in enumerate(request.current_visual_plan):
        if index == 0:
            rows.append(
                {
                    "element_id": row["element_id"],
                    "action": "modify",
                    "change_fields": ["name", "notes"],
                    "suggested": {
                        "name": "Reflected element",
                        "type": row.get("type") or "scene",
                        "status": row.get("status") or "new",
                        "notes": "Reflection added stable visual notes.",
                        "reason": row.get("reason") or "Reflection kept the status.",
                    },
                    "rationale": "Fake reflection modifies the first row.",
                }
            )
        else:
            rows.append(
                {
                    "element_id": row["element_id"],
                    "action": "no_change",
                    "change_fields": [],
                    "suggested": {
                        "name": row.get("name", ""),
                        "type": row.get("type", "scene"),
                        "status": row.get("status", "optional_or_uncertain"),
                        "notes": row.get("notes", ""),
                        "reason": row.get("reason", ""),
                    },
                    "rationale": "No change needed.",
                }
            )
    return VisualPlanReflectionResult(
        summary="fake reflection applied",
        rows=rows,
        warnings=[],
        raw_response='{"summary":"fake reflection applied"}',
        prompt="fake reflection prompt",
        metadata={"usage": {"total_tokens": 0}, "fake": True},
    )


def _wait_for_project_status(store: ProjectStore, project_path: str, status) -> None:
    project_id = project_path.rstrip("/").split("/")[-1]
    deadline = time.time() + 5.0
    last = None
    expected = {status} if isinstance(status, str) else set(status)
    while time.time() < deadline:
        bundle = store.get_project(project_id)
        last = bundle["project"].get("status")
        if last in expected:
            return
        time.sleep(0.05)
    raise AssertionError(f"Project {project_id} did not reach {sorted(expected)}; last={last}")


def _wait_for_visual_reflection(store: ProjectStore, project_id: str, shot_id: str) -> None:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        bundle = store.get_project(project_id)
        shot = next(item for item in bundle["shots"] if item["shot_id"] == shot_id)
        details = (shot.get("attempt") or {}).get("visual_element_details") or {}
        if details.get("reflection"):
            return
        time.sleep(0.05)
    raise AssertionError(f"Shot {shot_id} did not receive visual plan reflection")


if __name__ == "__main__":
    unittest.main()
