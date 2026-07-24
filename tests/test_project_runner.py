import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from storymem_web.project_runner import ProjectRunner
from storymem_web.repository import ProjectRepository


class SmoothTransitionError(RuntimeError):
    pass


class FakeSeedanceClient:
    def __init__(self):
        self.submissions = []
        self.queried_tasks = []

    def create_task(self, content, **kwargs):
        task_id = f"task-{len(self.submissions) + 1}"
        self.submissions.append({"task_id": task_id, "content": content, "options": kwargs})
        return {"id": task_id}

    def wait_task(self, task_id, **kwargs):
        if kwargs.get("should_cancel") and kwargs["should_cancel"]():
            raise AssertionError("unexpected cancellation")
        return {
            "id": task_id,
            "status": "succeeded",
            "content": {"video_url": f"https://example.test/{task_id}.mp4"},
            "usage": {"total_tokens": 100_000, "completion_tokens": 100_000},
            "duration": 8,
            "resolution": "720p",
            "model": "fake-seedance",
        }

    def get_task(self, task_id):
        self.queried_tasks.append(task_id)
        return {
            "id": task_id,
            "status": "succeeded",
            "content": {
                "video_url": f"https://example.test/{task_id}.mp4",
                "last_frame_url": f"https://example.test/{task_id}-last-frame.jpeg",
            },
        }

    def download_video(self, url, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(url.encode("utf-8"))


class FailingSeedanceClient(FakeSeedanceClient):
    def create_task(self, content, **kwargs):
        raise RuntimeError("submission failed")


def fake_memory_selector(
    config, store, story_script, shot, memory_query_generator, memory_policy=None
):
    records = []
    for path in store.memory_keyframes():
        records.append(
            {
                "source_path": path,
                "file": Path(path).name,
                "label": "memory_keyframe",
                "roles": ["recent_window_memory"],
                "source_scene_num": 1,
                "source_shot_num": 1,
                "source_prompt": "first",
            }
        )
    return records


def fake_memory_extractor(
    video_path,
    existing_memory_paths=None,
    keyframe_profile=None,
    keyframe_config_path=None,
):
    base = Path(video_path).with_suffix("")
    keyframe = Path(f"{base}_keyframe0.jpg")
    ending = Path(video_path).parent / "last_frame.jpg"
    motion = Path(video_path).parent / "motion_frames.mp4"
    keyframe.write_bytes(b"jpeg")
    ending.write_bytes(b"jpeg")
    motion.write_bytes(b"mp4")
    return {
        "keyframe_indices": [0],
        "keyframe_paths": [str(keyframe)],
        "last_frame_path": str(ending),
        "motion_frames_path": str(motion),
    }


def fake_concat(videos, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"".join(Path(path).read_bytes() for path in videos))
    return str(output)


class RecordingMemoryExtractor:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        video_path,
        existing_memory_paths=None,
        keyframe_profile=None,
        keyframe_config_path=None,
    ):
        self.calls.append(
            {
                "video_path": video_path,
                "existing_memory_paths": list(existing_memory_paths or []),
                "keyframe_profile": keyframe_profile,
                "keyframe_config_path": keyframe_config_path,
            }
        )
        return fake_memory_extractor(
            video_path,
            existing_memory_paths=existing_memory_paths,
            keyframe_profile=keyframe_profile,
            keyframe_config_path=keyframe_config_path,
        )


class FakePublisher:
    def __init__(self, url="https://public.test/smooth-tail.mp4"):
        self.url = url
        self.paths = []

    def publish(self, path):
        self.paths.append(str(path))
        return self.url


def fake_tail_extractor(source, output, *, seconds):
    Path(output).write_bytes(b"tail")
    return str(output)


class FakeSmoothAssembler:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def __call__(self, clips, output_path):
        self.calls.append(clips)
        if self.error:
            raise self.error
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"smooth assembly")
        return str(output)


class FakeVisualMemory:
    def __init__(self, config, story_script, shots):
        self.config = config
        self.story_script = story_script
        self.shots = shots
        self.snapshot_payload = {}
        self.records = []

    def load_snapshot(self, snapshot):
        self.snapshot_payload = snapshot

    def prepare_references_for_shot(self, shot_index, shot):
        references = []
        selected = []
        annotations = self.snapshot_payload.get("annotations") or []
        if annotations:
            source_path = annotations[0]["frame_path"]
            metadata = {
                "source_path": source_path,
                "file": Path(source_path).name,
                "label": "visual_element_memory",
                "roles": ["visual_element_memory"],
                "source_scene_num": 1,
                "source_shot_num": 1,
                "score": 2.0,
                "visual_element_selection": {
                    "score": 2.0,
                    "newly_covered_element_ids": ["element-0001"],
                    "already_covered_element_ids": [],
                    "should_reference": [{"id": "element-0001", "name": "Pilot"}],
                    "should_exclude": [],
                    "optional_or_uncertain": [],
                },
                "reference_intent": "应参考：Pilot",
            }
            selected.append(metadata)
            references.append(
                {
                    "type": "image_url",
                    "role": "reference_image",
                    "image_url": {"url": source_path},
                    "metadata": metadata,
                }
            )
        decision = {
            "existing_element_states": [
                {
                    "id": "element-0001",
                    "state": "should_reference",
                    "reason": "continue the same character",
                }
            ]
            if annotations
            else [],
            "new_elements": []
            if annotations
            else [{"name": "Pilot", "type": "character", "notes": "main role"}],
            "shot_notes": [],
        }
        inserted = []
        if not annotations:
            inserted = [
                {
                    "id": "element-0001",
                    "name": "Pilot",
                    "type": "character",
                    "introduced_at": "shot-0001",
                    "notes": "main role",
                }
            ]
        record = {
            "decision": decision,
            "inserted": inserted,
            "selected_references": selected,
            "llm_attempts": [{"raw_response": "{}"}],
        }
        self.records.append(record)
        return SimpleNamespace(
            selected_references=references,
            prompt_context=f"VISUAL PROMPT: {shot.prompt}",
            report_record={
                "visual_element_plan": {
                    "should_reference": [],
                    "should_exclude": [],
                    "optional_or_uncertain": [],
                    "new_elements": inserted,
                },
                "visual_element_decision": decision,
                "visual_element_inserted": inserted,
                "visual_element_selected_references": selected,
                "visual_element_retrieved_references": selected,
                "visual_element_sink_references": [],
            },
        )

    def annotate_completed_shot(self, shot_index, shot, keyframe_paths):
        annotations = [
            {
                "source_shot_id": f"shot-{shot_index:04d}",
                "source_scene_num": shot.scene_num,
                "source_shot_num": shot.shot_num,
                "frame_path": path,
                "width": 16,
                "height": 9,
                "elements": [
                    {
                        "id": "element-0001",
                        "name": "Pilot",
                        "type": "character",
                        "bbox_1000": [0, 0, 500, 500],
                        "bbox_pixels": [0, 0, 8, 5],
                    }
                ],
            }
            for path in keyframe_paths
        ]
        self.snapshot_payload = {
            "registry": [
                {
                    "id": "element-0001",
                    "name": "Pilot",
                    "type": "character",
                    "introduced_at": "shot-0001",
                    "notes": "main role",
                }
            ],
            "annotations": annotations,
            "records": self.records,
        }
        if self.records:
            self.records[-1]["current_annotation_visualizations"] = []
        return annotations

    def snapshot(self):
        return self.snapshot_payload


class ProjectRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.repository = ProjectRepository(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_run_all_executes_in_order_and_uses_produced_memory(self):
        project = self.repository.create_project(
            {
                "story_name": "Notebook run",
                "scenes": [
                    {
                        "scene_num": 1,
                        "video_prompts": ["first", "second"],
                        "cut": [True, False],
                    }
                ],
            },
            generation_config={"pipeline_version": "classic", "prompt_retrieval": False, "enhanced_text_prompt": True},
        )
        job = self.repository.create_job(project["project_id"], "all")
        self.repository.update_shot(
            project["shots"][1]["shot_id"],
            generation_mode="default",
            duration_seconds=12,
            memory_retrieve=False,
            memory_recent=True,
        )
        client = FakeSeedanceClient()
        memory_extractor = RecordingMemoryExtractor()
        runner = ProjectRunner(
            self.repository,
            job["job_id"],
            client=client,
            memory_selector=fake_memory_selector,
            memory_extractor=memory_extractor,
            concat_function=fake_concat,
        )

        completed = runner.run()

        self.assertEqual(completed["status"], "completed")
        self.assertEqual(len(client.submissions), 2)
        self.assertEqual(
            [item["options"]["duration"] for item in client.submissions], [8, 12]
        )
        self.assertEqual(
            [call["keyframe_profile"] for call in memory_extractor.calls],
            ["storymem_original", "storymem_original"],
        )
        self.assertEqual(
            [call["keyframe_config_path"] for call in memory_extractor.calls],
            [None, None],
        )
        self.assertEqual(client.submissions[0]["content"][0]["text"], "first")
        second_text = client.submissions[1]["content"][0]["text"]
        self.assertIn("参考图 Image 1 是上一段视频的最后一帧", second_text)
        self.assertIn("新视频的第一帧必须与 Image 1 保持一致", second_text)
        self.assertTrue(second_text.rstrip().endswith("second"))
        self.assertNotIn("FULL STORY SHOT PLAN", second_text)
        self.assertNotIn("REFERENCE IMAGE GUIDE", second_text)
        second_images = [item for item in client.submissions[1]["content"] if item["type"] == "image_url"]
        self.assertEqual(len(second_images), 2)
        self.assertEqual(client.queried_tasks, ["task-1"])
        self.assertEqual(second_images[0]["role"], "reference_image")
        self.assertEqual(
            second_images[0]["image_url"]["url"],
            "https://example.test/task-1-last-frame.jpeg",
        )
        self.assertEqual(
            second_images[0]["metadata"]["roles"], ["previous_last_frame"]
        )
        self.assertEqual(
            second_images[0]["metadata"]["input_origin"],
            "seedance_last_frame_url",
        )
        self.assertEqual(
            second_images[1]["metadata"]["roles"], ["recent_window_memory"]
        )
        reloaded = self.repository.get_project(project["project_id"])
        self.assertEqual([shot["state"] for shot in reloaded["shots"]], ["completed", "completed"])
        self.assertTrue(Path(reloaded["current_final_video"]).exists())
        assets = self.repository.list_memory_assets(project["project_id"])
        self.assertEqual(
            [asset["asset_type"] for asset in assets],
            ["retrieval_keyframe", "ending_frame", "retrieval_keyframe", "ending_frame"],
        )
        cost = self.repository.cost_summary(project["project_id"])
        self.assertEqual(cost["all_attempts"]["tasks"], 2)
        self.assertEqual(cost["all_attempts"]["duration_seconds"], 16)
        attempts = self.repository.list_attempts(project["project_id"])
        second_attempt = next(
            item for item in attempts if item["shot_id"] == project["shots"][1]["shot_id"]
        )
        self.assertEqual(second_attempt["input_snapshot"]["duration_seconds"], 12)

    def test_last_frame_mode_uses_seedance_first_frame_without_changing_memory_selection(self):
        project = self.repository.create_project(
            {
                "story_name": "First-frame experiment",
                "scenes": [
                    {
                        "scene_num": 1,
                        "video_prompts": ["first", "second"],
                        "cut": [True, False],
                    }
                ],
            },
            generation_config={"pipeline_version": "classic", "prompt_retrieval": False, "enhanced_text_prompt": True},
        )
        second = project["shots"][1]
        self.repository.update_shot(
            second["shot_id"],
            generation_mode="last_frame_only",
            memory_retrieve=False,
            memory_recent=True,
        )
        job = self.repository.create_job(project["project_id"], "all")
        client = FakeSeedanceClient()
        runner = ProjectRunner(
            self.repository,
            job["job_id"],
            client=client,
            memory_selector=fake_memory_selector,
            memory_extractor=fake_memory_extractor,
            concat_function=fake_concat,
        )

        runner.run()

        second_content = client.submissions[1]["content"]
        second_images = [item for item in second_content if item["type"] == "image_url"]
        self.assertEqual(len(second_images), 1)
        self.assertEqual(second_images[0]["role"], "first_frame")
        self.assertEqual(client.queried_tasks, ["task-1"])
        self.assertEqual(
            second_images[0]["image_url"]["url"],
            "https://example.test/task-1-last-frame.jpeg",
        )
        self.assertEqual(
            second_images[0]["metadata"]["roles"], ["previous_last_frame"]
        )
        self.assertEqual(
            second_images[0]["metadata"]["input_origin"],
            "seedance_last_frame_url",
        )
        attempts = self.repository.list_attempts(project["project_id"])
        second_attempt = next(item for item in attempts if item["shot_id"] == second["shot_id"])
        self.assertEqual(
            second_attempt["input_snapshot"]["generation_mode"], "last_frame_only"
        )
        self.assertEqual(len(second_attempt["input_snapshot"]["references"]), 1)
        self.assertEqual(len(second_attempt["memory_selection"]["references"]), 2)
        input_path = self.repository.attempt_dir(
            project["project_id"], second["shot_id"], second_attempt["attempt_id"]
        ) / "input.json"
        self.assertNotIn(
            "https://example.test/task-1-last-frame.jpeg",
            input_path.read_text(encoding="utf-8"),
        )

    def test_smooth_mode_replaces_tail_image_with_video_and_keeps_memory(self):
        project = self.repository.create_project(
            {
                "story_name": "Smooth generation",
                "scenes": [
                    {
                        "scene_num": 1,
                        "video_prompts": ["first", "continue moving"],
                        "cut": [True, False],
                    }
                ],
            },
            generation_config={"pipeline_version": "classic", "prompt_retrieval": False, "enhanced_text_prompt": True},
        )
        second = project["shots"][1]
        self.repository.update_shot(
            second["shot_id"],
            generation_mode="smooth",
            memory_retrieve=False,
            memory_recent=True,
        )
        client = FakeSeedanceClient()
        publisher = FakePublisher()
        assembler = FakeSmoothAssembler()
        job = self.repository.create_job(project["project_id"], "all")
        ProjectRunner(
            self.repository,
            job["job_id"],
            client=client,
            memory_selector=fake_memory_selector,
            memory_extractor=fake_memory_extractor,
            concat_function=fake_concat,
            reference_video_publisher=publisher,
            tail_extractor=fake_tail_extractor,
            smooth_assembler=assembler,
        ).run()

        content = client.submissions[1]["content"]
        self.assertEqual([item["type"] for item in content], ["text", "video_url", "image_url"])
        self.assertEqual(content[1]["role"], "reference_video")
        self.assertEqual(content[1]["video_url"]["url"], publisher.url)
        self.assertEqual(content[2]["metadata"]["roles"], ["recent_window_memory"])
        self.assertTrue(content[0]["text"].startswith("向后延长 Video 1"))
        self.assertIn("不要把 Video 1 当作普通参考图或重新生成一个相似开头", content[0]["text"])
        self.assertEqual(client.queried_tasks, [])
        self.assertEqual(len(assembler.calls), 1)

        second_attempt = self.repository.get_attempt(
            self.repository.get_shot(second["shot_id"])["current_attempt_id"]
        )
        self.assertEqual(second_attempt["input_snapshot"]["generation_mode"], "smooth")
        self.assertEqual(
            second_attempt["input_snapshot"]["memory_policy"],
            {"sink": True, "retrieve": False, "recent": True},
        )
        self.assertEqual(len(second_attempt["memory_selection"]["references"]), 2)
        attempt_dir = self.repository.attempt_dir(
            project["project_id"], second["shot_id"], second_attempt["attempt_id"]
        )
        persisted = "\n".join(
            path.read_text(encoding="utf-8")
            for path in attempt_dir.glob("*.json")
        )
        self.assertNotIn(publisher.url, persisted)
        for database_file in self.repository.workspace.glob("storymem_web.sqlite3*"):
            self.assertNotIn(publisher.url.encode(), database_file.read_bytes())

    def test_smooth_postprocess_retry_does_not_resubmit_seedance(self):
        project = self.repository.create_project(
            {
                "story_name": "Smooth retry",
                "scenes": [
                    {
                        "scene_num": 1,
                        "video_prompts": ["first", "second"],
                        "cut": [True, False],
                    }
                ],
            }
        )
        first, second = project["shots"]
        first_attempt = self.repository.create_attempt(first["shot_id"], {"generation_mode": "default"})
        first_output = self.repository.attempt_dir(
            project["project_id"], first["shot_id"], first_attempt["attempt_id"]
        ) / "first.mp4"
        first_output.parent.mkdir(parents=True)
        first_output.write_bytes(b"first raw")
        self.repository.update_attempt(
            first_attempt["attempt_id"], status="completed", output_video=str(first_output), finished=True
        )
        self.repository.update_shot(second["shot_id"], generation_mode="smooth")
        second_attempt = self.repository.create_attempt(second["shot_id"], {"generation_mode": "smooth"})
        second_output = self.repository.attempt_dir(
            project["project_id"], second["shot_id"], second_attempt["attempt_id"]
        ) / "second.mp4"
        second_output.parent.mkdir(parents=True)
        second_output.write_bytes(b"second raw")
        self.repository.update_attempt(
            second_attempt["attempt_id"],
            status="failed",
            output_video=str(second_output),
            error={"type": "SmoothTransitionError", "message": "RIFE failed"},
            finished=True,
        )

        job = self.repository.create_job(project["project_id"], "smooth_postprocess", second["shot_id"])
        client = FakeSeedanceClient()
        assembler = FakeSmoothAssembler()
        result = ProjectRunner(
            self.repository,
            job["job_id"],
            client=client,
            smooth_assembler=assembler,
        ).run()

        self.assertEqual(result["status"], "completed")
        self.assertEqual(client.submissions, [])
        self.assertEqual(len(assembler.calls), 1)
        self.assertEqual(self.repository.get_attempt(second_attempt["attempt_id"])["status"], "completed")

    def test_failure_marks_the_in_progress_attempt_and_job(self):
        project = self.repository.create_empty_project(
            "Failure",
            generation_config={"pipeline_version": "classic"},
        )
        job = self.repository.create_job(project["project_id"], "all")
        runner = ProjectRunner(
            self.repository,
            job["job_id"],
            client=FailingSeedanceClient(),
            memory_selector=fake_memory_selector,
            memory_extractor=fake_memory_extractor,
            concat_function=fake_concat,
        )

        with self.assertLogs(level="ERROR") as captured:
            with self.assertRaisesRegex(RuntimeError, "submission failed"):
                runner.run()
        self.assertIn("Project runner failed", "\n".join(captured.output))

        reloaded = self.repository.get_project(project["project_id"])
        self.assertEqual(reloaded["status"], "failed")
        self.assertEqual(reloaded["shots"][0]["state"], "failed")
        attempts = self.repository.list_attempts(project["project_id"])
        self.assertEqual(attempts[0]["status"], "failed")
        self.assertEqual(attempts[0]["error"]["type"], "RuntimeError")
        self.assertEqual(self.repository.get_job(job["job_id"])["status"], "failed")

    def test_keyframe_retry_repairs_existing_attempt_without_seedance(self):
        project = self.repository.create_empty_project(
            "Repair post-processing",
            generation_config={"pipeline_version": "classic"},
        )
        shot = project["shots"][0]
        attempt = self.repository.create_attempt(shot["shot_id"], {"references": []})
        attempt_dir = self.repository.attempt_dir(
            project["project_id"],
            shot["shot_id"],
            attempt["attempt_id"],
        )
        attempt_dir.mkdir(parents=True)
        output = attempt_dir / "01_01.mp4"
        output.write_bytes(b"generated video")
        self.repository.update_attempt(
            attempt["attempt_id"],
            status="failed",
            task_id="paid-task",
            output_video=str(output),
            usage={"total_tokens": 173_700, "duration_seconds": 8},
            error={"type": "OutOfMemoryError", "message": "CUDA out of memory"},
            finished=True,
        )
        job = self.repository.create_job(
            project["project_id"],
            "keyframes",
            shot["shot_id"],
        )
        client = FakeSeedanceClient()
        runner = ProjectRunner(
            self.repository,
            job["job_id"],
            client=client,
            memory_extractor=fake_memory_extractor,
            concat_function=fake_concat,
        )

        completed = runner.run()

        self.assertEqual(completed["status"], "completed")
        self.assertEqual(client.submissions, [])
        attempts = self.repository.list_attempts(project["project_id"])
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["attempt_id"], attempt["attempt_id"])
        self.assertEqual(attempts[0]["status"], "completed")
        self.assertEqual(attempts[0]["error"], {})
        self.assertEqual(attempts[0]["task_id"], "paid-task")
        self.assertEqual(self.repository.get_job(job["job_id"])["status"], "completed")
        self.assertTrue(Path(completed["current_final_video"]).exists())
        self.assertEqual(len(self.repository.list_memory_assets(project["project_id"])), 2)
        cost = self.repository.cost_summary(project["project_id"])
        self.assertEqual(cost["all_attempts"]["tasks"], 1)
        self.assertEqual(cost["all_attempts"]["duration_seconds"], 8)

    def test_visual_element_project_uses_visual_prompt_and_persists_stage_outputs(self):
        project = self.repository.create_project(
            {
                "story_name": "Visual run",
                "scenes": [
                    {
                        "scene_num": 1,
                        "video_prompts": ["introduce pilot", "pilot returns"],
                        "cut": [True, True],
                    }
                ],
            },
            generation_config={
                "pipeline_version": "visual_element_v1",
                "prompt_retrieval": False,
                "enhanced_text_prompt": True,
                "visual_element_sink_frame_count": 1,
                "visual_element_max_retrieved_frames": 1,
            },
        )
        job = self.repository.create_job(project["project_id"], "all")
        client = FakeSeedanceClient()
        runner = ProjectRunner(
            self.repository,
            job["job_id"],
            client=client,
            memory_selector=fake_memory_selector,
            memory_extractor=fake_memory_extractor,
            concat_function=fake_concat,
            visual_memory_factory=FakeVisualMemory,
        )

        completed = runner.run()

        self.assertEqual(completed["status"], "completed")
        self.assertEqual(client.submissions[0]["content"][0]["text"], "VISUAL PROMPT: introduce pilot")
        self.assertEqual(client.submissions[1]["content"][0]["text"], "VISUAL PROMPT: pilot returns")
        second_images = [item for item in client.submissions[1]["content"] if item["type"] == "image_url"]
        self.assertEqual(len(second_images), 1)
        self.assertEqual(second_images[0]["metadata"]["roles"], ["visual_element_memory"])
        second_attempt = self.repository.get_attempt(
            self.repository.get_shot(project["shots"][1]["shot_id"])["current_attempt_id"]
        )
        visual = second_attempt["memory_selection"]["visual_element"]
        self.assertEqual(second_attempt["memory_selection"]["policy"], "visual_element_v1")
        self.assertEqual(visual["settings"]["sink_frame_count"], 1)
        self.assertIn("state_after", visual)
        self.assertEqual(len(visual["current_annotations"]), 1)


if __name__ == "__main__":
    unittest.main()
