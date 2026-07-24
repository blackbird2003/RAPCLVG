import tempfile
import unittest
from pathlib import Path

from seedance_client import (
    SeedanceCancelledError,
    SeedanceClient,
    SeedanceError,
    extract_last_frame_url,
)
from storymem_seedance.executor import complete_shot, prepare_shot, submit_shot
from storymem_seedance.models import RunConfig, ShotSpec


class FakeSeedanceClient:
    def __init__(self):
        self.created_content = None

    def create_task(self, content, **kwargs):
        self.created_content = content
        self.created_options = kwargs
        return {"id": "task-1"}

    def wait_task(self, task_id, **kwargs):
        self.waited_task_id = task_id
        return {
            "id": task_id,
            "status": "succeeded",
            "content": {"video_url": "https://example.test/video.mp4"},
            "usage": {"total_tokens": 10},
        }

    def download_video(self, url, output_path):
        self.downloaded_url = url
        Path(output_path).write_bytes(b"video")


class ShotExecutorTests(unittest.TestCase):
    def test_extract_last_frame_url_uses_the_explicit_task_output(self):
        response = {
            "content": {
                "video_url": "https://example.test/video.mp4",
                "last_frame_url": "https://example.test/original-last-frame.jpeg",
            }
        }

        self.assertEqual(
            extract_last_frame_url(response),
            "https://example.test/original-last-frame.jpeg",
        )
        with self.assertRaises(SeedanceError):
            extract_last_frame_url({"content": {"video_url": "https://example.test/video.mp4"}})

    def test_prepare_submit_and_complete_are_separate_steps(self):
        config = RunConfig(output_dir="unused", duration=8, enhanced_text_prompt=True)
        shot = ShotSpec(
            scene={"scene_num": 1, "video_prompts": ["A calm room"]},
            scene_num=1,
            shot_num=1,
            prompt="A calm room",
            is_cut=True,
        )
        client = FakeSeedanceClient()

        prepared = prepare_shot(
            config=config,
            story_script={"scenes": [shot.scene]},
            shot=shot,
            references=[],
            is_first=True,
        )
        submission = submit_shot(config=config, client=client, prepared=prepared)

        self.assertEqual(submission.task_id, "task-1")
        self.assertIn("CURRENT GENERATION TASK", prepared.submitted_prompt)
        self.assertEqual(client.created_options["duration"], 8)

        with tempfile.TemporaryDirectory() as tempdir:
            output = str(Path(tempdir) / "output.mp4")
            statuses = []
            completed = complete_shot(
                config=config,
                client=client,
                task_id=submission.task_id,
                output_video=output,
                on_status=statuses.append,
            )

            self.assertEqual(statuses, ["waiting_for_seedance", "downloading"])
            self.assertEqual(completed.output_video, output)
            self.assertEqual(Path(output).read_bytes(), b"video")

    def test_submit_uses_shot_duration_when_present(self):
        config = RunConfig(output_dir="unused", duration=8)
        shot = ShotSpec(
            scene={"scene_num": 1, "video_prompts": ["A short shot"]},
            scene_num=1,
            shot_num=1,
            prompt="A short shot",
            is_cut=True,
            duration_seconds=5,
        )
        client = FakeSeedanceClient()

        prepared = prepare_shot(
            config=config,
            story_script={"scenes": [shot.scene]},
            shot=shot,
            references=[],
            is_first=True,
        )
        submit_shot(config=config, client=client, prepared=prepared)

        self.assertEqual(client.created_options["duration"], 5)

    def test_wait_task_can_be_cancelled_before_polling(self):
        client = SeedanceClient(api_key="test-key", api_base="https://example.test")

        with self.assertRaises(SeedanceCancelledError):
            client.wait_task("task-1", should_cancel=lambda: True)


if __name__ == "__main__":
    unittest.main()
