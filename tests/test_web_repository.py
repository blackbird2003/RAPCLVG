import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from storymem_web.costs import Pricing
from storymem_web.repository import (
    SCHEMA,
    ConflictError,
    InvalidStateError,
    ProjectRepository,
)
from storymem_web.views import project_view


class ProjectRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.repository = ProjectRepository(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_create_empty_project_persists_one_shot(self):
        project = self.repository.create_empty_project("Notebook")

        self.assertEqual(project["name"], "Notebook")
        self.assertEqual(len(project["shots"]), 1)
        self.assertEqual(project["shots"][0]["state"], "draft")
        self.assertEqual(project["shots"][0]["duration_seconds"], 8)
        self.assertTrue(project["shots"][0]["memory_sink"])
        self.assertFalse(project["shots"][0]["memory_retrieve"])
        self.assertTrue(project["shots"][0]["memory_recent"])
        self.assertEqual(project["generation_config"]["pipeline_version"], "classic")
        self.assertEqual(project["generation_config"]["visual_element_sink_frame_count"], 0)
        self.assertEqual(project["generation_config"]["visual_element_max_retrieved_frames"], 4)
        self.assertTrue((self.repository.project_dir(project["project_id"]) / "source_story.json").exists())

    def test_project_visual_settings_validate_seedance_image_budget(self):
        project = self.repository.create_empty_project("Settings")

        updated = self.repository.update_project_generation_config(
            project["project_id"],
            {
                "pipeline_version": "classic",
                "visual_element_sink_frame_count": 2,
                "visual_element_max_retrieved_frames": 6,
            },
        )

        self.assertEqual(updated["generation_config"]["pipeline_version"], "classic")
        self.assertEqual(updated["generation_config"]["visual_element_sink_frame_count"], 2)
        self.assertEqual(updated["generation_config"]["visual_element_max_retrieved_frames"], 6)
        with self.assertRaisesRegex(ValueError, "<= 9"):
            self.repository.update_project_generation_config(
                project["project_id"],
                {
                    "visual_element_sink_frame_count": 6,
                    "visual_element_max_retrieved_frames": 4,
                },
            )

    def test_legacy_last_frame_switch_migrates_to_generation_mode(self):
        with tempfile.TemporaryDirectory() as workspace:
            db_path = Path(workspace) / "storymem_web.sqlite3"
            legacy_schema = SCHEMA.replace(
                "    generation_mode TEXT NOT NULL DEFAULT 'default',\n",
                "    last_frame_only INTEGER NOT NULL DEFAULT 0,\n",
            )
            connection = sqlite3.connect(db_path)
            connection.executescript(legacy_schema)
            connection.execute(
                """
                INSERT INTO projects (
                    project_id, name, generation_config_json, source_story_json,
                    created_at, updated_at
                ) VALUES ('legacy', 'Legacy', '{}', '{}', 'now', 'now')
                """
            )
            connection.execute(
                """
                INSERT INTO shots (
                    shot_id, project_id, order_index, scene_num, shot_num,
                    video_prompt, is_cut, last_frame_only, created_at, updated_at
                ) VALUES ('legacy:shot-0002', 'legacy', 1, 1, 2,
                          'continue', 0, 1, 'now', 'now')
                """
            )
            connection.commit()
            connection.close()

            migrated = ProjectRepository(workspace).get_shot("legacy:shot-0002")

            self.assertEqual(migrated["generation_mode"], "last_frame_only")
            self.assertEqual(
                ProjectRepository(workspace).get_project("legacy")["generation_config"]["pipeline_version"],
                "classic",
            )

    def test_legacy_shots_inherit_project_duration(self):
        with tempfile.TemporaryDirectory() as workspace:
            db_path = Path(workspace) / "storymem_web.sqlite3"
            legacy_schema = SCHEMA.replace(
                "    duration_seconds INTEGER NOT NULL DEFAULT 8,\n", ""
            )
            connection = sqlite3.connect(db_path)
            connection.executescript(legacy_schema)
            connection.execute(
                """
                INSERT INTO projects (
                    project_id, name, generation_config_json, source_story_json,
                    created_at, updated_at
                ) VALUES ('legacy', 'Legacy', '{"duration": 10}', '{}', 'now', 'now')
                """
            )
            connection.execute(
                """
                INSERT INTO shots (
                    shot_id, project_id, order_index, scene_num, shot_num,
                    video_prompt, is_cut, created_at, updated_at
                ) VALUES ('legacy:shot-0001', 'legacy', 0, 1, 1,
                          'start', 1, 'now', 'now')
                """
            )
            connection.commit()
            connection.close()

            migrated = ProjectRepository(workspace).get_shot("legacy:shot-0001")

            self.assertEqual(migrated["duration_seconds"], 10)

    def test_legacy_shots_migrate_to_original_memory_policy(self):
        with tempfile.TemporaryDirectory() as workspace:
            db_path = Path(workspace) / "storymem_web.sqlite3"
            legacy_schema = SCHEMA
            for line in (
                "    memory_sink INTEGER NOT NULL DEFAULT 1,\n",
                "    memory_retrieve INTEGER NOT NULL DEFAULT 1,\n",
                "    memory_recent INTEGER NOT NULL DEFAULT 0,\n",
            ):
                legacy_schema = legacy_schema.replace(line, "")
            connection = sqlite3.connect(db_path)
            connection.executescript(legacy_schema)
            connection.execute(
                """
                INSERT INTO projects (
                    project_id, name, generation_config_json, source_story_json,
                    created_at, updated_at
                ) VALUES ('legacy', 'Legacy', '{}', '{}', 'now', 'now')
                """
            )
            for index, is_cut in enumerate((1, 0)):
                connection.execute(
                    """
                    INSERT INTO shots (
                        shot_id, project_id, order_index, scene_num, shot_num,
                        video_prompt, is_cut, created_at, updated_at
                    ) VALUES (?, 'legacy', ?, 1, ?, 'prompt', ?, 'now', 'now')
                    """,
                    (f"legacy:shot-{index}", index, index + 1, is_cut),
                )
            connection.commit()
            connection.close()

            project = ProjectRepository(workspace).get_project("legacy")

            self.assertEqual(
                [
                    (shot["memory_sink"], shot["memory_retrieve"], shot["memory_recent"])
                    for shot in project["shots"]
                ],
                [(True, False, True), (True, False, True)],
            )

    def test_rename_project_updates_database_and_source_story(self):
        project = self.repository.create_empty_project("Before")

        renamed = self.repository.rename_project(project["project_id"], "  After  ")

        self.assertEqual(renamed["name"], "After")
        self.assertEqual(renamed["source_story"]["story_name"], "After")
        source = json.loads(
            (self.repository.project_dir(project["project_id"]) / "source_story.json").read_text()
        )
        self.assertEqual(source["story_name"], "After")
        with self.assertRaises(ValueError):
            self.repository.rename_project(project["project_id"], "   ")

    def test_completed_shots_stop_polling_while_project_runs(self):
        project = self.repository.create_project(
            {
                "story_name": "Polling",
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
        self.repository.set_shot_state(first["shot_id"], "completed")
        self.repository.set_project_state(
            project["project_id"],
            "running",
            active_shot_id=second["shot_id"],
        )

        view = project_view(self.repository, project["project_id"])

        self.assertFalse(view["shots"][0]["poll"])
        self.assertTrue(view["shots"][1]["poll"])

    def test_failed_attempt_with_video_can_retry_keyframes(self):
        project = self.repository.create_empty_project("Repairable")
        shot = project["shots"][0]
        attempt = self.repository.create_attempt(shot["shot_id"], {})
        attempt_dir = self.repository.attempt_dir(
            project["project_id"],
            shot["shot_id"],
            attempt["attempt_id"],
        )
        attempt_dir.mkdir(parents=True)
        output = attempt_dir / "01_01.mp4"
        output.write_bytes(b"video")
        self.repository.update_attempt(
            attempt["attempt_id"],
            status="failed",
            task_id="task-repairable",
            output_video=str(output),
            error={"type": "OutOfMemoryError"},
            finished=True,
        )

        view = project_view(self.repository, project["project_id"])

        self.assertTrue(view["shots"][0]["can_retry_keyframes"])
        output.unlink()
        view = project_view(self.repository, project["project_id"])
        self.assertFalse(view["shots"][0]["can_retry_keyframes"])

    def test_edit_uses_optimistic_row_version_and_locks_active_shot(self):
        project = self.repository.create_empty_project()
        shot = project["shots"][0]

        updated = self.repository.update_shot(
            shot["shot_id"],
            video_prompt="A revised prompt",
            is_cut=False,
            generation_mode="last_frame_only",
            duration_seconds=12,
            memory_sink=False,
            memory_retrieve=True,
            memory_recent=False,
            expected_row_version=shot["row_version"],
        )
        self.assertEqual(updated["video_prompt"], "A revised prompt")
        self.assertFalse(updated["is_cut"])
        self.assertEqual(updated["generation_mode"], "default")
        self.assertEqual(updated["duration_seconds"], 12)
        self.assertEqual(
            (updated["memory_sink"], updated["memory_retrieve"], updated["memory_recent"]),
            (False, True, False),
        )

        with self.assertRaisesRegex(ValueError, "between 2 and 15"):
            self.repository.update_shot(shot["shot_id"], duration_seconds=16)

        with self.assertRaises(ConflictError):
            self.repository.update_shot(
                shot["shot_id"],
                video_prompt="stale browser edit",
                expected_row_version=shot["row_version"],
            )

        self.repository.create_attempt(
            shot["shot_id"],
            input_snapshot={"video_prompt": "A revised prompt", "references": []},
        )
        with self.assertRaises(InvalidStateError):
            self.repository.update_shot(shot["shot_id"], video_prompt="too late")

    def test_generation_mode_requires_a_non_cut_shot_with_a_predecessor(self):
        project = self.repository.create_project(
            {
                "story_name": "Continuity mode",
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

        first = self.repository.update_shot(
            first["shot_id"],
            is_cut=False,
            generation_mode="last_frame_only",
        )
        self.assertEqual(first["generation_mode"], "default")

        second = self.repository.update_shot(
            second["shot_id"],
            generation_mode="last_frame_only",
        )
        self.assertEqual(second["generation_mode"], "last_frame_only")

        second = self.repository.update_shot(
            second["shot_id"],
            is_cut=True,
        )
        self.assertEqual(second["generation_mode"], "default")
        with self.assertRaisesRegex(ValueError, "Unknown generation mode"):
            self.repository.update_shot(
                second["shot_id"], generation_mode="future_mode"
            )

    def test_smooth_mode_can_be_configured_before_predecessor_runs(self):
        project = self.repository.create_project(
            {
                "story_name": "Smooth eligibility",
                "scenes": [
                    {
                        "scene_num": 1,
                        "video_prompts": ["first", "second"],
                        "cut": [True, False],
                    }
                ],
            }
        )
        _, second = project["shots"]

        self.assertEqual(second["generation_mode"], "default")

        updated = self.repository.update_shot(
            second["shot_id"], generation_mode="smooth"
        )
        self.assertEqual(updated["generation_mode"], "smooth")

        updated = self.repository.update_shot(second["shot_id"], is_cut=True)
        self.assertEqual(updated["generation_mode"], "default")

    def test_failed_shot_can_be_edited_before_creating_a_new_attempt(self):
        project = self.repository.create_project(
            {
                "story_name": "Failed retry",
                "scenes": [
                    {
                        "scene_num": 1,
                        "video_prompts": ["first", "second"],
                        "cut": [True, False],
                    }
                ],
            }
        )
        second = project["shots"][1]
        self.repository.set_shot_state(second["shot_id"], "failed")

        updated = self.repository.update_shot(
            second["shot_id"],
            generation_mode="last_frame_only",
        )

        self.assertEqual(updated["state"], "failed")
        self.assertEqual(updated["generation_mode"], "last_frame_only")
        view = project_view(self.repository, project["project_id"])
        self.assertTrue(view["shots"][1]["can_edit"])

    def test_reset_from_shot_invalidates_downstream_attempts(self):
        story = {
            "story_name": "Three shots",
            "scenes": [
                {"scene_num": 1, "video_prompts": ["one", "two", "three"], "cut": [True, False, True]}
            ],
        }
        project = self.repository.create_project(story)
        first, second, third = project["shots"]
        for shot in (first, second, third):
            self.repository.create_attempt(shot["shot_id"], {"video_prompt": shot["video_prompt"], "references": []})
            self.repository.set_shot_state(shot["shot_id"], "completed")

        reset = self.repository.reset_from_shot(project["project_id"], second["shot_id"])

        self.assertEqual([shot["state"] for shot in reset["shots"]], ["completed", "draft", "stale"])
        self.assertEqual([shot["revision"] for shot in reset["shots"]], [1, 2, 2])
        attempts = self.repository.list_attempts(project["project_id"])
        current_by_shot = {attempt["shot_id"]: attempt["is_current"] for attempt in attempts}
        self.assertTrue(current_by_shot[first["shot_id"]])
        self.assertFalse(current_by_shot[second["shot_id"]])
        self.assertFalse(current_by_shot[third["shot_id"]])

    def test_cost_summary_separates_current_and_all_attempts(self):
        project = self.repository.create_empty_project()
        shot = project["shots"][0]
        old = self.repository.create_attempt(
            shot["shot_id"], {"references": []}
        )
        self._set_usage(old["attempt_id"], "completed", 1_000_000)
        self.repository.set_shot_state(shot["shot_id"], "failed")
        current = self.repository.create_attempt(
            shot["shot_id"], {"references": []}
        )
        self._set_usage(current["attempt_id"], "completed", 500_000)

        summary = self.repository.cost_summary(
            project["project_id"], Pricing(no_video_cny_per_m_tokens=40)
        )

        self.assertEqual(summary["all_attempts"]["tasks"], 0)
        self.assertEqual(summary["current_version"]["estimated_cny"], 0.0)

        # Task IDs are required before provider usage counts as incurred.
        with self.repository.connect() as connection, connection:
            connection.execute("UPDATE attempts SET task_id = 'old' WHERE attempt_id = ?", (old["attempt_id"],))
            connection.execute("UPDATE attempts SET task_id = 'current' WHERE attempt_id = ?", (current["attempt_id"],))
        summary = self.repository.cost_summary(
            project["project_id"], Pricing(no_video_cny_per_m_tokens=40)
        )
        self.assertEqual(summary["all_attempts"]["tasks"], 2)
        self.assertEqual(summary["all_attempts"]["estimated_cny"], 60.0)
        self.assertEqual(summary["current_version"]["estimated_cny"], 20.0)

    def test_references_memory_assets_and_valid_prefix_are_versioned(self):
        story = {
            "story_name": "Two shots",
            "scenes": [{"scene_num": 1, "video_prompts": ["one", "two"], "cut": [True, False]}],
        }
        project = self.repository.create_project(story)
        first, second = project["shots"]
        attempt = self.repository.create_attempt(first["shot_id"], {"references": []})
        references = self.repository.replace_references(
            attempt["attempt_id"],
            [{"source_path": "frame.jpg", "roles": ["early_sink_memory"], "score": 0.8}],
        )
        self.assertEqual(references[0]["roles"], ["early_sink_memory"])

        self.repository.update_attempt(
            attempt["attempt_id"], status="completed", output_video="one.mp4", finished=True
        )
        self.repository.add_memory_assets(
            project["project_id"],
            first["shot_id"],
            attempt["attempt_id"],
            [
                {"asset_type": "retrieval_keyframe", "source_path": "one_keyframe0.jpg", "rank": 0},
                {"asset_type": "ending_frame", "source_path": "last_frame.jpg", "rank": 0},
            ],
        )
        active = self.repository.list_memory_assets(project["project_id"])
        self.assertEqual([asset["asset_type"] for asset in active], ["retrieval_keyframe", "ending_frame"])
        self.assertEqual([item["shot_id"] for item in self.repository.valid_completed_prefix(project["project_id"])], [first["shot_id"]])

        reset = self.repository.reset_from_shot(project["project_id"], first["shot_id"])
        self.assertEqual(reset["shots"][1]["state"], "stale")
        self.assertEqual(self.repository.list_memory_assets(project["project_id"]), [])

    def test_only_one_job_can_hold_the_global_execution_lease(self):
        first = self.repository.create_empty_project("First")
        second = self.repository.create_empty_project("Second")
        job = self.repository.create_job(first["project_id"], "all")

        with self.assertRaises(ConflictError):
            self.repository.create_job(second["project_id"], "all")

        self.repository.update_job(job["job_id"], status="completed")
        next_job = self.repository.create_job(second["project_id"], "single", second["shots"][0]["shot_id"])
        self.assertEqual(next_job["status"], "queued")

    def _set_usage(self, attempt_id, status, tokens):
        with self.repository.connect() as connection, connection:
            connection.execute(
                "UPDATE attempts SET status = ?, usage_json = ? WHERE attempt_id = ?",
                (status, '{"total_tokens": %d}' % tokens, attempt_id),
            )


if __name__ == "__main__":
    unittest.main()
