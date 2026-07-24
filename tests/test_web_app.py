import json
import tempfile
import unittest

from fastapi.testclient import TestClient

from storymem_web.app import create_app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.app = create_app(self.tempdir.name)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.tempdir.cleanup()

    def test_create_project_and_render_notebook(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Video projects", response.text)

        created = self.client.post(
            "/projects",
            data={"name": "Web notebook"},
            follow_redirects=False,
        )
        self.assertEqual(created.status_code, 303)
        project_url = created.headers["location"]

        page = self.client.get(project_url)
        self.assertEqual(page.status_code, 200)
        self.assertIn("Web notebook", page.text)
        self.assertIn("Video prompt", page.text)
        self.assertIn("Generation duration", page.text)
        self.assertIn("Classic", page.text)
        self.assertIn("Memory sources", page.text)
        self.assertIn("Memory decision", page.text)
        self.assertIn("Produced memory candidates", page.text)
        self.assertIn("Current assembly", page.text)

        renamed = self.client.post(
            f"{project_url}/rename",
            data={"name": "Renamed notebook"},
            follow_redirects=False,
        )
        self.assertEqual(renamed.status_code, 303)
        self.assertIn("Renamed notebook", self.client.get(project_url).text)

        api_renamed = self.client.patch(
            f"/api{project_url}",
            json={"name": "API renamed notebook"},
        )
        self.assertEqual(api_renamed.status_code, 200)
        self.assertEqual(api_renamed.json()["name"], "API renamed notebook")

    def test_import_story_and_update_shot_through_api(self):
        story = {
            "story_name": "Imported",
            "scenes": [
                {"scene_num": 1, "video_prompts": ["one", "two"], "cut": [True, False]}
            ],
        }
        imported = self.client.post(
            "/projects/import",
            files={"story_file": ("story.json", json.dumps(story), "application/json")},
            follow_redirects=False,
        )
        self.assertEqual(imported.status_code, 303)
        project_id = imported.headers["location"].rsplit("/", 1)[-1]
        detail = self.client.get(f"/api/projects/{project_id}").json()
        self.assertEqual(len(detail["shots"]), 2)
        shot = detail["shots"][1]

        updated = self.client.patch(
            f"/api/projects/{project_id}/shots/{shot['shot_id']}",
            json={
                "video_prompt": "revised second shot",
                "is_cut": False,
                "generation_mode": "last_frame_only",
                "duration_seconds": 12,
                "memory_sink": False,
                "memory_retrieve": True,
                "memory_recent": False,
                "expected_row_version": shot["row_version"],
            },
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["video_prompt"], "revised second shot")
        self.assertFalse(updated.json()["is_cut"])
        self.assertEqual(updated.json()["generation_mode"], "last_frame_only")
        self.assertEqual(updated.json()["duration_seconds"], 12)
        self.assertFalse(updated.json()["memory_sink"])
        self.assertTrue(updated.json()["memory_retrieve"])
        self.assertFalse(updated.json()["memory_recent"])
        page = self.client.get(f"/projects/{project_id}")
        self.assertIn("Last frame only", page.text)
        self.assertIn('name="generation_mode"', page.text)
        self.assertIn('name="duration_seconds"', page.text)
        self.assertIn('name="memory_sink"', page.text)
        self.assertIn('name="memory_retrieve"', page.text)
        self.assertIn('name="memory_recent"', page.text)

        stale = self.client.patch(
            f"/api/projects/{project_id}/shots/{shot['shot_id']}",
            json={"video_prompt": "stale", "expected_row_version": shot["row_version"]},
        )
        self.assertEqual(stale.status_code, 409)

    def test_static_assets_are_local(self):
        for path in (
            "/static/app.css",
            "/static/app.js",
            "/static/vendor/htmx.min.js",
            "/static/vendor/lucide.min.js",
        ):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertGreater(len(response.content), 100, path)
        app_js = self.client.get("/static/app.js").text
        self.assertNotIn("MutationObserver", app_js)
        self.assertIn("htmx:afterSwap", app_js)
        self.assertIn("pollScrollPositions", app_js)
        self.assertIn("data-keyframe-retry", app_js)
        self.assertIn('form.dataset.mode || "keyframes"', app_js)

    def test_completed_shot_renders_references_output_and_produced_memory(self):
        repository = self.app.state.repository
        project = repository.create_empty_project("Completed preview")
        project = repository.update_project_generation_config(
            project["project_id"],
            {"pipeline_version": "classic"},
        )
        shot = project["shots"][0]
        attempt = repository.create_attempt(shot["shot_id"], {"references": []})
        attempt_dir = repository.attempt_dir(project["project_id"], shot["shot_id"], attempt["attempt_id"])
        attempt_dir.mkdir(parents=True)
        output = attempt_dir / "01_01.mp4"
        reference = attempt_dir / "01_01_keyframe0.jpg"
        output.write_bytes(b"video")
        reference.write_bytes(b"jpeg")
        repository.replace_references(
            attempt["attempt_id"],
            [{"source_path": str(reference), "source_shot_id": shot["shot_id"], "roles": ["early_sink_memory"]}],
        )
        repository.add_memory_assets(
            project["project_id"],
            shot["shot_id"],
            attempt["attempt_id"],
            [{"asset_type": "retrieval_keyframe", "source_path": str(reference), "rank": 0}],
        )
        repository.update_attempt(
            attempt["attempt_id"],
            status="completed",
            task_id="task-preview",
            submitted_prompt="FULL SEEDANCE INPUT PROMPT",
            output_video=str(output),
            usage={"total_tokens": 1000, "duration_seconds": 8},
            finished=True,
        )

        page = self.client.get(f"/projects/{project['project_id']}")

        self.assertEqual(page.status_code, 200)
        self.assertIn("early sink memory", page.text)
        self.assertIn("candidate pool", page.text)
        self.assertIn("task-preview", page.text)
        self.assertIn("Input prompt", page.text)
        self.assertIn("FULL SEEDANCE INPUT PROMPT", page.text)
        self.assertIn(f'id="input-prompt-{attempt["attempt_id"]}"', page.text)
        self.assertIn(f'id="output-video-{attempt["attempt_id"]}"', page.text)
        self.assertGreaterEqual(page.text.count("hx-preserve"), 2)
        self.assertLess(page.text.index("Input prompt"), page.text.index("Output"))
        self.assertIn(f"/media/{project['project_id']}/", page.text)

    def test_visual_element_sections_render_as_tables(self):
        repository = self.app.state.repository
        project = repository.create_empty_project(
            "Visual tables",
            generation_config={"pipeline_version": "visual_element_v1"},
        )
        shot = project["shots"][0]
        attempt = repository.create_attempt(shot["shot_id"], {"references": []})
        attempt_dir = repository.attempt_dir(
            project["project_id"], shot["shot_id"], attempt["attempt_id"]
        )
        attempt_dir.mkdir(parents=True)
        frame = attempt_dir / "frame.jpg"
        frame.write_bytes(b"jpeg")
        repository.update_attempt(
            attempt["attempt_id"],
            status="completed",
            memory_selection={
                "policy": "visual_element_v1",
                "visual_element": {
                    "decision": {
                        "existing_element_states": [
                            {
                                "id": "element-0001",
                                "state": "should_reference",
                                "reason": "same protagonist",
                            }
                        ]
                    },
                    "inserted": [],
                    "state_after": {
                        "registry": [
                            {
                                "id": "element-0001",
                                "name": "Pilot",
                                "type": "character",
                                "introduced_at": "shot-0001",
                                "notes": "main role",
                            }
                        ]
                    },
                    "selected_references": [
                        {
                            "source_path": str(frame),
                            "reference_index": 1,
                            "source_scene_num": 1,
                            "source_shot_num": 1,
                            "score": 1.25,
                            "visual_element_selection": {
                                "newly_covered_element_ids": ["element-0001"],
                                "already_covered_element_ids": [],
                                "should_exclude": [],
                                "optional_or_uncertain": [],
                            },
                            "reference_intent": "应参考：Pilot",
                        },
                        {
                            "source_path": str(frame),
                            "reference_index": 2,
                            "source_scene_num": 1,
                            "source_shot_num": 1,
                            "roles": ["visual_sink_memory"],
                            "reference_intent": "早期参考锚点",
                        }
                    ],
                    "current_annotations": [
                        {
                            "source_scene_num": 1,
                            "source_shot_num": 1,
                            "frame_path": str(frame),
                            "elements": [{"name": "Pilot"}],
                        }
                    ],
                    "record": {},
                    "settings": {},
                },
            },
            finished=True,
        )

        page = self.client.get(f"/projects/{project['project_id']}")

        self.assertEqual(page.status_code, 200)
        self.assertGreaterEqual(page.text.count('<table class="visual-data-table'), 3)
        self.assertIn("<th>Name</th>", page.text)
        self.assertIn("<th>ID</th>", page.text)
        self.assertIn("<th>Type</th>", page.text)
        self.assertIn("<th>Reference intent</th>", page.text)
        self.assertIn("<th>Visible tracked elements</th>", page.text)


if __name__ == "__main__":
    unittest.main()
