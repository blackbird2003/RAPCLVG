import argparse
import asyncio
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from storymem_web.app import create_app
from storymem_web.routes.api import health
from storymem_web.runtime import build_runtime_info, source_fingerprint
from tools.storymem_status import _project_detail, collect_status


class RuntimeStatusTests(unittest.TestCase):
    def test_source_fingerprint_changes_with_python_source(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            package = root / "storymem_web"
            package.mkdir()
            source = package / "module.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            first = source_fingerprint(root)

            source.write_text("VALUE = 2\n", encoding="utf-8")

            self.assertNotEqual(first, source_fingerprint(root))

    def test_health_reports_startup_version_and_schema(self):
        with tempfile.TemporaryDirectory() as workspace:
            app = create_app(workspace)
            payload = asyncio.run(health(SimpleNamespace(app=app)))

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["schema_version"], 5)
        self.assertEqual(len(payload["source_fingerprint"]), 16)
        self.assertEqual(payload["workspace"], str(Path(workspace).resolve()))

    @patch("tools.storymem_status._gpu_status", return_value={"available": False, "devices": []})
    @patch("tools.storymem_status._web_health", return_value={"reachable": False, "error": "offline"})
    def test_status_selects_project_without_mutating_database(self, _health, _gpu):
        with tempfile.TemporaryDirectory() as workspace:
            app = create_app(workspace)
            app.state.repository.create_empty_project("Miss D diagnostic")
            args = argparse.Namespace(
                project="miss d",
                workspace=Path(workspace),
                web_port=7860,
                limit=10,
                as_json=False,
            )

            status = collect_status(args)

        selected = status["database"]["selected_project"]
        self.assertEqual(selected["name"], "Miss D diagnostic")
        self.assertEqual(selected["shots"][0]["generation_mode"], "default")
        self.assertEqual(
            selected["shots"][0]["memory_policy"],
            {"sink": True, "retrieve": False, "recent": True},
        )

    def test_runtime_info_contains_git_and_process_identity(self):
        info = build_runtime_info(schema_version=2, workspace="/tmp/storymem-test")

        self.assertIn("git_commit_short", info)
        self.assertEqual(info["schema_version"], 2)
        self.assertGreater(info["pid"], 0)

    def test_status_reads_legacy_generation_switch_without_migrating_it(self):
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            CREATE TABLE projects (
                project_id TEXT, name TEXT, status TEXT, active_shot_id TEXT,
                updated_at TEXT
            );
            CREATE TABLE shots (
                project_id TEXT, order_index INTEGER, scene_num INTEGER,
                shot_num INTEGER, state TEXT, last_frame_only INTEGER,
                current_attempt_id TEXT
            );
            CREATE TABLE attempts (
                attempt_id TEXT, status TEXT, task_id TEXT, error_json TEXT,
                usage_json TEXT, updated_at TEXT
            );
            INSERT INTO projects VALUES ('p1', 'Legacy', 'draft', NULL, 'now');
            INSERT INTO shots VALUES ('p1', 1, 1, 2, 'failed', 1, NULL);
            """
        )
        project = connection.execute("SELECT * FROM projects").fetchone()

        detail = _project_detail(connection, project)
        connection.close()

        self.assertEqual(detail["shots"][0]["generation_mode"], "last_frame_only")


if __name__ == "__main__":
    unittest.main()
