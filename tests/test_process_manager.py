import tempfile
import unittest
from unittest.mock import Mock, patch

from storymem_web.process_manager import ProcessManager
from storymem_web.repository import ProjectRepository


class ProcessManagerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.repository = ProjectRepository(self.tempdir.name)
        self.manager = ProcessManager(self.repository, spawn_reaper=False)

    def tearDown(self):
        self.tempdir.cleanup()

    @patch("storymem_web.process_manager.subprocess.Popen")
    def test_start_records_worker_pid_and_interrupt_requests_cancel(self, popen):
        popen.return_value = Mock(pid=4321)
        project = self.repository.create_empty_project()

        job = self.manager.start(project["project_id"], "all")

        self.assertEqual(job["pid"], 4321)
        self.assertFalse(job["cancel_requested"])
        worker_env = popen.call_args.kwargs["env"]
        self.assertEqual(worker_env["OMP_NUM_THREADS"], "8")
        self.assertEqual(worker_env["MKL_NUM_THREADS"], "8")
        interrupted = self.manager.interrupt(project["project_id"])
        self.assertTrue(interrupted["cancel_requested"])

    def test_reconcile_fails_a_queued_job_that_never_received_a_pid(self):
        project = self.repository.create_empty_project()
        job = self.repository.create_job(project["project_id"], "all")

        self.manager.reconcile()

        self.assertEqual(self.repository.get_job(job["job_id"])["status"], "failed")
        self.assertEqual(self.repository.get_project(project["project_id"])["status"], "failed")

    def test_reaper_waits_for_child_and_preserves_worker_terminal_state(self):
        project = self.repository.create_empty_project()
        job = self.repository.create_job(project["project_id"], "all")
        process = Mock()
        process.wait.return_value = 0
        self.repository.update_job(job["job_id"], status="completed", pid=99)

        self.manager._reap(job["job_id"], process)

        process.wait.assert_called_once_with()
        self.assertEqual(self.repository.get_job(job["job_id"])["status"], "completed")


if __name__ == "__main__":
    unittest.main()
