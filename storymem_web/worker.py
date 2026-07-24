from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .project_runner import ProjectRunner
from .repository import ProjectRepository
from .runtime import build_runtime_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a StoryMem notebook project job")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = ProjectRepository(args.workspace)
    job = repository.get_job(args.job_id)
    log_path = repository.project_dir(job["project_id"]) / "project.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )
    runtime = build_runtime_info(
        schema_version=repository.schema_version(),
        workspace=repository.workspace,
    )
    logging.info(
        "Starting notebook job %s in PID %d | commit=%s source=%s",
        args.job_id,
        os.getpid(),
        runtime["git_commit_short"],
        runtime["source_fingerprint"],
    )
    ProjectRunner(repository, args.job_id).run()


if __name__ == "__main__":
    main()
