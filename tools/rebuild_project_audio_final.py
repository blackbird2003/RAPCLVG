from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storymem_web.repository import ProjectRepository
from storymem_web.smooth_transition import _mux_concatenated_audio


DEFAULT_PROJECT_NAME = "锣老师别这样：咖啡店杯型风波"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a project's final video and preserve clip audio."
    )
    parser.add_argument(
        "--workspace",
        default="/home/wxh/world_model_projects/StoryMem/.runtime/web",
        help="StoryMem Web workspace containing storymem_web.sqlite3",
    )
    parser.add_argument("--project-id", default="", help="Exact project_id to rebuild")
    parser.add_argument(
        "--project-name",
        default=DEFAULT_PROJECT_NAME,
        help="Project name used when project_id is omitted",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path; defaults to <project>/final/current_with_audio.mp4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = ProjectRepository(args.workspace)
    project_id = args.project_id.strip() or _latest_completed_project_id(
        repository, args.project_name.strip()
    )
    prefix = repository.valid_completed_prefix(project_id)
    if not prefix:
        raise SystemExit(f"No completed clips available for project {project_id}")
    project = repository.get_project(project_id)
    final_input = Path(project.get("current_final_video") or "")
    if not final_input.is_file():
        raise SystemExit(f"Project has no existing final video: {project_id}")
    output = Path(args.output) if args.output else (
        repository.project_dir(project_id) / "final" / "current_with_audio.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_input, output)
    _mux_concatenated_audio(output, [Path(item["output_video"]) for item in prefix])
    print(project["name"])
    print(project_id)
    print(output)


def _latest_completed_project_id(repository: ProjectRepository, project_name: str) -> str:
    candidates = [
        project
        for project in repository.list_projects()
        if project["name"] == project_name and project["status"] == "completed"
    ]
    if not candidates:
        raise SystemExit(f"No completed project named {project_name}")
    candidates.sort(key=lambda item: item["updated_at"], reverse=True)
    return candidates[0]["project_id"]


if __name__ == "__main__":
    main()
