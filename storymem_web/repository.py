from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .costs import Pricing, summarize_attempts
from .domain import (
    DEFAULT_SHOT_DURATION_SECONDS,
    EDITABLE_SHOT_STATES,
    GENERATION_MODES,
    LEGACY_PIPELINE_VERSION,
    SHOT_STATES,
    empty_story,
    normalize_generation_config,
    normalize_story,
    validate_shot_duration,
)


SCHEMA_VERSION = 5


class RepositoryError(RuntimeError):
    pass


class NotFoundError(RepositoryError):
    pass


class ConflictError(RepositoryError):
    pass


class InvalidStateError(RepositoryError):
    pass


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    run_mode TEXT,
    active_shot_id TEXT,
    generation_config_json TEXT NOT NULL,
    source_story_json TEXT NOT NULL,
    current_final_video TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shots (
    shot_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    order_index INTEGER NOT NULL,
    scene_num INTEGER NOT NULL,
    shot_num INTEGER NOT NULL,
    video_prompt TEXT NOT NULL,
    is_cut INTEGER NOT NULL,
    generation_mode TEXT NOT NULL DEFAULT 'default',
    duration_seconds INTEGER NOT NULL DEFAULT 8,
    memory_sink INTEGER NOT NULL DEFAULT 1,
    memory_retrieve INTEGER NOT NULL DEFAULT 1,
    memory_recent INTEGER NOT NULL DEFAULT 0,
    first_frame_prompt TEXT NOT NULL DEFAULT '',
    memory_query TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT 'draft',
    revision INTEGER NOT NULL DEFAULT 1,
    row_version INTEGER NOT NULL DEFAULT 1,
    current_attempt_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, order_index)
);

CREATE INDEX IF NOT EXISTS idx_shots_project_order
ON shots(project_id, order_index);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    shot_id TEXT NOT NULL REFERENCES shots(shot_id) ON DELETE CASCADE,
    revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 1,
    input_snapshot_json TEXT NOT NULL DEFAULT '{}',
    memory_selection_json TEXT NOT NULL DEFAULT '{}',
    submitted_prompt TEXT NOT NULL DEFAULT '',
    task_id TEXT,
    started_at TEXT,
    finished_at TEXT,
    usage_json TEXT NOT NULL DEFAULT '{}',
    error_json TEXT NOT NULL DEFAULT '{}',
    output_video TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_attempts_project_shot
ON attempts(project_id, shot_id, created_at);

CREATE TABLE IF NOT EXISTS reference_images (
    reference_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
    order_index INTEGER NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'auto',
    roles_json TEXT NOT NULL DEFAULT '[]',
    source_shot_id TEXT,
    source_path TEXT NOT NULL,
    score REAL,
    frame_score REAL,
    video_score REAL,
    created_at TEXT NOT NULL,
    UNIQUE(attempt_id, order_index)
);

CREATE TABLE IF NOT EXISTS memory_assets (
    memory_asset_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    source_shot_id TEXT NOT NULL REFERENCES shots(shot_id) ON DELETE CASCADE,
    source_attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id) ON DELETE CASCADE,
    asset_type TEXT NOT NULL,
    source_path TEXT NOT NULL,
    rank INTEGER,
    active_in_memory_pool INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_assets_project_active
ON memory_assets(project_id, active_in_memory_pool, created_at);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    shot_id TEXT,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    pid INTEGER,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_single_active
ON jobs((1)) WHERE status IN ('queued', 'running');
"""


_UNSET = object()


class ProjectRepository:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.db_path = self.workspace / "storymem_web.sqlite3"
        self._initialize()

    @classmethod
    def from_environment(cls) -> "ProjectRepository":
        return cls(os.getenv("STORYMEM_WEB_WORKSPACE", ".runtime/web"))

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection, connection:
            connection.executescript(SCHEMA)
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(shots)").fetchall()
            }
            if "generation_mode" not in columns:
                connection.execute(
                    "ALTER TABLE shots ADD COLUMN generation_mode TEXT NOT NULL DEFAULT 'default'"
                )
            if "duration_seconds" not in columns:
                connection.execute(
                    "ALTER TABLE shots ADD COLUMN duration_seconds INTEGER NOT NULL DEFAULT 8"
                )
                project_rows = connection.execute(
                    "SELECT project_id, generation_config_json FROM projects"
                ).fetchall()
                for project_row in project_rows:
                    config = _loads(project_row["generation_config_json"], {})
                    try:
                        duration = validate_shot_duration(
                            config.get("duration", DEFAULT_SHOT_DURATION_SECONDS)
                        )
                    except ValueError:
                        duration = DEFAULT_SHOT_DURATION_SECONDS
                    connection.execute(
                        "UPDATE shots SET duration_seconds = ? WHERE project_id = ?",
                        (duration, project_row["project_id"]),
                    )
            policy_columns = {"memory_sink", "memory_retrieve", "memory_recent"}
            if not policy_columns.issubset(columns):
                for column, default in (
                    ("memory_sink", 1),
                    ("memory_retrieve", 0),
                    ("memory_recent", 1),
                ):
                    if column not in columns:
                        connection.execute(
                            f"ALTER TABLE shots ADD COLUMN {column} "
                            f"INTEGER NOT NULL DEFAULT {default}"
                        )
                connection.execute(
                    """
                    UPDATE shots
                    SET memory_sink = 1,
                        memory_retrieve = 0,
                        memory_recent = 1
                    """
                )
            if "last_frame_only" in columns:
                connection.execute(
                    """
                    UPDATE shots
                    SET generation_mode = 'last_frame_only'
                    WHERE last_frame_only = 1 AND generation_mode = 'default'
                    """
                )
                connection.execute("UPDATE shots SET last_frame_only = 0")
            project_rows = connection.execute(
                "SELECT project_id, generation_config_json FROM projects"
            ).fetchall()
            for project_row in project_rows:
                config = _loads(project_row["generation_config_json"], {})
                if "pipeline_version" in config:
                    continue
                config = normalize_generation_config(
                    config,
                    default_pipeline_version=LEGACY_PIPELINE_VERSION,
                )
                connection.execute(
                    """
                    UPDATE projects
                    SET generation_config_json = ?
                    WHERE project_id = ?
                    """,
                    (_dumps(config), project_row["project_id"]),
                )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def schema_version(self) -> int:
        with self.connect() as connection:
            row = connection.execute("PRAGMA user_version").fetchone()
        return int(row[0])

    def create_empty_project(
        self,
        name: str = "Untitled video project",
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        story = empty_story()
        story["story_name"] = name.strip() or story["story_name"]
        return self.create_project(story, generation_config=generation_config)

    def create_project(
        self,
        story: Dict[str, Any],
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized = normalize_story(story)
        project_id = uuid.uuid4().hex
        now = _now()
        config = normalize_generation_config(generation_config)
        default_duration = validate_shot_duration(
            config.get("duration", DEFAULT_SHOT_DURATION_SECONDS)
        )
        with self.connect() as connection, connection:
            connection.execute(
                """
                INSERT INTO projects (
                    project_id, name, generation_config_json, source_story_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    normalized.story_name,
                    _dumps(config),
                    _dumps(normalized.source),
                    now,
                    now,
                ),
            )
            for shot in normalized.shots:
                shot_id = f"{project_id}:{shot.shot_id}"
                memory_sink, memory_retrieve, memory_recent = _default_memory_policy(
                    config
                )
                connection.execute(
                    """
                    INSERT INTO shots (
                        shot_id, project_id, order_index, scene_num, shot_num,
                        video_prompt, is_cut, generation_mode, duration_seconds,
                        memory_sink, memory_retrieve, memory_recent,
                        first_frame_prompt, memory_query, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        shot_id,
                        project_id,
                        shot.order_index,
                        shot.scene_num,
                        shot.shot_num,
                        shot.video_prompt,
                        int(shot.is_cut),
                        "default",
                        (
                            default_duration
                            if shot.duration_seconds is None
                            else shot.duration_seconds
                        ),
                        int(memory_sink),
                        int(memory_retrieve),
                        int(memory_recent),
                        shot.first_frame_prompt,
                        shot.memory_query,
                        now,
                        now,
                    ),
                )
        project_dir = self.project_dir(project_id)
        project_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(project_dir / "source_story.json", normalized.source)
        return self.get_project(project_id)

    def rename_project(self, project_id: str, name: str) -> Dict[str, Any]:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Project name cannot be empty")
        if len(normalized_name) > 160:
            raise ValueError("Project name cannot exceed 160 characters")

        now = _now()
        with self.connect() as connection, connection:
            row = connection.execute(
                "SELECT source_story_json FROM projects WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Project not found: {project_id}")
            source_story = _loads(row["source_story_json"], {})
            source_story["story_name"] = normalized_name
            connection.execute(
                """
                UPDATE projects
                SET name = ?, source_story_json = ?, updated_at = ?
                WHERE project_id = ?
                """,
                (normalized_name, _dumps(source_story), now, project_id),
            )

        _atomic_write_json(self.project_dir(project_id) / "source_story.json", source_story)
        return self.get_project(project_id)

    def update_project_generation_config(
        self,
        project_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        with self.connect() as connection, connection:
            row = connection.execute(
                "SELECT generation_config_json FROM projects WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Project not found: {project_id}")
            active_job = connection.execute(
                """
                SELECT 1 FROM jobs
                WHERE project_id = ? AND status IN ('queued', 'running')
                """,
                (project_id,),
            ).fetchone()
            if active_job is not None:
                raise ConflictError("Project settings are locked while a job is active")
            config = _loads(row["generation_config_json"], {})
            config.update(updates)
            config = normalize_generation_config(config)
            now = _now()
            connection.execute(
                """
                UPDATE projects
                SET generation_config_json = ?, updated_at = ?
                WHERE project_id = ?
                """,
                (_dumps(config), now, project_id),
            )
        return self.get_project(project_id)

    def list_projects(self) -> List[Dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT p.*,
                       COUNT(s.shot_id) AS shot_count,
                       SUM(CASE WHEN s.state = 'completed' THEN 1 ELSE 0 END) AS completed_count
                FROM projects p
                LEFT JOIN shots s ON s.project_id = p.project_id
                GROUP BY p.project_id
                ORDER BY p.updated_at DESC
                """
            ).fetchall()
        return [self._project_row(row, include_source=False) for row in rows]

    def get_project(self, project_id: str) -> Dict[str, Any]:
        with self.connect() as connection:
            project_row = connection.execute(
                "SELECT * FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone()
            if project_row is None:
                raise NotFoundError(f"Project not found: {project_id}")
            shot_rows = connection.execute(
                "SELECT * FROM shots WHERE project_id = ? ORDER BY order_index",
                (project_id,),
            ).fetchall()
        project = self._project_row(project_row, include_source=True)
        project["shots"] = [self._shot_row(row) for row in shot_rows]
        return project

    def get_shot(self, shot_id: str) -> Dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM shots WHERE shot_id = ?", (shot_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"Shot not found: {shot_id}")
        return self._shot_row(row)

    def update_shot(
        self,
        shot_id: str,
        *,
        video_prompt: Optional[str] = None,
        is_cut: Optional[bool] = None,
        generation_mode: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        memory_sink: Optional[bool] = None,
        memory_retrieve: Optional[bool] = None,
        memory_recent: Optional[bool] = None,
        expected_row_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        with self.connect() as connection, connection:
            row = connection.execute("SELECT * FROM shots WHERE shot_id = ?", (shot_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"Shot not found: {shot_id}")
            if row["state"] not in EDITABLE_SHOT_STATES:
                raise InvalidStateError(f"Shot {shot_id} is locked in state {row['state']}")
            if expected_row_version is not None and int(row["row_version"]) != expected_row_version:
                raise ConflictError(
                    f"Shot {shot_id} changed from row version {expected_row_version} "
                    f"to {row['row_version']}"
                )
            prompt = row["video_prompt"] if video_prompt is None else video_prompt.strip()
            if not prompt:
                raise ValueError("video_prompt cannot be empty")
            cut = int(row["is_cut"]) if is_cut is None else int(bool(is_cut))
            mode = (
                str(row["generation_mode"])
                if generation_mode is None
                else str(generation_mode).strip()
            )
            if mode not in GENERATION_MODES:
                raise ValueError(f"Unknown generation mode: {mode}")
            if cut or int(row["order_index"]) == 0:
                mode = "default"
            duration = (
                int(row["duration_seconds"])
                if duration_seconds is None
                else validate_shot_duration(duration_seconds)
            )
            sink = int(row["memory_sink"]) if memory_sink is None else int(bool(memory_sink))
            retrieve = (
                int(row["memory_retrieve"])
                if memory_retrieve is None
                else int(bool(memory_retrieve))
            )
            recent = (
                int(row["memory_recent"])
                if memory_recent is None
                else int(bool(memory_recent))
            )
            now = _now()
            connection.execute(
                """
                UPDATE shots
                SET video_prompt = ?, is_cut = ?, generation_mode = ?, duration_seconds = ?,
                    memory_sink = ?, memory_retrieve = ?, memory_recent = ?,
                    row_version = row_version + 1, updated_at = ?
                WHERE shot_id = ?
                """,
                (prompt, cut, mode, duration, sink, retrieve, recent, now, shot_id),
            )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE project_id = ?",
                (now, row["project_id"]),
            )
        return self.get_shot(shot_id)

    def set_shot_state(self, shot_id: str, state: str) -> Dict[str, Any]:
        if state not in SHOT_STATES:
            raise ValueError(f"Unknown shot state: {state}")
        with self.connect() as connection, connection:
            row = connection.execute("SELECT project_id FROM shots WHERE shot_id = ?", (shot_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"Shot not found: {shot_id}")
            now = _now()
            connection.execute(
                "UPDATE shots SET state = ?, row_version = row_version + 1, updated_at = ? WHERE shot_id = ?",
                (state, now, shot_id),
            )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE project_id = ?",
                (now, row["project_id"]),
            )
        return self.get_shot(shot_id)

    def create_attempt(
        self,
        shot_id: str,
        input_snapshot: Dict[str, Any],
        memory_selection: Optional[Dict[str, Any]] = None,
        submitted_prompt: str = "",
    ) -> Dict[str, Any]:
        attempt_id = uuid.uuid4().hex
        now = _now()
        with self.connect() as connection, connection:
            shot = connection.execute("SELECT * FROM shots WHERE shot_id = ?", (shot_id,)).fetchone()
            if shot is None:
                raise NotFoundError(f"Shot not found: {shot_id}")
            if shot["state"] not in EDITABLE_SHOT_STATES | {"failed", "interrupted"}:
                raise InvalidStateError(f"Cannot prepare shot {shot_id} from state {shot['state']}")
            connection.execute(
                "UPDATE attempts SET is_current = 0 WHERE shot_id = ? AND is_current = 1",
                (shot_id,),
            )
            connection.execute(
                """
                INSERT INTO attempts (
                    attempt_id, project_id, shot_id, revision, status, is_current,
                    input_snapshot_json, memory_selection_json, submitted_prompt,
                    started_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'preparing', 1, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    shot["project_id"],
                    shot_id,
                    shot["revision"],
                    _dumps(input_snapshot),
                    _dumps(memory_selection or {}),
                    submitted_prompt,
                    now,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE shots
                SET state = 'preparing', current_attempt_id = ?, row_version = row_version + 1,
                    updated_at = ?
                WHERE shot_id = ?
                """,
                (attempt_id, now, shot_id),
            )
            connection.execute(
                """
                UPDATE projects
                SET status = 'running', active_shot_id = ?, updated_at = ?
                WHERE project_id = ?
                """,
                (shot_id, now, shot["project_id"]),
            )
        return self.get_attempt(attempt_id)

    def get_attempt(self, attempt_id: str) -> Dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Attempt not found: {attempt_id}")
        return self._attempt_row(row)

    def list_attempts(self, project_id: str) -> List[Dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM attempts WHERE project_id = ? ORDER BY created_at, attempt_id",
                (project_id,),
            ).fetchall()
        return [self._attempt_row(row) for row in rows]

    def update_attempt(
        self,
        attempt_id: str,
        *,
        status: Optional[str] = None,
        task_id: Any = _UNSET,
        input_snapshot: Any = _UNSET,
        memory_selection: Any = _UNSET,
        submitted_prompt: Any = _UNSET,
        usage: Any = _UNSET,
        error: Any = _UNSET,
        output_video: Any = _UNSET,
        finished: bool = False,
    ) -> Dict[str, Any]:
        with self.connect() as connection, connection:
            attempt = connection.execute(
                "SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if attempt is None:
                raise NotFoundError(f"Attempt not found: {attempt_id}")
            fields = []
            values: List[Any] = []
            if status is not None:
                fields.append("status = ?")
                values.append(status)
            for column, value, encoder in (
                ("task_id", task_id, lambda item: item),
                ("input_snapshot_json", input_snapshot, _dumps),
                ("memory_selection_json", memory_selection, _dumps),
                ("submitted_prompt", submitted_prompt, lambda item: item),
                ("usage_json", usage, _dumps),
                ("error_json", error, _dumps),
                ("output_video", output_video, lambda item: item),
            ):
                if value is not _UNSET:
                    fields.append(f"{column} = ?")
                    values.append(encoder(value))
            now = _now()
            if finished:
                fields.append("finished_at = ?")
                values.append(now)
            fields.append("updated_at = ?")
            values.append(now)
            values.append(attempt_id)
            connection.execute(
                f"UPDATE attempts SET {', '.join(fields)} WHERE attempt_id = ?",
                values,
            )
            if status is not None and bool(attempt["is_current"]):
                connection.execute(
                    """
                    UPDATE shots
                    SET state = ?, row_version = row_version + 1, updated_at = ?
                    WHERE shot_id = ? AND current_attempt_id = ?
                    """,
                    (status, now, attempt["shot_id"], attempt_id),
                )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE project_id = ?",
                (now, attempt["project_id"]),
            )
        return self.get_attempt(attempt_id)

    def replace_references(self, attempt_id: str, references: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        now = _now()
        with self.connect() as connection, connection:
            exists = connection.execute(
                "SELECT 1 FROM attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if exists is None:
                raise NotFoundError(f"Attempt not found: {attempt_id}")
            connection.execute("DELETE FROM reference_images WHERE attempt_id = ?", (attempt_id,))
            for index, reference in enumerate(references, start=1):
                connection.execute(
                    """
                    INSERT INTO reference_images (
                        reference_id, attempt_id, order_index, source_type, roles_json,
                        source_shot_id, source_path, score, frame_score, video_score, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        attempt_id,
                        index,
                        str(reference.get("source_type") or "auto"),
                        _dumps(reference.get("roles") or []),
                        reference.get("source_shot_id"),
                        str(reference.get("source_path") or ""),
                        reference.get("score"),
                        reference.get("frame_score"),
                        reference.get("video_score"),
                        now,
                    ),
                )
        return self.list_references(attempt_id)

    def list_references(self, attempt_id: str) -> List[Dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT r.*, s.scene_num AS source_scene_num,
                       s.shot_num AS source_shot_num,
                       s.video_prompt AS source_prompt
                FROM reference_images r
                LEFT JOIN shots s ON s.shot_id = r.source_shot_id
                WHERE r.attempt_id = ?
                ORDER BY r.order_index
                """,
                (attempt_id,),
            ).fetchall()
        return [
            {
                "reference_id": row["reference_id"],
                "attempt_id": row["attempt_id"],
                "order_index": int(row["order_index"]),
                "source_type": row["source_type"],
                "roles": _loads(row["roles_json"], []),
                "source_shot_id": row["source_shot_id"],
                "source_path": row["source_path"],
                "score": row["score"],
                "frame_score": row["frame_score"],
                "video_score": row["video_score"],
                "source_scene_num": row["source_scene_num"],
                "source_shot_num": row["source_shot_num"],
                "source_prompt": row["source_prompt"],
            }
            for row in rows
        ]

    def add_memory_assets(
        self,
        project_id: str,
        shot_id: str,
        attempt_id: str,
        assets: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        now = _now()
        with self.connect() as connection, connection:
            connection.execute(
                "DELETE FROM memory_assets WHERE source_attempt_id = ?",
                (attempt_id,),
            )
            connection.execute(
                "UPDATE memory_assets SET active_in_memory_pool = 0 WHERE source_shot_id = ?",
                (shot_id,),
            )
            for asset in assets:
                connection.execute(
                    """
                    INSERT INTO memory_assets (
                        memory_asset_id, project_id, source_shot_id, source_attempt_id,
                        asset_type, source_path, rank, active_in_memory_pool, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        project_id,
                        shot_id,
                        attempt_id,
                        str(asset["asset_type"]),
                        str(asset["source_path"]),
                        asset.get("rank"),
                        int(bool(asset.get("active_in_memory_pool", True))),
                        now,
                    ),
                )
        return self.list_memory_assets(project_id, shot_id=shot_id, current_only=False)

    def list_memory_assets(
        self,
        project_id: str,
        *,
        shot_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        asset_type: Optional[str] = None,
        current_only: bool = True,
    ) -> List[Dict[str, Any]]:
        clauses = ["m.project_id = ?"]
        values: List[Any] = [project_id]
        if shot_id is not None:
            clauses.append("m.source_shot_id = ?")
            values.append(shot_id)
        if attempt_id is not None:
            clauses.append("m.source_attempt_id = ?")
            values.append(attempt_id)
        if asset_type is not None:
            clauses.append("m.asset_type = ?")
            values.append(asset_type)
        if current_only:
            clauses.append("m.active_in_memory_pool = 1")
            clauses.append("a.is_current = 1")
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT m.*, s.scene_num, s.shot_num, s.video_prompt
                FROM memory_assets m
                JOIN shots s ON s.shot_id = m.source_shot_id
                JOIN attempts a ON a.attempt_id = m.source_attempt_id
                WHERE {' AND '.join(clauses)}
                ORDER BY s.order_index,
                         CASE m.asset_type
                             WHEN 'retrieval_keyframe' THEN 0
                             WHEN 'ending_frame' THEN 1
                             ELSE 2
                         END,
                         m.rank,
                         m.created_at
                """,
                values,
            ).fetchall()
        return [
            {
                "memory_asset_id": row["memory_asset_id"],
                "project_id": row["project_id"],
                "source_shot_id": row["source_shot_id"],
                "source_attempt_id": row["source_attempt_id"],
                "asset_type": row["asset_type"],
                "source_path": row["source_path"],
                "rank": row["rank"],
                "active_in_memory_pool": bool(row["active_in_memory_pool"]),
                "source_scene_num": int(row["scene_num"]),
                "source_shot_num": int(row["shot_num"]),
                "source_prompt": row["video_prompt"],
            }
            for row in rows
        ]

    def valid_completed_prefix(self, project_id: str) -> List[Dict[str, Any]]:
        project = self.get_project(project_id)
        prefix = []
        for shot in project["shots"]:
            if shot["state"] != "completed" or not shot["current_attempt_id"]:
                break
            attempt = self.get_attempt(shot["current_attempt_id"])
            if not attempt["is_current"] or not attempt["output_video"]:
                break
            prefix.append(attempt)
        return prefix

    def set_project_state(
        self,
        project_id: str,
        status: str,
        *,
        active_shot_id: Any = _UNSET,
        current_final_video: Any = _UNSET,
        run_mode: Any = _UNSET,
    ) -> Dict[str, Any]:
        fields = ["status = ?", "updated_at = ?"]
        values: List[Any] = [status, _now()]
        for column, value in (
            ("active_shot_id", active_shot_id),
            ("current_final_video", current_final_video),
            ("run_mode", run_mode),
        ):
            if value is not _UNSET:
                fields.append(f"{column} = ?")
                values.append(value)
        values.append(project_id)
        with self.connect() as connection, connection:
            cursor = connection.execute(
                f"UPDATE projects SET {', '.join(fields)} WHERE project_id = ?",
                values,
            )
            if cursor.rowcount != 1:
                raise NotFoundError(f"Project not found: {project_id}")
        return self.get_project(project_id)

    def create_job(self, project_id: str, mode: str, shot_id: Optional[str] = None) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = _now()
        with self.connect() as connection, connection:
            if connection.execute(
                "SELECT 1 FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone() is None:
                raise NotFoundError(f"Project not found: {project_id}")
            try:
                connection.execute(
                    """
                    INSERT INTO jobs (
                        job_id, project_id, shot_id, mode, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'queued', ?, ?)
                    """,
                    (job_id, project_id, shot_id, mode, now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ConflictError("Another StoryMem project runner is already active") from exc
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"Job not found: {job_id}")
        return self._job_row(row)

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        pid: Any = _UNSET,
        cancel_requested: Any = _UNSET,
    ) -> Dict[str, Any]:
        fields = ["updated_at = ?"]
        values: List[Any] = [_now()]
        for column, value in (("status", status), ("pid", pid), ("cancel_requested", cancel_requested)):
            if value is not _UNSET and value is not None:
                fields.append(f"{column} = ?")
                values.append(int(value) if column == "cancel_requested" else value)
        values.append(job_id)
        with self.connect() as connection, connection:
            cursor = connection.execute(
                f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?", values
            )
            if cursor.rowcount != 1:
                raise NotFoundError(f"Job not found: {job_id}")
        return self.get_job(job_id)

    def request_job_cancel(self, job_id: str) -> Dict[str, Any]:
        return self.update_job(job_id, cancel_requested=True)

    def job_cancel_requested(self, job_id: str) -> bool:
        return bool(self.get_job(job_id)["cancel_requested"])

    def list_jobs(
        self,
        *,
        project_id: Optional[str] = None,
        active_only: bool = False,
    ) -> List[Dict[str, Any]]:
        clauses = []
        values: List[Any] = []
        if project_id is not None:
            clauses.append("project_id = ?")
            values.append(project_id)
        if active_only:
            clauses.append("status IN ('queued', 'running')")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM jobs {where} ORDER BY created_at DESC", values
            ).fetchall()
        return [self._job_row(row) for row in rows]

    def active_job(self, project_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        jobs = self.list_jobs(project_id=project_id, active_only=True)
        return jobs[0] if jobs else None

    def reset_from_shot(self, project_id: str, shot_id: str) -> Dict[str, Any]:
        with self.connect() as connection, connection:
            target = connection.execute(
                "SELECT * FROM shots WHERE project_id = ? AND shot_id = ?",
                (project_id, shot_id),
            ).fetchone()
            if target is None:
                raise NotFoundError(f"Shot {shot_id} does not belong to project {project_id}")
            affected = connection.execute(
                """
                SELECT * FROM shots
                WHERE project_id = ? AND order_index >= ?
                ORDER BY order_index
                """,
                (project_id, target["order_index"]),
            ).fetchall()
            now = _now()
            for row in affected:
                connection.execute(
                    "UPDATE attempts SET is_current = 0, updated_at = ? WHERE shot_id = ? AND is_current = 1",
                    (now, row["shot_id"]),
                )
                connection.execute(
                    """
                    UPDATE memory_assets
                    SET active_in_memory_pool = 0
                    WHERE source_shot_id = ? AND active_in_memory_pool = 1
                    """,
                    (row["shot_id"],),
                )
                next_state = "draft" if row["shot_id"] == shot_id else "stale"
                connection.execute(
                    """
                    UPDATE shots
                    SET state = ?, revision = revision + 1, row_version = row_version + 1,
                        current_attempt_id = NULL, updated_at = ?
                    WHERE shot_id = ?
                    """,
                    (next_state, now, row["shot_id"]),
                )
            connection.execute(
                """
                UPDATE projects
                SET status = 'draft', active_shot_id = NULL, current_final_video = NULL,
                    updated_at = ?
                WHERE project_id = ?
                """,
                (now, project_id),
            )
        return self.get_project(project_id)

    def cost_summary(self, project_id: str, pricing: Pricing | None = None) -> Dict[str, Any]:
        return summarize_attempts(self.list_attempts(project_id), pricing=pricing)

    def project_dir(self, project_id: str) -> Path:
        return self.workspace / "projects" / project_id

    def attempt_dir(self, project_id: str, shot_id: str, attempt_id: str) -> Path:
        safe_shot = shot_id.split(":")[-1]
        return self.project_dir(project_id) / "shots" / safe_shot / "attempts" / attempt_id

    @staticmethod
    def _project_row(row: sqlite3.Row, include_source: bool) -> Dict[str, Any]:
        result = {
            "project_id": row["project_id"],
            "name": row["name"],
            "status": row["status"],
            "run_mode": row["run_mode"],
            "active_shot_id": row["active_shot_id"],
            "generation_config": _loads(row["generation_config_json"], {}),
            "current_final_video": row["current_final_video"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        if "shot_count" in row.keys():
            result["shot_count"] = int(row["shot_count"] or 0)
            result["completed_count"] = int(row["completed_count"] or 0)
        if include_source:
            result["source_story"] = _loads(row["source_story_json"], {})
        return result

    @staticmethod
    def _shot_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "shot_id": row["shot_id"],
            "project_id": row["project_id"],
            "order_index": int(row["order_index"]),
            "scene_num": int(row["scene_num"]),
            "shot_num": int(row["shot_num"]),
            "video_prompt": row["video_prompt"],
            "is_cut": bool(row["is_cut"]),
            "generation_mode": row["generation_mode"],
            "duration_seconds": int(row["duration_seconds"]),
            "memory_sink": bool(row["memory_sink"]),
            "memory_retrieve": bool(row["memory_retrieve"]),
            "memory_recent": bool(row["memory_recent"]),
            "first_frame_prompt": row["first_frame_prompt"],
            "memory_query": row["memory_query"],
            "state": row["state"],
            "revision": int(row["revision"]),
            "row_version": int(row["row_version"]),
            "current_attempt_id": row["current_attempt_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _attempt_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "attempt_id": row["attempt_id"],
            "project_id": row["project_id"],
            "shot_id": row["shot_id"],
            "revision": int(row["revision"]),
            "status": row["status"],
            "is_current": bool(row["is_current"]),
            "input_snapshot": _loads(row["input_snapshot_json"], {}),
            "memory_selection": _loads(row["memory_selection_json"], {}),
            "submitted_prompt": row["submitted_prompt"],
            "task_id": row["task_id"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "usage": _loads(row["usage_json"], {}),
            "error": _loads(row["error_json"], {}),
            "output_video": row["output_video"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _job_row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "project_id": row["project_id"],
            "shot_id": row["shot_id"],
            "mode": row["mode"],
            "status": row["status"],
            "pid": row["pid"],
            "cancel_requested": bool(row["cancel_requested"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: str, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _default_memory_policy(config: Dict[str, Any]) -> tuple[bool, bool, bool]:
    if str(config.get("pipeline_version") or LEGACY_PIPELINE_VERSION) == LEGACY_PIPELINE_VERSION:
        return True, False, True
    return True, True, False


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
