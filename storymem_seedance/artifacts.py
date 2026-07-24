from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import RunConfig, ShotRunRecord, ShotSpec


SENSITIVE_URL_KEYS = {"video_url", "last_frame_url", "url", "uri"}


class ArtifactStore:
    """File-backed run state used by the CLI today and a web UI later."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.output_dir, "run_manifest.json")

    @property
    def task_log_path(self) -> str:
        return os.path.join(self.output_dir, "seedance_tasks.jsonl")

    @property
    def shot_log_path(self) -> str:
        return os.path.join(self.output_dir, "shots.jsonl")

    @property
    def reference_log_path(self) -> str:
        return os.path.join(self.output_dir, "references.jsonl")

    def init_manifest(self, config: RunConfig, story_name: str) -> Dict[str, Any]:
        if os.path.exists(self.manifest_path) and config.resume:
            manifest = self.read_manifest()
            manifest["config"] = config.to_dict()
            manifest["resumed_at"] = _now()
            self.write_manifest(manifest)
            return manifest
        manifest = {
            "run_id": uuid.uuid4().hex,
            "story_name": story_name,
            "story_script_path": config.story_script_path,
            "output_dir": self.output_dir,
            "status": "initialized",
            "created_at": _now(),
            "updated_at": _now(),
            "current_shot": None,
            "config": config.to_dict(),
        }
        self.write_manifest(manifest)
        return manifest

    def read_manifest(self) -> Dict[str, Any]:
        if not os.path.exists(self.manifest_path):
            return {}
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_manifest(self, manifest: Dict[str, Any]) -> None:
        manifest = dict(manifest)
        manifest["updated_at"] = _now()
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def update_manifest(self, status: str, current_shot: Optional[ShotSpec] = None, **extra: Any) -> None:
        manifest = self.read_manifest()
        if not manifest:
            manifest = {"run_id": uuid.uuid4().hex, "created_at": _now()}
        manifest["status"] = status
        if current_shot is not None:
            manifest["current_shot"] = {
                "scene_num": current_shot.scene_num,
                "shot_num": current_shot.shot_num,
                "prompt": current_shot.prompt,
                "is_cut": current_shot.is_cut,
            }
        for key, value in extra.items():
            manifest[key] = value
        self.write_manifest(manifest)

    def output_video_path(self, shot: ShotSpec) -> str:
        return os.path.join(self.output_dir, f"{shot.scene_num:02d}_{shot.shot_num:02d}.mp4")

    def has_keyframes(self, shot: ShotSpec) -> bool:
        pattern = os.path.join(self.output_dir, f"{shot.scene_num:02d}_{shot.shot_num:02d}_keyframe*.jpg")
        return bool(glob.glob(pattern))

    def memory_keyframes(self) -> List[str]:
        return sorted(glob.glob(os.path.join(self.output_dir, "*keyframe*.jpg")))

    def append_jsonl(self, path: str, record: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        return load_jsonl(path)

    def append_task_record(self, record: Dict[str, Any]) -> None:
        self.append_jsonl(self.task_log_path, record)

    def append_shot_record(self, record: ShotRunRecord | Dict[str, Any]) -> None:
        data = record.to_dict() if isinstance(record, ShotRunRecord) else record
        self.append_jsonl(self.shot_log_path, data)

    def append_reference_records(self, shot: ShotSpec, references: List[Dict[str, Any]]) -> None:
        for reference in references:
            self.append_jsonl(
                self.reference_log_path,
                {
                    "time": _now(),
                    "scene_num": shot.scene_num,
                    "shot_num": shot.shot_num,
                    "reference": reference,
                },
            )

    def find_resume_task(self, scene_num: int, shot_num: int) -> Optional[Dict[str, Any]]:
        matches = [
            record for record in self.load_jsonl(self.task_log_path)
            if record.get("scene_num") == scene_num and record.get("shot_num") == shot_num and record.get("task_id")
        ]
        return matches[-1] if matches else None

    def concat_videos(self) -> Optional[str]:
        videos = sorted(glob.glob(os.path.join(self.output_dir, "[0-9][0-9]_[0-9][0-9].mp4")))
        if not videos:
            return None
        out = os.path.join(self.output_dir, f"{os.path.basename(self.output_dir)}.mp4")
        return concat_video_paths(videos, out, list_path=os.path.join(self.output_dir, "concat_list.txt"))


def concat_video_paths(
    videos: Iterable[str],
    output_path: str,
    *,
    list_path: Optional[str] = None,
) -> Optional[str]:
    ordered = [str(video) for video in videos if os.path.exists(video)]
    if not ordered:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    list_path = list_path or f"{output_path}.concat.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for video in ordered:
            escaped = os.path.abspath(video).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            logging.warning("Skipping concat because ffmpeg is unavailable: %s", exc)
            return None
    result = subprocess.run(
        [ffmpeg, "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", "-y", output_path],
        check=False,
    )
    if result.returncode != 0:
        logging.error("Failed to concatenate %d clips into %s", len(ordered), output_path)
        return None
    logging.info("Updated concatenated video %s with %d clips", output_path, len(ordered))
    return output_path


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = str(path)
    if not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning("Skipping malformed jsonl line in %s: %s", path, line[:200])
    return records


def redact_urls(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ("<redacted_url>" if key.lower() in SENSITIVE_URL_KEYS and isinstance(item, str) and item.startswith("http") else redact_urls(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_urls(item) for item in value]
    return value


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
