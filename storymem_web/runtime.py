from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def source_files(root: Path = PROJECT_ROOT) -> Iterable[Path]:
    for package in ("storymem_web", "storymem_seedance"):
        yield from sorted((root / package).rglob("*.py"))
    for name in ("seedance_client.py", "extract_keyframes.py", "memory_query_llm.py"):
        path = root / name
        if path.exists():
            yield path


def source_fingerprint(root: Path = PROJECT_ROOT) -> str:
    digest = hashlib.sha256()
    for path in source_files(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def git_info(root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    commit = _git(root, "rev-parse", "HEAD") or "unknown"
    branch = _git(root, "branch", "--show-current") or "unknown"
    porcelain = _git(root, "status", "--porcelain", "--untracked-files=normal")
    changes = [line for line in porcelain.splitlines() if line]
    return {
        "git_commit": commit,
        "git_commit_short": commit[:12] if commit != "unknown" else commit,
        "git_branch": branch,
        "git_dirty": bool(changes),
        "git_change_count": len(changes),
    }


def build_runtime_info(
    *,
    schema_version: Optional[int] = None,
    workspace: Optional[str | Path] = None,
    root: Path = PROJECT_ROOT,
) -> Dict[str, Any]:
    info = git_info(root)
    info.update(
        {
            "source_fingerprint": source_fingerprint(root),
            "schema_version": schema_version,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "workspace": str(Path(workspace).resolve()) if workspace else None,
        }
    )
    return info


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Print StoryMem source version metadata")
    parser.add_argument("--field")
    args = parser.parse_args()
    info = build_runtime_info()
    if args.field:
        value = info.get(args.field)
        if value is None:
            raise SystemExit(f"Unknown or empty field: {args.field}")
        print(value)
    else:
        print(json.dumps(info, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
