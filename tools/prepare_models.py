#!/usr/bin/env python3
"""Pre-download the model weights used by the video generation system."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


HPSV3_REPO = "MizzenAI/HPSv3"
HPSV3_FILE = "HPSv3.safetensors"
QWEN_REPO = "Qwen/Qwen2-VL-7B-Instruct"

RIFE_REPO_URL = "https://github.com/hzwer/Practical-RIFE.git"
RIFE_REVISION = "17d8c7a1005b37f4c97bfee04e316aaec7fdc536"
RIFE_DRIVE_ID = "1_l4OgBp3GrrHOcQB87xXCI7OtTzyeXZL"
RIFE_FLOWNET_SHA256 = "6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2"


def _status(message: str) -> None:
    print(f"[setup] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_hf_models() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    # Offline flags are useful at runtime, but setup must be allowed to fetch.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    _status(f"downloading {HPSV3_REPO}/{HPSV3_FILE}")
    hf_hub_download(HPSV3_REPO, HPSV3_FILE, repo_type="model")

    _status(f"downloading {QWEN_REPO}")
    snapshot_download(QWEN_REPO, repo_type="model")


def prepare_clip() -> None:
    import clip
    import torch

    _status("downloading CLIP ViT-B/32")
    clip.load("ViT-B/32", device="cpu", jit=False)


def _rife_is_valid(dependency_dir: Path) -> bool:
    flownet = dependency_dir / "train_log" / "flownet.pkl"
    if not flownet.is_file():
        return False
    if _sha256(flownet) != RIFE_FLOWNET_SHA256:
        return False
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(dependency_dir), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return False
    return revision == RIFE_REVISION


def prepare_rife(project_root: Path) -> None:
    default_dir = project_root / ".runtime" / "deps" / "Practical-RIFE"
    dependency_dir = Path(os.getenv("VIDEOGEN_RIFE_DIR", default_dir)).resolve()
    if _rife_is_valid(dependency_dir):
        _status(f"Practical-RIFE already ready at {dependency_dir}")
        return

    import gdown

    _status("downloading Practical-RIFE source and RIFEv4.25 weights")
    with tempfile.TemporaryDirectory(prefix="videogen-rife-") as tmp:
        tmp_path = Path(tmp)
        repo_dir = tmp_path / "Practical-RIFE"
        subprocess.run(["git", "clone", RIFE_REPO_URL, str(repo_dir)], check=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "checkout", RIFE_REVISION],
            check=True,
        )
        zip_path = tmp_path / "RIFEv4.25_0919.zip"
        gdown.download(id=RIFE_DRIVE_ID, output=str(zip_path), quiet=False)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp_path)
        flownet = next(
            (path for path in tmp_path.rglob("flownet.pkl")),
            None,
        )
        if flownet is None:
            raise RuntimeError("RIFEv4.25 archive did not contain train_log/flownet.pkl")
        target_flownet = repo_dir / "train_log" / "flownet.pkl"
        target_flownet.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(flownet, target_flownet)
        if _sha256(target_flownet) != RIFE_FLOWNET_SHA256:
            raise RuntimeError("Downloaded RIFEv4.25 weights failed the SHA-256 check")

        if dependency_dir.exists():
            shutil.rmtree(dependency_dir)
        dependency_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(repo_dir, dependency_dir)
    if not _rife_is_valid(dependency_dir):
        raise RuntimeError("Practical-RIFE setup failed validation")
    _status(f"Practical-RIFE ready at {dependency_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hf", action="store_true", help="Skip Hugging Face model downloads.")
    parser.add_argument("--skip-clip", action="store_true", help="Skip the OpenAI CLIP checkpoint.")
    parser.add_argument("--skip-rife", action="store_true", help="Skip Practical-RIFE setup.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if not args.skip_hf:
        prepare_hf_models()
    if not args.skip_clip:
        prepare_clip()
    if not args.skip_rife:
        prepare_rife(project_root)
    _status("model preparation complete")


if __name__ == "__main__":
    main()
