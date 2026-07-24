from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download
from modelscope import snapshot_download as modelscope_snapshot_download


ROOT = Path("/root/autodl-tmp/models/StoryMem")

MODELSCOPE_MODELS = [
    ("Wan-AI/Wan2.2-T2V-A14B-BF16", ROOT / "Wan2.2-T2V-A14B"),
    ("Wan-AI/Wan2.2-I2V-A14B-BF16", ROOT / "Wan2.2-I2V-A14B"),
]

HF_MODELS = [
    ("Kevin-thu/StoryMem", ROOT / "StoryMem"),
]

HF_PROXIES = {
    "http": "http://127.0.0.1:7897",
    "https": "http://127.0.0.1:7897",
}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for repo_id, local_dir in MODELSCOPE_MODELS:
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} with ModelScope -> {local_dir}", flush=True)
        modelscope_snapshot_download(
            repo_id,
            local_dir=str(local_dir),
        )
        print(f"Finished {repo_id}", flush=True)

    for repo_id, local_dir in HF_MODELS:
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} with Hugging Face -> {local_dir}", flush=True)
        os.environ.setdefault("HTTP_PROXY", HF_PROXIES["http"])
        os.environ.setdefault("HTTPS_PROXY", HF_PROXIES["https"])
        os.environ.setdefault("http_proxy", HF_PROXIES["http"])
        os.environ.setdefault("https_proxy", HF_PROXIES["https"])
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            proxies=HF_PROXIES,
        )
        print(f"Finished {repo_id}", flush=True)


if __name__ == "__main__":
    main()
