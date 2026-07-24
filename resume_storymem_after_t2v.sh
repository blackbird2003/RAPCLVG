#!/usr/bin/env bash
set -euo pipefail

cd /root/world_model_projects/StoryMem
source storymem_env.sh

OUTPUT_PATH="${1:-results/daiyu_single_gpu_example}"
LOG_PATH="/root/autodl-tmp/logs/storymem_single_gpu_example_resume.log"

python - <<'PY'
import os

import clip
from huggingface_hub import hf_hub_download, snapshot_download

print("Downloading runtime dependencies if missing...")
print(hf_hub_download("MizzenAI/HPSv3", "HPSv3.safetensors", repo_type="model"))
print(snapshot_download("Qwen/Qwen2-VL-7B-Instruct"))
clip.load("ViT-B/32", device="cpu", jit=False)
print("Runtime dependencies are ready.")
PY

OUTPUT_PATH="${OUTPUT_PATH}" python - <<'PY'
import os
from pathlib import Path

from extract_keyframes import save_keyframes

video = Path(os.environ["OUTPUT_PATH"]) / "01_01.mp4"
if not video.exists():
    raise FileNotFoundError(video)
save_keyframes(str(video))
PY

nohup python pipeline.py \
  --story_script_path ./story/daiyu.json \
  --t2v_model_path ./models/Wan2.2-T2V-A14B \
  --i2v_model_path ./models/Wan2.2-I2V-A14B \
  --lora_weight_path ./models/StoryMem/Wan2.2-MI2V-A14B \
  --size "832*480" \
  --max_memory_size 10 \
  --output_dir "${OUTPUT_PATH}" \
  --offload_model \
  --convert_model_dtype \
  --t5_cpu \
  --lora_rank 128 \
  --mi2v \
  > "${LOG_PATH}" \
  2>&1 &

echo $! > /root/autodl-tmp/logs/storymem_single_gpu_example_resume.pid
echo "pid=$(cat /root/autodl-tmp/logs/storymem_single_gpu_example_resume.pid)"
echo "log=${LOG_PATH}"
echo "output=${OUTPUT_PATH}"
