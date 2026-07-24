#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/lzg/wxh/world_model_projects/StoryMem"
OUTPUT_DIR="results/mr_e_seedance_enhanced_8s"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"
exec >>"${OUTPUT_DIR}/a6000_run.stdout.log" 2>&1

echo "[$(date '+%F %T')] Starting Mr. E enhanced 8s experiment"
source ./a6000_env.sh

exec python seedance_pipeline.py \
  --story_script_path ./story/mr_e.json \
  --output_dir "${OUTPUT_DIR}" \
  --max_shots 999 \
  --duration 8 \
  --ratio 16:9 \
  --max_memory_size 10 \
  --fix 3 \
  --retrieval_top_k 2 \
  --prompt_retrieval \
  --enhanced_text_prompt
