#!/usr/bin/env bash
set -euo pipefail

source /root/world_model_projects/StoryMem/storymem_env.sh

OUTPUT_PATH="${1:-results/daiyu_single_gpu_smoke}"
mkdir -p "${OUTPUT_PATH}"

python pipeline.py \
  --story_script_path ./story/daiyu.json \
  --t2v_model_path ./models/Wan2.2-T2V-A14B \
  --i2v_model_path ./models/Wan2.2-I2V-A14B \
  --lora_weight_path ./models/StoryMem/Wan2.2-MI2V-A14B \
  --size "832*480" \
  --max_memory_size 8 \
  --output_dir "${OUTPUT_PATH}" \
  --offload_model \
  --convert_model_dtype \
  --t5_cpu \
  --lora_rank 128 \
  --mi2v \
  --t2v_first_shot
