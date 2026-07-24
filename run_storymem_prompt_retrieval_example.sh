#!/usr/bin/env bash
set -euo pipefail

source /root/world_model_projects/StoryMem/storymem_env.sh

OUTPUT_PATH="${1:-results/daiyu_prompt_retrieval_example}"
mkdir -p "${OUTPUT_PATH}"

COMMON_ARGS=(
  --story_script_path ./story/daiyu.json
  --t2v_model_path ./models/Wan2.2-T2V-A14B
  --i2v_model_path ./models/Wan2.2-I2V-A14B
  --lora_weight_path ./models/StoryMem/Wan2.2-MI2V-A14B
  --size "832*480"
  --max_memory_size 10
  --output_dir "${OUTPUT_PATH}"
  --offload_model
  --convert_model_dtype
  --t5_cpu
  --lora_rank 128
  --mi2v
  --prompt_retrieval
  --retrieval_top_k 2
  --retrieval_frame_weight 0.7
  --retrieval_video_weight 0.3
  --retrieval_clip_device cpu
)

python pipeline.py "${COMMON_ARGS[@]}" --t2v_first_shot
python pipeline.py "${COMMON_ARGS[@]}"
