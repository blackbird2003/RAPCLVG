#!/usr/bin/env bash
set -euo pipefail

source /root/world_model_projects/StoryMem/storymem_env.sh

NGPUS="${NGPUS:-4}"
MASTER_PORT="${MASTER_PORT:-9999}"
STORY_SCRIPT_PATH="${STORY_SCRIPT_PATH:-./story/daiyu.json}"
PROMPT_RETRIEVAL="${PROMPT_RETRIEVAL:-1}"
OUTPUT_PATH="${1:-results/daiyu_prompt_retrieval_multigpu_${NGPUS}gpu}"
mkdir -p "${OUTPUT_PATH}"

DISTRIBUTED_ARGS=(
  --nproc_per_node="${NGPUS}"
  --master_port="${MASTER_PORT}"
)

COMMON_ARGS=(
  --story_script_path "${STORY_SCRIPT_PATH}"
  --t2v_model_path ./models/Wan2.2-T2V-A14B
  --i2v_model_path ./models/Wan2.2-I2V-A14B
  --lora_weight_path ./models/StoryMem/Wan2.2-MI2V-A14B
  --size "832*480"
  --max_memory_size 10
  --output_dir "${OUTPUT_PATH}"
  --dit_fsdp
  --t5_fsdp
  --ulysses_size "${NGPUS}"
  --offload_model
  --lora_rank 128
  --mi2v
)

if [[ "${PROMPT_RETRIEVAL}" == "1" ]]; then
  COMMON_ARGS+=(
    --prompt_retrieval
    --retrieval_top_k 2
    --retrieval_frame_weight 0.7
    --retrieval_video_weight 0.3
    --retrieval_clip_device cpu
  )
fi

python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" pipeline.py "${COMMON_ARGS[@]}" --t2v_first_shot
python -m torch.distributed.run "${DISTRIBUTED_ARGS[@]}" pipeline.py "${COMMON_ARGS[@]}"
