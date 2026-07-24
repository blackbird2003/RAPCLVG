#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/storymem_env.sh"

OUTPUT_PATH="${1:-results/little_prince_seedance_smoke}"
mkdir -p "${OUTPUT_PATH}"

ARGS=(
  seedance_pipeline.py
  --story_script_path ./story/little_prince.json \
  --output_dir "${OUTPUT_PATH}" \
  --max_shots "${MAX_SHOTS:-2}" \
  --duration "${DURATION:-4}" \
  --ratio "16:9" \
  --max_memory_size 10 \
  --fix 3 \
  --retrieval_top_k 2 \
  --poll_interval 10 \
  --max_wait_seconds 1800
)

if [[ "${PROMPT_RETRIEVAL:-1}" == "1" ]]; then
  ARGS+=(--prompt_retrieval)
fi

if [[ "${LLM_MEMORY_QUERY:-0}" == "1" ]]; then
  ARGS+=(--llm_memory_query)
fi

if [[ "${ENHANCED_TEXT_PROMPT:-0}" == "1" ]]; then
  ARGS+=(--enhanced_text_prompt)
fi

if [[ "${RESUME:-0}" == "1" ]]; then
  ARGS+=(--resume)
fi

python "${ARGS[@]}"
