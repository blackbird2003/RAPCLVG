#!/usr/bin/env bash
set -euo pipefail

source /root/world_model_projects/StoryMem/storymem_env.sh

DEFAULT_OUTPUT="${DEFAULT_OUTPUT:-results/little_prince_seedance_default_full}"
RETRIEVAL_OUTPUT="${RETRIEVAL_OUTPUT:-results/little_prince_seedance_prompt_retrieval_full}"
MAX_SHOTS="${MAX_SHOTS:-999}"
DURATION="${DURATION:-4}"

mkdir -p /root/autodl-tmp/logs

PROMPT_RETRIEVAL=0 MAX_SHOTS="${MAX_SHOTS}" DURATION="${DURATION}" \
  bash run_seedance_little_prince_smoke.sh "${DEFAULT_OUTPUT}" \
  > /root/autodl-tmp/logs/seedance_little_prince_default_full.log 2>&1

PROMPT_RETRIEVAL=1 MAX_SHOTS="${MAX_SHOTS}" DURATION="${DURATION}" \
  bash run_seedance_little_prince_smoke.sh "${RETRIEVAL_OUTPUT}" \
  > /root/autodl-tmp/logs/seedance_little_prince_prompt_retrieval_full.log 2>&1
