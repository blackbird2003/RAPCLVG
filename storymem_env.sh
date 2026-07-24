#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STORYMEM_ROOT="${STORYMEM_ROOT:-${SCRIPT_DIR}}"
if [[ -z "${STORYMEM_DATA:-}" ]]; then
  if [[ -d /root/autodl-tmp ]]; then
    export STORYMEM_DATA="/root/autodl-tmp"
  else
    export STORYMEM_DATA="${STORYMEM_ROOT}/.runtime"
  fi
fi

export HF_HOME="${STORYMEM_DATA}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
unset TRANSFORMERS_CACHE
export PIP_CACHE_DIR="${STORYMEM_DATA}/cache/pip"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}"

if [[ -z "${SEEDANCE_API_KEY:-}" && -f "${HOME}/.seedance_api_key" ]]; then
  export SEEDANCE_API_KEY="$(python3 -c 'from pathlib import Path; print(Path("'"${HOME}"'/.seedance_api_key").read_text(encoding="utf-8-sig").strip())' 2>/dev/null || tr -d '\r\n' < "${HOME}/.seedance_api_key")"
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" && -f "${HOME}/.deepseek_api_key" ]]; then
  export DEEPSEEK_API_KEY="$(python3 -c 'from pathlib import Path; print(Path("'"${HOME}"'/.deepseek_api_key").read_text(encoding="utf-8-sig").strip())' 2>/dev/null || tr -d '\r\n' < "${HOME}/.deepseek_api_key")"
fi

__storymem_had_nounset=0
case "$-" in
  *u*) __storymem_had_nounset=1; set +u ;;
esac

if [[ -n "${STORYMEM_VENV:-}" && -f "${STORYMEM_VENV}/bin/activate" ]]; then
  source "${STORYMEM_VENV}/bin/activate"
elif [[ -f /root/autodl-tmp/venvs/storymem/bin/activate ]]; then
  source /root/autodl-tmp/venvs/storymem/bin/activate
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate py311
elif [[ -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda/etc/profile.d/conda.sh"
  conda activate py311
elif [[ -f /home/wxh/miniconda/etc/profile.d/conda.sh ]]; then
  source /home/wxh/miniconda/etc/profile.d/conda.sh
  conda activate py311
elif [[ -f /home/wxh/miniconda/bin/activate ]]; then
  source /home/wxh/miniconda/bin/activate py311
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate py311
else
  echo "No StoryMem Python environment found. Set STORYMEM_VENV or install/activate py311." >&2
  return 1 2>/dev/null || exit 1
fi

if [[ "${__storymem_had_nounset}" == "1" ]]; then
  set -u
fi
unset __storymem_had_nounset

cd "${STORYMEM_ROOT}"
