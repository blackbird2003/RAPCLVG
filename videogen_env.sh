#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VIDEOGEN_ROOT="${VIDEOGEN_ROOT:-${SCRIPT_DIR}}"
if [[ -z "${VIDEOGEN_DATA:-}" ]]; then
  if [[ -n "${VIDEOGEN_EXTERNAL_DATA_DIR:-}" && -d "${VIDEOGEN_EXTERNAL_DATA_DIR}" ]]; then
    export VIDEOGEN_DATA="${VIDEOGEN_EXTERNAL_DATA_DIR}"
  else
    export VIDEOGEN_DATA="${VIDEOGEN_ROOT}/.runtime"
  fi
fi

export HF_HOME="${VIDEOGEN_DATA}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
unset TRANSFORMERS_CACHE
export PIP_CACHE_DIR="${VIDEOGEN_DATA}/cache/pip"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}"

if [[ -z "${SEEDANCE_API_KEY:-}" && -f "${HOME}/.seedance_api_key" ]]; then
  export SEEDANCE_API_KEY="$(python3 -c 'from pathlib import Path; print(Path("'"${HOME}"'/.seedance_api_key").read_text(encoding="utf-8-sig").strip())' 2>/dev/null || tr -d '\r\n' < "${HOME}/.seedance_api_key")"
fi
__videogen_had_nounset=0
case "$-" in
  *u*) __videogen_had_nounset=1; set +u ;;
esac

if [[ -n "${VIDEOGEN_VENV:-}" && -f "${VIDEOGEN_VENV}/bin/activate" ]]; then
  source "${VIDEOGEN_VENV}/bin/activate"
elif [[ -f "${VIDEOGEN_DATA}/venvs/videogen/bin/activate" ]]; then
  source "${VIDEOGEN_DATA}/venvs/videogen/bin/activate"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate py311
elif [[ -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda/etc/profile.d/conda.sh"
  conda activate py311
elif [[ -f "${HOME}/miniconda/bin/activate" ]]; then
  source "${HOME}/miniconda/bin/activate" py311
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate py311
else
  echo "No Python environment found. Set VIDEOGEN_VENV or install/activate py311." >&2
  return 1 2>/dev/null || exit 1
fi

if [[ "${__videogen_had_nounset}" == "1" ]]; then
  set -u
fi
unset __videogen_had_nounset

cd "${VIDEOGEN_ROOT}"
