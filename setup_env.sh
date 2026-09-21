#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/videogen_env.sh"

if [[ "${VIDEOGEN_SKIP_PIP:-0}" != "1" ]]; then
  echo "[setup] installing Python dependencies from requirements.txt"
  python -m pip install -r "${SCRIPT_DIR}/requirements.txt"
fi

python "${SCRIPT_DIR}/tools/prepare_models.py" "$@"
echo "[setup] environment is ready."
