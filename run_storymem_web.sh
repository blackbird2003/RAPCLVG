#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
source ./storymem_env.sh

export STORYMEM_WEB_WORKSPACE="${STORYMEM_WEB_WORKSPACE:-${STORYMEM_DATA}/web}"
export STORYMEM_REFERENCE_VIDEO_PUBLISHER="${STORYMEM_REFERENCE_VIDEO_PUBLISHER:-tmpfiles}"
exec python -m uvicorn storymem_web.app:app \
  --host "${STORYMEM_WEB_HOST:-0.0.0.0}" \
  --port "${STORYMEM_WEB_PORT:-7860}"
