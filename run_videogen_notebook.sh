#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
source ./storymem_env.sh

export VIDEOGEN_NOTEBOOK_WORKSPACE="${VIDEOGEN_NOTEBOOK_WORKSPACE:-${STORYMEM_ROOT}/.runtime/videogen_notebook}"
export VIDEOGEN_NOTEBOOK_PORT="${VIDEOGEN_NOTEBOOK_PORT:-7870}"
export VIDEOGEN_NOTEBOOK_RUNNER="${VIDEOGEN_NOTEBOOK_RUNNER:-real}"
export VIDEOGEN_NOTEBOOK_MAX_WORKERS="${VIDEOGEN_NOTEBOOK_MAX_WORKERS:-50}"
export STORYMEM_REFERENCE_VIDEO_PUBLISHER="${STORYMEM_REFERENCE_VIDEO_PUBLISHER:-cloudflare_tunnel}"

LOG_FILE="${VIDEOGEN_NOTEBOOK_WORKSPACE}/server.log"
PID_FILE="${VIDEOGEN_NOTEBOOK_WORKSPACE}/server.pid"
mkdir -p "${VIDEOGEN_NOTEBOOK_WORKSPACE}"

port_pid() {
  ss -ltnp 2>/dev/null \
    | awk -v port=":${VIDEOGEN_NOTEBOOK_PORT}" '$4 ~ port {print $0}' \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
    | head -1
}

pid_alive() {
  local pid="$1"
  kill -0 "${pid}" 2>/dev/null
}

pid_valid_alive() {
  local pid="$1"
  [[ "${pid}" =~ ^[0-9]+$ ]] && pid_alive "${pid}"
}

is_notebook_pid() {
  local pid="$1"
  [[ ! -r "/proc/${pid}/cmdline" ]] && return 1
  tr '\0' ' ' <"/proc/${pid}/cmdline" | grep -q 'videogen_notebook'
}

stop_server() {
  local pid=""
  if [[ -f "${PID_FILE}" ]]; then
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  fi
  if [[ -z "${pid}" ]] || ! pid_valid_alive "${pid}"; then
    pid="$(port_pid || true)"
  fi
  if [[ -z "${pid}" ]]; then
    rm -f "${PID_FILE}"
    echo "videogen_notebook is already stopped."
    return 0
  fi
  if ! is_notebook_pid "${pid}"; then
    echo "Port ${VIDEOGEN_NOTEBOOK_PORT} is used by PID ${pid}, but it is not videogen_notebook." >&2
    return 1
  fi
  kill "${pid}"
  for _ in {1..40}; do
    if ! pid_alive "${pid}"; then
      rm -f "${PID_FILE}"
      echo "videogen_notebook stopped (PID ${pid})."
      return 0
    fi
    sleep 0.25
  done
  kill -KILL "${pid}"
  rm -f "${PID_FILE}"
  echo "videogen_notebook killed (PID ${pid})."
}

start_server() {
  local pid
  pid="$(port_pid || true)"
  if [[ -n "${pid}" ]]; then
    echo "Port ${VIDEOGEN_NOTEBOOK_PORT} is already in use by PID ${pid}. Use restart." >&2
    return 1
  fi
  setsid python -m videogen_notebook start \
    --host 0.0.0.0 \
    --port "${VIDEOGEN_NOTEBOOK_PORT}" \
    --workspace "${VIDEOGEN_NOTEBOOK_WORKSPACE}" \
    --runner "${VIDEOGEN_NOTEBOOK_RUNNER}" \
    </dev/null \
    >"${LOG_FILE}" 2>&1 &
  pid=$!
  printf '%s\n' "${pid}" >"${PID_FILE}"
  for _ in {1..40}; do
    if curl --noproxy '*' --max-time 2 --fail --silent "http://127.0.0.1:${VIDEOGEN_NOTEBOOK_PORT}/" >/dev/null 2>&1; then
      echo "videogen_notebook started (PID ${pid})."
      echo "Open: http://localhost:${VIDEOGEN_NOTEBOOK_PORT}"
      echo "Log: ${LOG_FILE}"
      return 0
    fi
    if ! pid_alive "${pid}"; then
      break
    fi
    sleep 0.5
  done
  echo "videogen_notebook failed to start. Check ${LOG_FILE}." >&2
  tail -n 40 "${LOG_FILE}" >&2 || true
  return 1
}

status_server() {
  local pid
  pid="$(port_pid || true)"
  if [[ -n "${pid}" ]] && is_notebook_pid "${pid}"; then
    echo "videogen_notebook is running (PID ${pid}, port ${VIDEOGEN_NOTEBOOK_PORT})."
    echo "Open: http://localhost:${VIDEOGEN_NOTEBOOK_PORT}"
    echo "Log: ${LOG_FILE}"
  else
    echo "videogen_notebook is stopped."
  fi
}

case "${1:-start}" in
  start) shift || true; start_server "$@" ;;
  stop) shift || true; stop_server "$@" ;;
  restart) shift || true; stop_server && start_server "$@" ;;
  status) shift || true; status_server "$@" ;;
  logs)
    shift || true
    if [[ "${1:-}" == "-f" || "${1:-}" == "--follow" ]]; then
      exec tail -n 100 -f "${LOG_FILE}"
    fi
    tail -n 100 "${LOG_FILE}"
    ;;
  *)
    exec python -m videogen_notebook start "$@"
    ;;
esac
