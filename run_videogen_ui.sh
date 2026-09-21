#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
source ./videogen_env.sh

export VIDEOGEN_WORKSPACE="${VIDEOGEN_WORKSPACE:-${VIDEOGEN_ROOT}/.runtime/videogen_ui}"
export VIDEOGEN_PORT="${VIDEOGEN_PORT:-7880}"
export VIDEOGEN_RUNNER="${VIDEOGEN_RUNNER:-real}"
export VIDEOGEN_MAX_WORKERS="${VIDEOGEN_MAX_WORKERS:-50}"
export VIDEOGEN_REFERENCE_VIDEO_PUBLISHER="${VIDEOGEN_REFERENCE_VIDEO_PUBLISHER:-cloudflare_tunnel}"

LOG_FILE="${VIDEOGEN_WORKSPACE}/server.log"
PID_FILE="${VIDEOGEN_WORKSPACE}/server.pid"
mkdir -p "${VIDEOGEN_WORKSPACE}"

port_pid() {
  ss -ltnp 2>/dev/null \
    | awk -v port=":${VIDEOGEN_PORT}" '$4 ~ port {print $0}' \
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
  tr '\0' ' ' <"/proc/${pid}/cmdline" | grep -q 'videogen_ui'
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
    echo "videogen_ui is already stopped."
    return 0
  fi
  if ! is_notebook_pid "${pid}"; then
    echo "Port ${VIDEOGEN_PORT} is used by PID ${pid}, but it is not videogen_ui." >&2
    return 1
  fi
  kill "${pid}"
  for _ in {1..40}; do
    if ! pid_alive "${pid}"; then
      rm -f "${PID_FILE}"
      echo "videogen_ui stopped (PID ${pid})."
      return 0
    fi
    sleep 0.25
  done
  kill -KILL "${pid}"
  rm -f "${PID_FILE}"
  echo "videogen_ui killed (PID ${pid})."
}

start_server() {
  local pid
  pid="$(port_pid || true)"
  if [[ -n "${pid}" ]]; then
    echo "Port ${VIDEOGEN_PORT} is already in use by PID ${pid}. Use restart." >&2
    return 1
  fi
  setsid python -m videogen_ui start \
    --host 0.0.0.0 \
    --port "${VIDEOGEN_PORT}" \
    --workspace "${VIDEOGEN_WORKSPACE}" \
    --runner "${VIDEOGEN_RUNNER}" \
    </dev/null \
    >"${LOG_FILE}" 2>&1 &
  pid=$!
  printf '%s\n' "${pid}" >"${PID_FILE}"
  for _ in {1..40}; do
    if curl --noproxy '*' --max-time 2 --fail --silent "http://127.0.0.1:${VIDEOGEN_PORT}/" >/dev/null 2>&1; then
      echo "videogen_ui started (PID ${pid})."
      echo "Open: http://localhost:${VIDEOGEN_PORT}"
      echo "Log: ${LOG_FILE}"
      return 0
    fi
    if ! pid_alive "${pid}"; then
      break
    fi
    sleep 0.5
  done
  echo "videogen_ui failed to start. Check ${LOG_FILE}." >&2
  tail -n 40 "${LOG_FILE}" >&2 || true
  return 1
}

status_server() {
  local pid
  pid="$(port_pid || true)"
  if [[ -n "${pid}" ]] && is_notebook_pid "${pid}"; then
    echo "videogen_ui is running (PID ${pid}, port ${VIDEOGEN_PORT})."
    echo "Open: http://localhost:${VIDEOGEN_PORT}"
    echo "Log: ${LOG_FILE}"
  else
    echo "videogen_ui is stopped."
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
    exec python -m videogen_ui start "$@"
    ;;
esac
