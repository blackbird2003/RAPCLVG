#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
source ./storymem_env.sh

export STORYMEM_WEB_WORKSPACE="${STORYMEM_WEB_WORKSPACE:-${STORYMEM_DATA}/web}"
export STORYMEM_WEB_PORT="${STORYMEM_WEB_PORT:-7860}"
export STORYMEM_REFERENCE_VIDEO_PUBLISHER="${STORYMEM_REFERENCE_VIDEO_PUBLISHER:-tmpfiles}"

PID_FILE="${STORYMEM_WEB_WORKSPACE}/storymem-web.pid"
LOG_FILE="${STORYMEM_WEB_WORKSPACE}/server.log"
HEALTH_URL="http://127.0.0.1:${STORYMEM_WEB_PORT}/api/health"

mkdir -p "${STORYMEM_WEB_WORKSPACE}"

health_json() {
  curl --noproxy '*' --max-time 3 --fail --silent "${HEALTH_URL}" 2>/dev/null
}

json_field() {
  local field="$1"
  python -c 'import json,sys; print(json.load(sys.stdin).get(sys.argv[1], ""))' "${field}"
}

expected_source() {
  python -m storymem_web.runtime --field source_fingerprint
}

port_open() {
  timeout 2 bash -c ": </dev/tcp/127.0.0.1/${STORYMEM_WEB_PORT}" >/dev/null 2>&1
}

pid_value() {
  [[ -f "${PID_FILE}" ]] || return 1
  local value
  value="$(cat "${PID_FILE}" 2>/dev/null || true)"
  [[ "${value}" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "${value}"
}

pid_alive() {
  local pid="$1"
  kill -0 "${pid}" 2>/dev/null
}

pid_is_storymem_web() {
  local pid="$1"
  local command_file="/proc/${pid}/cmdline"
  [[ ! -r "${command_file}" ]] && return 0
  tr '\0' ' ' <"${command_file}" | grep -q 'uvicorn storymem_web.app:app'
}

print_locations() {
  echo "Open through SSH forwarding: http://localhost:${STORYMEM_WEB_PORT}"
  echo "Health: ${HEALTH_URL}"
  echo "Log: ${LOG_FILE}"
}

web_status() {
  local health expected running pid=""
  expected="$(expected_source)"
  if health="$(health_json)"; then
    running="$(printf '%s' "${health}" | json_field source_fingerprint)"
    pid="$(printf '%s' "${health}" | json_field pid)"
    if [[ "${running}" == "${expected}" ]]; then
      echo "StoryMem Web UI is running and current (PID ${pid}, source ${running})."
    else
      echo "StoryMem Web UI is STALE (PID ${pid})." >&2
      echo "Running source: ${running:-unknown}" >&2
      echo "Current source: ${expected}" >&2
      print_locations
      return 2
    fi
    print_locations
    return 0
  fi
  if port_open; then
    echo "Port ${STORYMEM_WEB_PORT} is open, but ${HEALTH_URL} is unavailable." >&2
    echo "The process is likely an older StoryMem Web version or unhealthy." >&2
    return 2
  fi
  if pid="$(pid_value 2>/dev/null)" && pid_alive "${pid}"; then
    echo "StoryMem Web PID ${pid} exists but is not healthy. See ${LOG_FILE}." >&2
    return 1
  fi
  echo "StoryMem Web UI is stopped."
  print_locations
  return 3
}

web_start() {
  local health running expected server_pid
  expected="$(expected_source)"
  if health="$(health_json)"; then
    running="$(printf '%s' "${health}" | json_field source_fingerprint)"
    if [[ "${running}" == "${expected}" ]]; then
      echo "StoryMem Web UI is already running and current."
      print_locations
      return 0
    fi
    echo "A stale StoryMem Web UI is running. Use './storymem web restart'." >&2
    echo "Running source: ${running:-unknown}; current source: ${expected}" >&2
    return 2
  fi
  if port_open; then
    echo "Port ${STORYMEM_WEB_PORT} is already occupied without a valid health response." >&2
    return 1
  fi
  if server_pid="$(pid_value 2>/dev/null)"; then
    if pid_alive "${server_pid}"; then
      echo "PID file points to live but unhealthy process ${server_pid}. Use restart." >&2
      return 1
    fi
    rm -f "${PID_FILE}"
  fi

  setsid ./run_storymem_web.sh </dev/null >>"${LOG_FILE}" 2>&1 &
  server_pid=$!
  printf '%s\n' "${server_pid}" >"${PID_FILE}"

  for _ in {1..120}; do
    if health="$(health_json)"; then
      running="$(printf '%s' "${health}" | json_field source_fingerprint)"
      if [[ "${running}" != "${expected}" ]]; then
        echo "Web UI started with unexpected source ${running}; expected ${expected}." >&2
        return 1
      fi
      echo "StoryMem Web UI started (PID ${server_pid}, source ${running})."
      print_locations
      return 0
    fi
    if ! pid_alive "${server_pid}"; then
      break
    fi
    sleep 0.5
  done

  rm -f "${PID_FILE}"
  echo "StoryMem Web UI failed to start. Check ${LOG_FILE}." >&2
  tail -n 30 "${LOG_FILE}" >&2 || true
  return 1
}

web_stop() {
  local pid
  if ! pid="$(pid_value 2>/dev/null)"; then
    if health_json >/dev/null; then
      echo "A Web UI responds on port ${STORYMEM_WEB_PORT}, but no managed PID file exists." >&2
      echo "Refusing to stop an unidentified process." >&2
      return 1
    fi
    echo "StoryMem Web UI is already stopped."
    return 0
  fi
  if ! pid_alive "${pid}"; then
    rm -f "${PID_FILE}"
    echo "Removed stale PID file for ${pid}."
    return 0
  fi
  if ! pid_is_storymem_web "${pid}"; then
    echo "PID ${pid} does not look like StoryMem Uvicorn; refusing to stop it." >&2
    return 1
  fi

  kill -TERM "${pid}"
  for _ in {1..60}; do
    if ! pid_alive "${pid}"; then
      rm -f "${PID_FILE}"
      echo "StoryMem Web UI stopped (PID ${pid})."
      return 0
    fi
    sleep 0.5
  done
  echo "StoryMem Web PID ${pid} did not exit after 30 seconds; sending KILL." >&2
  kill -KILL "${pid}"
  rm -f "${PID_FILE}"
}

web_logs() {
  if [[ "${1:-}" == "--follow" || "${1:-}" == "-f" ]]; then
    exec tail -n 100 -f "${LOG_FILE}"
  fi
  tail -n 100 "${LOG_FILE}"
}

action="${1:-status}"
shift || true
case "${action}" in
  start) web_start ;;
  stop) web_stop ;;
  restart) web_stop && web_start ;;
  status) web_status ;;
  logs) web_logs "$@" ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs [-f]}" >&2
    exit 2
    ;;
esac
