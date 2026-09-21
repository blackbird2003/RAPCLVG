#!/usr/bin/env bash
set -euo pipefail

# Download the smallest official ComfyUI H3 bundle that supports both:
# - FL2VA: text/image/first-last-frame generation
# - Ref2VA: multimodal reference-to-video generation
ROOT="${MINIMAX_H3_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.runtime/models/Comfy-Org/MiniMax-H3}"
REPO_URL="https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3/resolve/master"

# The server's local proxy is useful for some services, but throttles and
# intermittently closes ModelScope CDN transfers. Override with
# MINIMAX_H3_USE_PROXY=1 when a proxy is genuinely required.
if [[ "${MINIMAX_H3_USE_PROXY:-0}" != "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
fi

FILES=(
  "20970379616 diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
  "20970379616 diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
  "15687142551 text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  "5207808496 vae/minimax_h3_video_vae_fp16.safetensors"
  "605254808 vae/minimax_h3_audio_vae_fp32.safetensors"
  "1956193000 loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
  "1956193000 loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
)

download() {
  local expected="$1"
  local relative="$2"
  local target="$ROOT/$relative"
  local partial="$target.part"
  local actual

  mkdir -p "$(dirname "$target")"
  if [[ -f "$target" ]] && [[ "$(stat -c%s "$target")" == "$expected" ]]; then
    printf 'Already complete: %s\n' "$relative"
    return
  fi
  if [[ -f "$target" ]]; then
    printf 'Removing unexpected-size file: %s\n' "$relative"
    rm -f "$target"
  fi

  printf 'Downloading: %s\n' "$relative"
  local transfer_attempt=1
  while true; do
    if curl --fail --location --continue-at - \
      --retry 12 --connect-timeout 30 \
      --output "$partial" "$REPO_URL/$relative"; then
      break
    fi
    printf 'Transfer interrupted for %s; resuming in 10s (attempt %s).\n' "$relative" "$transfer_attempt" >&2
    transfer_attempt=$((transfer_attempt + 1))
    sleep 10
  done
  actual="$(stat -c%s "$partial")"
  if [[ "$actual" != "$expected" ]]; then
    printf 'Size verification failed for %s: expected %s, got %s\n' "$relative" "$expected" "$actual" >&2
    return 1
  fi
  mv "$partial" "$target"
  printf 'Completed: %s\n' "$relative"
}

for item in "${FILES[@]}"; do
  read -r expected relative <<<"$item"
  download "$expected" "$relative"
done

printf 'MiniMax H3 Comfy bundle is ready at %s\n' "$ROOT"
