#!/usr/bin/env bash
set -euo pipefail

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 1; }
device="${PEARL_GPU_DEVICE:-}"
if [[ -z "$device" ]]; then
  device=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -n1 | tr -d '[:space:]')
fi
[[ -n "$device" ]] || { echo "no NVIDIA GPU is visible" >&2; exit 1; }

export PEARL_GPU_DEVICE="$device"
export PEARL_REQUIRE_GPU_ACCEPTANCE=1
cargo test -p pearl-gpu --features gpu m5 -- --nocapture
