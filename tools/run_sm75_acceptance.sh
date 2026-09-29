#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null; then
  echo "T4 acceptance requires nvidia-smi and an NVIDIA driver" >&2
  exit 2
fi

gpu_index=$(nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader,nounits 2>/dev/null \
  | awk -F, '$2 ~ /^[[:space:]]*7\.5[[:space:]]*$/ {gsub(/[[:space:]]/, "", $1); print $1; exit}')
if [[ -z "$gpu_index" ]]; then
  echo "T4 acceptance requires an sm_75 GPU; detected compute capabilities:" >&2
  nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader >&2 || true
  exit 2
fi

echo "Using sm_75 GPU ordinal ${gpu_index}"
PEARL_GPU_DEVICE="$gpu_index" \
PEARL_REQUIRE_GPU_ACCEPTANCE=1 \
  cargo test -p pearl-gpu --features gpu --test sm75 -- --nocapture
