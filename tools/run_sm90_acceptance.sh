#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "an NVIDIA driver is required for M4 acceptance" >&2
  exit 2
}

device=$(nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader,nounits |
  awk -F, '{gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); if ($2 == "9.0") {print $1; exit}}')
[[ -n "$device" ]] || {
  echo "an sm_90 GPU is required for M4 acceptance" >&2
  exit 2
}

export PEARL_GPU_DEVICE="$device"
export PEARL_REQUIRE_GPU_ACCEPTANCE=1
cargo test -p pearl-gpu --features gpu sm90_wgmma -- --nocapture
