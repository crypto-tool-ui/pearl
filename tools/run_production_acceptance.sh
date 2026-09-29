#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v nvidia-smi >/dev/null || { echo 'Real NVIDIA GPU/driver required' >&2; exit 2; }
nvidia-smi --query-gpu=index,name,compute_cap --format=csv
export PEARL_GPU_DEVICE="${PEARL_GPU_DEVICE:-0}"
export PEARL_REQUIRE_GPU_ACCEPTANCE=1
cargo test --release -p pearl-gpu --features gpu --test production -- --nocapture
