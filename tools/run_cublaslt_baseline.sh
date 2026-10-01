#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
nvcc_bin=${NVCC:-nvcc}
output=${1:-/tmp/pearl-cublaslt-int8-baseline}
shift || true
"$nvcc_bin" -std=c++17 -O3 -arch=sm_120a -Xptxas=-v \
  "$repo_root/cuda/cublaslt_int8_baseline.cu" -lcublasLt -lcublas \
  -o "$output"
"$output" "$@"
