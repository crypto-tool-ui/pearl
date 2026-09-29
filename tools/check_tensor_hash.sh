#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
nvcc_bin=${NVCC:-}
if [[ -z "$nvcc_bin" && -n "${CUDA_PATH:-}" ]]; then nvcc_bin="$CUDA_PATH/bin/nvcc"; fi
if [[ -z "$nvcc_bin" && -n "${CUDA_HOME:-}" ]]; then nvcc_bin="$CUDA_HOME/bin/nvcc"; fi
if [[ -z "$nvcc_bin" && -n "${CUDA_ROOT:-}" ]]; then nvcc_bin="$CUDA_ROOT/bin/nvcc"; fi
if [[ -z "$nvcc_bin" ]]; then nvcc_bin=$(command -v nvcc || true); fi
[[ -n "$nvcc_bin" ]] || { echo "nvcc is required" >&2; exit 1; }

source_file="$repo_root/cuda/tensor_hash.cu"
for required in \
  'pearl_tensor_hash_leaf_cv' \
  'pearl_tensor_hash_parent_cv' \
  'kChunkBytes = 1024' \
  'kChunkStart = 1' \
  'kChunkEnd = 2' \
  'kParent = 4' \
  'kRoot = 8' \
  'kKeyedHash = 16'; do
  rg -q "$required" "$source_file" || {
    echo "tensor_hash source is missing required contract: $required" >&2
    exit 1
  }
done

out_dir=${TMPDIR:-/tmp}/pearl-tensor-hash-check
mkdir -p "$out_dir"
for arch in sm_75 sm_90a sm_120a; do
  "$nvcc_bin" -std=c++17 -O3 -cubin -arch="$arch" \
    -o "$out_dir/tensor_hash_${arch}.cubin" "$source_file"
done
printf 'tensor_hash cubins: sm_75 sm_90a sm_120a\n'
