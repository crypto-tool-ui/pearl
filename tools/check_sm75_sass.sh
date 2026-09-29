#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out_dir=$(mktemp -d "${TMPDIR:-/tmp}/pearl-sm75-sass.XXXXXX")
trap 'rm -rf "$out_dir"' EXIT

find_cuda_tool() {
  local tool_name=$1
  local explicit_path=$2
  if [[ -n "$explicit_path" ]] && "$explicit_path" --version >/dev/null 2>&1; then
    printf '%s\n' "$explicit_path"
    return 0
  fi
  local cuda_root
  for cuda_root in "${CUDA_PATH:-}" "${CUDA_HOME:-}" "${CUDA_ROOT:-}"; do
    if [[ -n "$cuda_root" ]] && [[ -x "$cuda_root/bin/$tool_name" ]] &&
      "$cuda_root/bin/$tool_name" --version >/dev/null 2>&1; then
      printf '%s\n' "$cuda_root/bin/$tool_name"
      return 0
    fi
  done
  command -v "$tool_name" || true
}

nvcc_bin=$(find_cuda_tool nvcc "${NVCC:-}")
cuobjdump_bin=$(find_cuda_tool cuobjdump "${CUOBJDUMP:-}")
[[ -n "$nvcc_bin" ]] || { echo "nvcc is required" >&2; exit 1; }
[[ -n "$cuobjdump_bin" ]] || { echo "cuobjdump is required" >&2; exit 1; }

source_file="$repo_root/cuda/sm75_mma.cu"
rg -q '__CUDA_ARCH__ != 750' "$source_file"
rg -q 'mma\.sync\.aligned\.m8n8k16\.row\.col\.s32\.s8\.s8\.s32' "$source_file"
cubin="$out_dir/pearl_sm75_mma.cubin"
sass="$out_dir/pearl_sm75_mma.sass"
"$nvcc_bin" -std=c++17 -O3 -cubin -arch=sm_75 -o "$cubin" "$source_file"
"$cuobjdump_bin" -sass "$cubin" > "$sass"
if ! rg -q 'IMMA' "$sass"; then
  echo 'sm_75 SASS is missing Tensor Core IMMA' >&2
  exit 1
fi
if rg -q 'LDGSTS|WGMMA|QGMMA' "$sass"; then
  echo 'sm_75 SASS contains an unsupported architecture instruction' >&2
  exit 1
fi
printf 'sm_75 IMMA instructions: '
rg -c 'IMMA' "$sass"
