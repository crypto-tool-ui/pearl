#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out_dir=$(mktemp -d "${TMPDIR:-/tmp}/pearl-sm120-sass.XXXXXX")
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

source_file="$repo_root/cuda/sm120_mma.cu"
for required in \
  'mma\.sync\.aligned\.m16n8k32\.row\.col\.s32\.s8\.s8\.s32' \
  'cp\.async\.ca\.shared\.global' \
  'cp\.async\.commit_group' \
  'cp\.async\.wait_group [01]' \
  'ldmatrix\.sync\.aligned\.m8n8\.x4\.shared\.b16' \
  'ldmatrix\.sync\.aligned\.m8n8\.x2\.trans\.shared\.b16' \
  'kMaxWarps = 32' \
  'extern __shared__' \
  'static_assert\(kBlake3Flags == 27u' \
  'static_assert\(kBlake3BlockBytes == 64u'; do
  if ! rg -q "$required" "$source_file"; then
    echo "sm_120a source is missing required contract: $required" >&2
    exit 1
  fi
done

if ! rg -q '__CUDA_ARCH__ != 1200' "$source_file"; then
  echo "sm_120a source is missing its architecture guard" >&2
  exit 1
fi

if rg -qi 'tcgen05\.' "$source_file"; then
  echo "sm_120a source must not use tcgen05" >&2
  exit 1
fi
if rg -qi 'tma\.(load|store|prefetch|mbarrier|copy)|cp\.(async|reduce)\.bulk\.tensor' "$source_file"; then
  echo "sm_120a source must not use TMA" >&2
  exit 1
fi

cubin="$out_dir/pearl_sm120_mma.cubin"
sass="$out_dir/pearl_sm120_mma.sass"
"$nvcc_bin" -std=c++17 -O3 -cubin -arch=sm_120a \
  -o "$cubin" "$repo_root/cuda/sm120_mma.cu"
"$cuobjdump_bin" -sass "$cubin" > "$sass"

for instruction in IMMA LDSM LDGSTS; do
  if ! rg -q "$instruction" "$sass"; then
    echo "sm_120a SASS is missing $instruction" >&2
    exit 1
  fi
  printf '%s=' "$instruction"
  rg -c "$instruction" "$sass"
done
