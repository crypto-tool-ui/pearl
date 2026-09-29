#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out_dir=$(mktemp -d "${TMPDIR:-/tmp}/pearl-sm90-sass.XXXXXX")
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

source_file="$repo_root/cuda/sm90_wgmma.cu"
for required in \
  'wgmma\.mma_async\.sync\.aligned\.m64n8k32\.s32\.s8\.s8' \
  'wgmma\.fence\.sync\.aligned' \
  'wgmma\.commit_group\.sync\.aligned' \
  'wgmma\.wait_group\.sync\.aligned 0' \
  'cp\.async\.bulk\.shared::cta\.global\.mbarrier::complete_tx::bytes' \
  'mbarrier\.arrive\.expect_tx\.shared::cta\.b64' \
  'mbarrier\.try_wait\.acquire\.cta\.shared::cta\.b64' \
  'const bool producer = warp_id == warpgroups \* 4' \
  'const uint32_t consumer_lane = thread & 127' \
  'kMaxWarpgroups = 8' \
  'static_assert\(kBlake3Flags == 27u' \
  'static_assert\(kBlake3BlockBytes == 64u'; do
  if ! rg -q "$required" "$source_file"; then
    echo "sm_90a source is missing required contract: $required" >&2
    exit 1
  fi
done

rg -q '__CUDA_ARCH__ != 900' "$source_file"
if rg -qi 'tcgen05\.' "$source_file"; then
  echo "sm_90a source must not use tcgen05" >&2
  exit 1
fi

cubin="$out_dir/pearl_sm90_wgmma.cubin"
sass="$out_dir/pearl_sm90_wgmma.sass"
"$nvcc_bin" -std=c++17 -O3 -cubin -arch=sm_90a \
  -o "$cubin" "$source_file"
"$cuobjdump_bin" -sass "$cubin" > "$sass"

if ! rg -Eq 'WGMMA|QGMMA' "$sass"; then
  echo "sm_90a SASS is missing WGMMA/QGMMA" >&2
  exit 1
fi
if ! rg -Eq 'LDGSTS|CP_ASYNC' "$sass"; then
  echo "sm_90a SASS is missing an async global-to-shared transfer" >&2
  exit 1
fi
printf 'WGMMA='; rg -ci 'WGMMA|QGMMA' "$sass"
