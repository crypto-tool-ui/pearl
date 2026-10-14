#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
out_dir=$(mktemp -d "${TMPDIR:-/tmp}/pearl-production-sass.XXXXXX")
trap 'rm -rf "$out_dir"' EXIT
nvcc_bin=${NVCC:-nvcc}
dump_bin=${CUOBJDUMP:-cuobjdump}
for arch in sm_75 sm_80 sm_86 sm_89 sm_90a sm_120 sm_120_direct sm_120_direct_padded; do
  compile_arch=$arch
  source=cuda/production.cu
  if [[ "$arch" == sm_90a ]]; then source=cuda/production_sm90.cu; fi
  if [[ "$arch" == sm_120 ]]; then source=cuda/production_sm120_tma.cu; fi
  symbol=pearl_production_scan
  if [[ "$arch" == sm_120 ]]; then symbol=pearl_production_scan_sm120_tma; fi
  extra=()
  if [[ "$arch" == sm_75 ]]; then
    extra=(-DPEARL_DIRECT_GLOBAL=1 -maxrregcount=122)
  elif [[ "$arch" == sm_80 || "$arch" == sm_86 || "$arch" == sm_89 || "$arch" == sm_120_direct || "$arch" == sm_120_direct_padded ]]; then
    extra=(-DPEARL_DIRECT_GLOBAL=1 -maxrregcount=128)
  fi
  if [[ "$arch" == sm_120_direct || "$arch" == sm_120_direct_padded ]]; then compile_arch=sm_120; fi
  if [[ "$arch" == sm_120_direct_padded ]]; then extra+=(-DPEARL_DIRECT_SMEM_PAD=1024); fi
  "$nvcc_bin" -std=c++17 -O3 -lineinfo -Xptxas=-v -cubin -arch="$compile_arch" \
    "${extra[@]}" -o "$out_dir/$arch.cubin" "$source" 2>"$out_dir/$arch.log"
  "$dump_bin" -res-usage "$out_dir/$arch.cubin" >"$out_dir/$arch.resources"
  "$dump_bin" -sass -fun "$symbol" "$out_dir/$arch.cubin" >"$out_dir/$arch.sass"
  if [[ "$arch" == sm_90a ]]; then
    rg -q 'IGMMA\.64x128x32\.S8\.S8' "$out_dir/$arch.sass"
    rg -q 'UTMALDG' "$out_dir/$arch.sass"
  elif [[ "$arch" == sm_120 ]]; then
    rg -q 'IMMA\.16832\.S8\.S8' "$out_dir/$arch.sass"
    rg -q 'UTMALDG' "$out_dir/$arch.sass"
    rg -q 'SYNCS' "$out_dir/$arch.sass"
  elif [[ "$arch" == sm_120_direct || "$arch" == sm_120_direct_padded ]]; then
    rg -q 'IMMA' "$out_dir/$arch.sass"
    rg -q 'LDG' "$out_dir/$arch.sass"
    if rg -q 'UTMALDG|LDGSTS' "$out_dir/$arch.sass"; then
      echo 'Unexpected shared-memory operand staging in direct-global sm_120 kernel' >&2
      exit 1
    fi
  else
    rg -q 'IMMA' "$out_dir/$arch.sass"
    if [[ "${extra[*]}" == *PEARL_DIRECT_GLOBAL* ]]; then
      rg -q 'LDG' "$out_dir/$arch.sass"
      if rg -q 'LDGSTS' "$out_dir/$arch.sass"; then
        echo "Unexpected shared-memory operand staging in direct-global $arch kernel" >&2
        exit 1
      fi
    fi
  fi
  if rg -q '[1-9][0-9]* bytes spill (stores|loads)' "$out_dir/$arch.log"; then
    cat "$out_dir/$arch.log" >&2
    echo "Register spilling in $arch production cubin" >&2
    exit 1
  fi
  if [[ "$arch" == sm_75 ]]; then
    if rg -q 'LDGSTS|WGMMA|QGMMA' "$out_dir/$arch.sass"; then
      echo 'Unsupported Turing instruction' >&2; exit 1
    fi
  fi
  # Keep the compiler's register/resource report visible in CI tuning artifacts.
  cat "$out_dir/$arch.log"
  cat "$out_dir/$arch.resources"
  echo "$arch: Tensor Core instructions verified; zero spills"
done
