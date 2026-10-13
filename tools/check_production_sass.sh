#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
out_dir=$(mktemp -d "${TMPDIR:-/tmp}/pearl-production-sass.XXXXXX")
trap 'rm -rf "$out_dir"' EXIT
nvcc_bin=${NVCC:-nvcc}
dump_bin=${CUOBJDUMP:-cuobjdump}
for arch in sm_75 sm_90a sm_120; do
  source=cuda/production.cu
  if [[ "$arch" == sm_90a ]]; then source=cuda/production_sm90.cu; fi
  "$nvcc_bin" -std=c++17 -O3 -lineinfo -Xptxas=-v -cubin -arch="$arch" \
    -o "$out_dir/$arch.cubin" "$source" 2>"$out_dir/$arch.log"
  "$dump_bin" -sass -fun pearl_production_scan "$out_dir/$arch.cubin" >"$out_dir/$arch.sass"
  if [[ "$arch" == sm_90a ]]; then
    rg -q 'IGMMA\.64x128x32\.S8\.S8' "$out_dir/$arch.sass"
    rg -q 'UTMALDG' "$out_dir/$arch.sass"
  else
    rg -q 'IMMA' "$out_dir/$arch.sass"
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
  elif [[ "$arch" == sm_120 ]]; then
    rg -q 'LDGSTS' "$out_dir/$arch.sass"
  fi
  # Keep the compiler's register/resource report visible in CI tuning artifacts.
  cat "$out_dir/$arch.log"
  echo "$arch: Tensor Core instructions verified; zero spills"
done
