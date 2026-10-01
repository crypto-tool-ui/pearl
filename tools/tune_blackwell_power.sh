#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo 'usage: tune_blackwell_power.sh /path/to/pearl-miner [device]' >&2
  exit 2
fi
binary=$(realpath "$1")
device=${2:-0}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
[[ -x "$binary" ]] || { echo "miner is not executable: $binary" >&2; exit 2; }
old_limit=$(nvidia-smi -i "$device" --query-gpu=power.limit --format=csv,noheader,nounits | tr -d '[:space:]')
[[ "$old_limit" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "cannot read original power limit: $old_limit" >&2; exit 2; }
out="blackwell-power-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$out"
restore() { nvidia-smi -i "$device" -pl "$old_limit" >/dev/null || echo 'WARNING: failed to restore GPU power limit' >&2; }
trap restore EXIT INT TERM
printf 'device=%s original_power_limit_w=%s waves=%s band=%s matrix=%sx%sx%s\n' "$device" "$old_limit" "${SCAN_WAVES:-16}" "${SCAN_BAND:-8}" "${MATRIX_M:-131072}" "${MATRIX_N:-131072}" "${MATRIX_K:-4096}" > "$out/manifest.txt"
nvidia-smi -i "$device" -q -d PERFORMANCE > "$out/performance-before.txt"
for watts in 350 400 450 500 600; do
  echo "Testing ${watts} W; original limit ${old_limit} W will be restored on exit"
  nvidia-smi -i "$device" -pl "$watts"
  nvidia-smi -i "$device" -q -d PERFORMANCE > "$out/performance-${watts}W.txt"
  nvidia-smi -i "$device" dmon -s pucm -d 1 > "$out/dmon-${watts}W.csv" 2>&1 &
  monitor=$!
  set +e
  python3 "$repo_root/tools/profile_production.py" --binary "$binary" --device "$device" \
    --preset blackwell --repeats 3 --output "$out/scan-${watts}W"
  result=$?
  set -e
  kill -INT "$monitor" 2>/dev/null || true
  wait "$monitor" 2>/dev/null || true
  if [[ $result -ne 0 ]]; then exit "$result"; fi
done
nvidia-smi -i "$device" -q -d PERFORMANCE > "$out/performance-after.txt"
python3 "$repo_root/tools/summarize_power_sweep.py" "$out"
echo "Power sweep saved under $out; original power limit restored on exit."
