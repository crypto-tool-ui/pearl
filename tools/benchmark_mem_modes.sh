#!/usr/bin/env bash
set -euo pipefail

MINER=${MINER:-target/release/pearl-miner}
DURATION=${DURATION:-30}
REPETITIONS=${REPETITIONS:-3}
OUT=${OUT:-mem-mode-results}
mkdir -p "$OUT"
: >"$OUT/runs.tsv"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 1; }
test -x "$MINER" || { echo "build the miner first: $MINER is missing" >&2; exit 1; }

for mode in fused buffered pipelined; do
  for groups in 3008 6016 12032 24064; do
    for rep in $(seq 1 "$REPETITIONS"); do
      tag="${mode}-g${groups}-r${rep}"
      log="$OUT/$tag.log"
      samples="$OUT/$tag.csv"
      before="$OUT/$tag-before.txt"
      after="$OUT/$tag-after.txt"
      nvidia-smi -q -d PERFORMANCE >"$before"
      nvidia-smi --query-gpu=power.draw,clocks.sm,memory.used --format=csv,noheader,nounits -l 1 >"$samples" &
      sampler=$!
      set +e
      "$MINER" --gpu-benchmark --gpu-benchmark-seconds "$DURATION" \
        --mem-mode "$mode" --groups-per-launch "$groups" >"$log" 2>&1
      status=$?
      set -e
      kill "$sampler" 2>/dev/null || true
      wait "$sampler" 2>/dev/null || true
      nvidia-smi -q -d PERFORMANCE >"$after"
      if (( status != 0 )); then
        echo "$tag failed (exit $status), see $log" >&2
        exit "$status"
      fi
      python3 tools/summarize_mem_mode_run.py "$tag" "$log" "$samples" "$before" "$after" >>"$OUT/runs.tsv"
    done
  done
done
python3 tools/summarize_mem_modes.py "$OUT/runs.tsv"
