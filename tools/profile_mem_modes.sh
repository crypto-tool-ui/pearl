#!/usr/bin/env bash
set -euo pipefail
MINER=${MINER:-target/release/pearl-miner}
OUT=${OUT:-mem-mode-ncu}
mkdir -p "$OUT"
METRICS='sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'
for mode in fused buffered pipelined; do
  ncu --target-processes all --launch-count 1 --metrics "$METRICS" \
    "$MINER" --gpu-benchmark --gpu-benchmark-seconds 1 --mem-mode "$mode" \
    --groups-per-launch 3008 >"$OUT/$mode.txt" 2>&1
done
