# Resident grouped mining

The previous T4 loop assigned an entire CTA to one 8x16 proof tile, reduced 128
accumulators with a single thread at each rank boundary, and copied every transcript
and digest back for CPU BLAKE3. The Blackwell path prepared one tile at a time; H100
dispatch was disabled. These are structural throughput limits, not a GH/s conversion error.

## References inspected

- `distributed-torch-flashatt-t4/CPPminer/src/cuda/cutlass/cp_cutlass_gemm_types.h`:
  Turing uses 128x128x64 CTAs, 32x64 warp tiles, m8n8k16, two stages.
- Its `mma_milestone.h` and `gemm_inline_xor_kernel.h` keep the cumulative accumulator
  across rank boundaries and integrate transcript updates and jackpot checking.
- `spark-pearl-miner/cuda/gemm/gemm_tma.cuh` maps a complete 8x16 periodic proof tile
  to one lane of a 64x64 warp tile and filters target hits on the GPU.
- IAminer's vendored CUDA source uses large CTAs and two stages but requires SM80+
  and rank 256. B200 Engine8 uses SM100 TMEM/two-SM instructions. Neither is a
  drop-in T4 kernel; their proof layouts also differ.

The new `cuda/production.cu` implements these general design principles against
this repository's existing V3 arithmetic. It does not import those projects'
build scripts, runners, binary bundles, or kernel dependencies.

## Implementation

All three targets use a 128x128x64 CTA and 128 independent 8x16 proof tiles.
All three targets use eight 32x64 warp tiles (256 threads), with 64 accumulators per lane.
Pairs of warps compute the two 4x16 halves of a proof. Since transcript XOR and
rotation are linear, their partial transcripts are merged once after the K loop.
The merge reuses matrix staging memory after the final reader barrier; there is
no extra barrier at each rank boundary. CUDA 13 compiles this to 128 registers per
thread with zero spills, using a two-block launch bound (previously 227 registers
with 128-thread CTAs). Hardware occupancy and speed still require measurement.

Two shared-memory stages use 32 KiB. T4 prefetches 16-byte global loads through
registers; H100 and Blackwell use cp.async and m16n8k32 rather than T4's m8n8k16.
The H100 path is MMA, not WGMMA; no peak-performance claim is made.

The proof pattern, rank=128, K=4096, keyed BLAKE3, target endianness and official
CPU proof verifier are preserved. The scan accepts only complete 128x128 groups,
the exact 8x16 periodic pattern, and K/rank multiples of 64. Unsupported shapes
fail explicitly. The small-tile APIs remain available for oracle tests.

A/B and their noise stay resident for an attempt. The device computes tile indices
from a banded raster; the CPU no longer allocates and transfers row/column arrays
for every mining batch. Launch size is sixteen CTAs per SM, bounded
to 32..4096 CTAs. The larger batch amortizes counter readback and launch costs;
band height is 4 on T4 and 8 on H100/Blackwell. These are initial bounded defaults,
not device-measured autotuning results.

Only matching candidates return to the host. A 256-record queue reports its full
hit count even on overflow; overflowing launches are replayed as disjoint halves,
so no candidates are silently discarded. A single CTA has at most 128 hits.
The counter and target still involve host/device synchronization once per launch.

CPU proof construction and official verification run on a separate thread with
a bounded queue of two tasks. Full queues apply backpressure. The scan continues
after shares and derives fresh matrix seeds after exhausting an attempt without
changing the pool header. Generation tags travel with jobs; stale candidates are
checked before proof construction and again before submission.

On H100 and RTX PRO 6000 Blackwell, deterministic matrix recipes are generated
on the GPU using the exact SplitMix word/byte contract from the CPU reference.
Keyed BLAKE3 roots are reduced on the same device buffers before noise is applied
in place. Only the two 32-byte roots return to the CPU for seed derivation; no
1 GiB host allocation or matrix upload occurs per attempt. The CPU matrix storage
is lazy and shared: the proof worker regenerates the exact original bytes only
when a candidate needs an official proof. Stale unused recipes cost no matrix RAM.
The GPU hash parent block length is 64 bytes (two 32-byte CVs); CPU-source tests
cover this explicitly, fixing a previously unused device-hash implementation bug.

T4 retains CPU preparation with one-attempt lookahead. Logs distinguish CPU
preparation time from CPU wait time; the reported average includes actual wait
and GPU setup. Device setup is currently sequential with scan, not double buffered
on the GPU. Merkle proof construction remains on the CPU and can cost seconds on
a share hit, while scanning continues via the bounded proof queue.

The requested 420 TH/s (RTX PRO 6000) and 700 TH/s (H100) are performance targets,
not measured results. B200 SM100 TMEM/two-SM kernels cannot be reused as SM120
kernels. H100 still uses the verified MMA path; this change does not introduce a
WGMMA or persistent TMA kernel. Benchmark scan throughput and live total throughput
separately before drawing performance conclusions.

## Validation and limits

- `cargo test --locked`: CPU reference, scan continuation/cancellation, proof-queue
  error propagation, seed rotation, and protocol tests.
- `python3 tools/verify_device_setup.py`: runs actual CUDA generator and hash
  source on the CPU against Rust vectors for 1, 2, 3, 17 and 512 chunks, including
  seed wraparound, byte packing and Merkle root flags.
- `python3 tools/verify_production_kernel.py`: executes the actual CUDA source
  control flow using CPU-emulated PTX fragments. Checks all lanes against scalar
  matrix multiplication at K=128, K=2176 and K=4096, staging, transcript wrap/rotation,
  target filtering, raster offsets and queue bounds for both instruction branches.
  This does not simulate CUDA scheduling, async-copy timing, or hardware faults.
- `bash tools/check_production_sass.sh`: compiles sm_75, sm_90a, sm_120a, verifies
  IMMA instructions, requires async-copy instructions only on newer architectures,
  and rejects register spills.
- `bash tools/run_production_acceptance.sh`: requires a real supported GPU and
  fails if prerequisites are absent. Tests the same startup self-test, invalid
  ranges/stale jobs, then reports resident throughput on an 8192x8192x4096 fixture.
- `pearl-miner --gpu-benchmark --device 0`: runs startup self-tests, then a ten-second
  scan of 131072x131072x4096 resident matrices, cycling the raster if exhausted.
  No pool, wallet or network connection is needed.

Hardware acceptance, sustained hashrate and live-pool acceptance must be measured
on each target. Compilation and CPU emulation do not establish any of those.
Use compute-sanitizer memcheck/racecheck/synccheck on the acceptance test binary
where available before a long mining run.

## Operational acceptance

- Run `pearl-miner --selftest --device 0` on the actual target. This now includes
  GPU candidate -> official verified proof -> V2 encoding, in addition to tile tests.
- Every production batch reports a device-written completed-tile count, checked
  against the requested range. A mismatch is fatal, never counted as hashrate.
- Default INFO output aggregates mining work over ten-second windows across attempts;
  share-response counters are logged every thirty seconds. Per-attempt details are DEBUG.
- A real pool accepted response is still required for live acceptance. Synthetic
  self-tests do not establish pool acceptance or sustained production stability.
- Run a sustained session and inspect accepted/rejected/stale totals, proof errors,
  temperatures, clocks and restarts. A few seconds with zero hits is inconclusive:
  at 262 TH/s and the observed target, expected spacing is about 34 seconds.
- Submission acknowledgement times out at 30 seconds. Failure closes the worker,
  cancels background work and propagates nonzero exit status to the supervisor.
