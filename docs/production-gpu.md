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

All three targets use a 128x128x64 CTA, four 64x64 warp tiles, and 128 independent
8x16 proof tiles per CTA. Each lane owns 128 cumulative int32 accumulators and its
16-word transcript. Two shared-memory stages use 32 KiB. T4 prefetches 16-byte global
loads through registers; H100 and Blackwell use cp.async and m16n8k32 rather than
T4's m8n8k16. The H100 path is MMA, not WGMMA; no peak-performance claim is made.

The proof pattern, rank=128, K=4096, keyed BLAKE3, target endianness and official
CPU proof verifier are preserved. The scan accepts only complete 128x128 groups,
the exact 8x16 periodic pattern, and K/rank multiples of 64. Unsupported shapes
fail explicitly. The small-tile APIs remain available for oracle tests.

A/B and their noise stay resident for an attempt. The device computes tile indices
from a banded raster; the CPU no longer allocates and transfers row/column arrays
for every mining batch. Launch size is four CTAs per SM, bounded to 32..1024 CTAs;
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

Matrix generation and commitment hashing run on a dedicated CPU worker with one
attempt of lookahead, overlapping the current GPU scan. The first attempt still
pays the full preparation cost. Logs distinguish CPU preparation duration from
CPU wait time; whole-attempt hashrate includes the actual wait and GPU preparation,
without counting overlapped CPU work twice. Stale prepared attempts are discarded.
This uses approximately one additional GiB of matrix RAM, plus hashing scratch.

Matrix generation and commitment hashing still run on the CPU, and Merkle proof
construction remains a CPU cost. These can limit whole-miner throughput on fast
GPUs even when scan throughput improves. Queued proof tasks retain their matrix
allocation until verified/discarded; RAM demand can exceed the ~1 GiB current
attempt. This implementation does not claim full GPU job preparation or persistent
cross-job execution.

## Validation and limits

- `cargo test --locked`: CPU reference, scan continuation/cancellation, proof-queue
  error propagation, seed rotation, and protocol tests.
- `python3 tools/verify_production_kernel.py`: executes the actual CUDA source
  control flow using CPU-emulated PTX fragments. Checks all lanes against scalar
  matrix multiplication at K=128 and K=2176, staging, transcript wrap/rotation,
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
