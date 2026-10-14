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

The production kernels implement these general design principles against this
repository's existing V3 arithmetic. Blackwell now follows Spark's TMA/mbarrier
producer-consumer design in `cuda/production_sm120_tma.cu`, adapted to the
root miner's 128x128 tile and proof contract. No reference build scripts,
runners, binary bundles, or kernel dependencies are imported.

## Implementation

All three targets use a 128x128x64 CTA and 128 independent 8x16 proof tiles.
T4 uses sixteen 16x64 warp tiles (512 threads), with 32 accumulators per lane.
Blackwell uses four 64x64 warp tiles (four MMA warps plus a producer warpgroup,
256 threads), with 128 accumulators per lane. H100 uses two 64x128 warpgroups
(256 threads), also with 64 accumulators per lane.
Four T4/H100 warps contribute to each proof and merge at each rank milestone into
128 shared transcripts. Blackwell merges two register transcripts after the K loop.
XOR and rotation are linear, so each path preserves the same proof arithmetic.

T4 has one 16 KiB matrix stage, 8 KiB of transcripts and 2 KiB of partial folds.
The next matrix tile is loaded into registers before the current tile's MMA work;
the stage is overwritten only after the reader barrier. Turing has no TMA or
`cp.async`, so this register preload and two-CTA shared-memory fit are its
architecture-specific path. With `launch_bounds(512, 2)`, CUDA 13.0.48 reports
64 registers/thread, 26 KiB shared memory and zero spills.
An initial version keeping all partial transcripts in registers spilled 140 bytes
of stores and 96 bytes of loads, so that layout was replaced. A forced register cap
below 64 is not required to fit two blocks in Turing's 64K-register file.

Blackwell loads four 128x64 A/B stages (64 KiB total) with TMA and mbarriers,
then consumes them with `ldmatrix` and signed `m16n8k32` MMA. The transcript
fold remains lane-local and the kernel hashes and filters hits on device. The
SplitMix generator packs eight INT7 bytes with 64-bit masks and preserves the
reference byte sequence.

H100 production lives in `cuda/production_sm90.cu`, compiled for sm_90a. Each
K64 iteration issues two signed INT8 m64n128k32 WGMMA operations per warpgroup.
Two 128x64-byte tensor copies load A and B′ into each stage using TMA SW64.
The stage base is 1024-byte aligned; WGMMA descriptors use K-major SW64, a 512-byte
stride between eight-row groups, and a zero swizzle base offset. K offsets 0 and
32 select the two MMA steps. Tensor maps are encoded once per prepared job using
`cuTensorMapEncodeTiled`, then passed by value as grid-constant kernel arguments.
Replacing a job discards its maps before releasing the old device buffers.

A single elected thread initializes and issues each transaction barrier. An async
proxy fence publishes barrier initialization before TMA. Every consumer waits for
the matching token before issuing WGMMA. `wgmma.wait_group 0` plus a CTA barrier
ensures both warpgroups finished reading before a stage is refilled; a second CTA
barrier publishes the replacement token. Explicit accumulator constraints on the
fence/wait assembly prevent compiler motion of register accesses across those points.
Preloading two stages overlaps next-stage TMA with current-stage WGMMA. The elected
producer also issues a bounded L2 tensor prefetch two K tiles ahead while enqueueing
a stage, only when that tile remains within the tensor-map extent. K64 and final-stage
draining issue no out-of-range prefetch.
The scan computes work indices in registers, so there is no global work queue to load.

CUDA 13.0.48 reports 90 registers/thread, 43,040 bytes shared memory and zero spills
for H100. Shared transcripts avoid spilling the 64 accumulators plus two sets of
16 transcript words. SASS checks require `UTMALDG.2D` and `IGMMA.64x128x32.S8.S8`.
Compiler reports do not establish achieved occupancy or throughput; those require
hardware measurement. The L2 lookahead distance is an initial tuning choice, not a
measured optimum. The startup GPU-vs-CPU self-test remains mandatory.

References: [NVIDIA GPU capabilities](https://developer.nvidia.com/cuda/gpus),
[Turing tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html),
[Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html),
[PTX WGMMA shapes](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma).

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
kernels. H100 uses the grouped TMA/WGMMA production path. Benchmark scan throughput and live total throughput
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
  target filtering, raster offsets and queue bounds for T4 and Blackwell.
  This does not simulate CUDA scheduling, async-copy timing, or hardware faults.
- `python3 tools/verify_hopper_kernel.py`: executes H100 production control flow with
  CPU-emulated TMA SW64 copies and PTX WGMMA fragments. Checks descriptor fields,
  all 128 proofs at K=64, 128, 2176 and 4096, transcript wrap, inclusive targets,
  zero-capacity and overflowing queues, and a partial raster band. This does not
  emulate hardware async timing or validate the driver-generated tensor-map bytes.
- `bash tools/check_production_sass.sh`: compiles sm_75, sm_80, sm_86, sm_89, sm_90a,
  sm_120 TMA and sm_120 direct-global images; checks their Tensor Core/load paths and rejects
  register spills in production images. The opt-in experimental SM120 direct-global images are
  still checked for their MMA/load paths, while known spills are reported without failing CI.
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

The SM120 TMA scan remains the default. Set `PEARL_SM120_DIRECT_GLOBAL=unpadded` or `padded`
before startup to select the experimental direct-global `mma.sync` image (the accepted alias
`1` selects `unpadded`). The padded variant adds 1024 shared-memory bytes. Run the same acceptance
test and `--gpu-benchmark` on the same RTX PRO 6000 to compare both variants with the default.
The CPU source emulator checks both operand paths and padding layouts, but only hardware
measurements can choose the faster path.

### Checks for the architecture tuning update

- Prior CUDA 13.0.48 compilation and SASS checks passed for sm_75 and sm_90a before
  the new H100 L2 prefetch instruction was added, and for sm_120 before its new TMA
  production kernel was wired in. Neither latest change has been compiled or checked
  on hardware.
- The actual production-source CPU emulator passed all 128 transcripts at
  K=128, 2176 and 4096 for T4/Blackwell. H100 additionally passed K=64 and
  inclusive-target/zero-capacity cases through `verify_hopper_kernel.py`.
- The packed INT7 generator matched the previous byte arithmetic for 30,000 words.
- `cargo check --locked -p pearl-gpu` passed with the CUDA cubins embedded.
- Two host unit tests passed: invalid tensor layouts fail before driver access,
  and the K64/K128 startup fixtures produce valid CPU transcripts/digests.
- The production GPU acceptance test binary built with `--features gpu --no-run`.
- No physical GPU benchmark or live-pool acceptance was run.

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
