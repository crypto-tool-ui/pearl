# CUDA kernel contract

M2 provides `pearl_naive_tile` in `cuda/naive.cu` as a correctness oracle. It deliberately uses one
CUDA thread per tile, so it is not a production mining loop and contains no tensor-core path.

The host passes the selected V3 tile after the official CPU reference has constructed the noised
operands. The kernel receives row-major `A'` with shape `h × k` and row-major `B'^T` with shape
`w × k`, both signed 8-bit values. It maintains cumulative signed 32-bit cell accumulators. After
each rank-sized K slice it folds every accumulator as a `uint32_t` XOR and updates
`transcript[(slice - 1) % 16]` with `rotl32(word, 13) ^ fold`. The host hashes the returned 16-word
transcript with the official keyed BLAKE3 jackpot function; the production tensor-core path is
described below.

The CUDA host crate exposes this through `GemmKernel`, whose metadata includes architecture,
precision, and certificate version. `NaiveCudaKernel` selects an oracle cubin for tests. Production
callers use `select_production_kernel`, which selects the architecture-specific tensor-core
implementation by runtime compute capability and never falls back to the naive CUDA-core kernel.
The sm_120 and sm_90 loaders require 32-wide warps and check device block-thread and opt-in
dynamic shared-memory limits before launching.

`crates/pearl-gpu/build.rs` compiles the cubins when `nvcc` is present, looking first at `NVCC`, then
`CUDA_PATH`/`CUDA_HOME`/`CUDA_ROOT`, and finally `PATH`. It emits a Cargo warning and skips them
when nvcc is absent. The direct no-driver build checks are:

```text
nvcc -cubin -arch=sm_90a  -o /tmp/k90.cubin  cuda/naive.cu
nvcc -cubin -arch=sm_120a -o /tmp/k120.cubin cuda/naive.cu
```

The M2 GPU test is enabled with `--features gpu`. It compares the device transcript and host digest
with the CPU reference golden tile. A no-toolkit build reports the missing cubin and skips the
hardware test; it does not create a CUDA context.

## M3 sm_120a INT8 path

`cuda/sm120_mma.cu` uses the requested `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
instruction. The host pads the selected logical tile to 16 rows by 8 columns with zero operands;
the logical dimensions still control the transcript fold. Each warp owns one padded subtile. The
kernel uses two shared-memory stages, `cp.async` global-to-shared copies, `ldmatrix` loads, and a
register-resident int32 accumulator. It writes only the logical accumulators to shared memory for
the fused XOR/rotate transcript update and never writes a full GEMM tile to global memory.
The source has a compile-time `__CUDA_ARCH__ == 1200` guard so an sm_90 or generic build cannot
silently produce a mismatched cubin. One launch is capped at 32 warps; the host checks this before
launch and the kernel repeats the guard for direct callers.

The host passes both the physical row stride and the official rank-truncated effective K length,
so a non-rank-aligned common dimension follows the reference loop's floor behavior. The host still
constructs the official V3 noised operands. After the final transcript barrier, the kernel computes
the official one-block keyed BLAKE3 jackpot digest and returns both transcript and digest; the CPU
reference remains the acceptance oracle. The host recomputes the keyed digest from the returned
transcript and fails the tile if the fused device digest differs. Runtime dispatch accepts this
kernel only on compute capability 12.0 and keeps the V3/int7 precision metadata separate from future
FP8 implementations.

The device epilogue is constrained to the official single 64-byte keyed root block: its source has
compile-time assertions for the 27-bit flag mask and 64-byte block length.

Before launch, the host checks both the device-wide thread limit and the one-dimensional block-X
limit because the kernel is launched as a 1D block. Module bind/unload failures are logged during
cleanup and never panic from a worker shutdown path.

Shared memory follows the [CUDA Programming Guide async-copy swizzle](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
at 16-byte granularity: `physical_chunk = (((row + base_offset) % 8) ^ logical_chunk)`, with
`base_offset = (shared_address(stage_base) / 128) % 8`. `cp.async` writes and `ldmatrix` reads use the same address transform,
so this is a storage-layout change only. The kernel does not use TMA on sm_120a.

For the INT8 fragments, A is viewed as a 16×16 matrix of 16-bit carriers, with each carrier holding
two adjacent K bytes. The raw 8×32-byte Bᵀ stage is viewed as an 8×16 carrier matrix and loaded
directly with `ldmatrix.x2.trans`; the host and kernel do not reinterpret adjacent N columns as a
packed K pair. The accumulator store follows NVIDIA's PTX `mma.m16n8k32` fragment contract:
`groupID = lane >> 2`, `threadID = lane & 3`, `row = groupID` for `i < 2` and `groupID + 8`
otherwise, and `col = threadID * 2 + (i & 1)`. The `ldmatrix.x4` A pointers address the four
8×8 carrier matrices in the MMA register order top-left, bottom-left, top-right, bottom-right
through lanes 0–7, 8–15, 16–23, and 24–31; `ldmatrix.x2.trans` B pointers address the two 8×8
carrier matrices through lanes 0–7 and 8–15, providing the K[0:16] and K[16:32] halves in
register order. In the raw signed-8-bit view, PTX defines A's four registers as `row = groupID`
for register elements 0–3 and 8–11, otherwise `groupID + 8`, with columns
`threadID * 4 + (element & 3)` and an extra 16 for elements 8–15; B's two registers use
`row = threadID * 4 + (element & 3)` and an extra 16 for the second register, with
`col = groupID`. This is the register and address contract described in the [NVIDIA PTX ISA
matrix-fragment documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-multiply-accumulate-instructions-mma),
including its `ldmatrix` lane/address table, and remains independent of the shared-memory swizzle.

The M3 cubin and SASS checks are:

```text
nvcc -cubin -arch=sm_120a -o /tmp/k120-mma.cubin cuda/sm120_mma.cu
cuobjdump -sass /tmp/k120-mma.cubin | grep IMMA
cuobjdump -sass /tmp/k120-mma.cubin | grep LDSM
```

The reproducible repository gate is `tools/check_sm120_sass.sh`; it resolves `NVCC`/`CUOBJDUMP`,
then `CUDA_PATH`/`CUDA_HOME`/`CUDA_ROOT`, and finally `PATH`. It checks the source contract for the
requested MMA, `cp.async.ca.shared.global`, commit/wait groups, `ldmatrix` forms, the 32-warp and
dynamic-shared-memory structure, and the single-block BLAKE3 assertions. It rejects `tcgen05` or
TMA, including tensor `cp.async.bulk` forms, then checks `IMMA`, `LDSM`, and `LDGSTS` in SASS. It
does not require a CUDA driver or GPU.

Run the hardware acceptance through `tools/run_sm120_acceptance.sh`. It requires an NVIDIA driver
and selects the ordinal of a detected compute-capability-12.0 GPU before invoking the feature-gated
randomized comparison and benchmark, preventing a no-driver skip or a wrong-device run from being
mistaken for acceptance. The wrapper also sets `PEARL_REQUIRE_GPU_ACCEPTANCE=1`, so missing cubins
or a runtime initialization error fails the gate instead of being reported as a skipped test.

The feature-gated GPU tests run the sm_120 kernel against the CPU reference golden tile, compare
10,000 independently committed randomized INT7 input tiles (alternating the official 4×8/rank32
shape and a full 16×16/rank128 pipeline shape) against the naive oracle, and print a benchmark such as
`sm_120 benchmark: <tiles>/s, <effective> effective TH/s`. These checks require a
compute-capability-12.0 device and driver; a host with no NVIDIA device only verifies compilation
and test setup.

## M4 sm_90a WGMMA path

`cuda/sm90_wgmma.cu` is the Hopper implementation for the same V3 INT8 contract. It uses the
official `wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8` operation with four consumer warps per
64x8 output tile. A dedicated producer warp issues the TMA work for all consumer warpgroups;
K32 panels are loaded into two shared-memory stages with TMA bulk copies. Each 16-byte copy
contributes to an `mbarrier` transaction, and the consumer waits on the returned barrier token
before issuing WGMMA. The WGMMA fence, commit, and wait operations establish the async-proxy
ordering required by the PTX ISA.

The source uses the PTX K-major canonical layout `((8,m),(T,2k)):((1T,SBO),(1,LBO))` for signed
8-bit data (`T=16`). Its software packer writes the two 128-byte K chunks and the descriptor uses
`LBO=128` and `SBO=256` for a K32 panel. Each warpgroup writes only its accumulator cells to
shared memory for the transcript fold; no complete GEMM tile is written to global memory.

The loader has a `WgmmaSm90CudaKernel` implementation for compute capability 9.0 and keeps it
behind the M4 acceptance gate; compute capability 12.0 selects `MmaSm120CudaKernel`. Until
`tools/run_sm90_acceptance.sh` passes, sm_90 production dispatch returns an update-required error
instead of selecting an unaccepted path. The no-GPU static gate is
`tools/check_sm90_sass.sh` and requires INT8 IGMMA and bulk-copy UBLKCP plus the async transfer instructions in SASS.

## M5 GPU-resident tensor hash and job ring

`cuda/tensor_hash.cu` implements the official keyed BLAKE3 tree used by the V3 matrix commitment:
one thread hashes each padded 1024-byte chunk to a chaining value, then one thread combines each
pair at every parent layer. Odd nodes are carried unchanged exactly as in
`pearl-blake3::MerkleTree`; the final two children use the root flag. The host keeps intermediate
levels device-resident and copies back only the 32-byte root. It compiles for both accepted tensor
core architectures and does not participate in the GEMM precision dispatch.

`m5::GpuJobSetup` derives `job_key`, GPU matrix roots, official bound roots, and both noise seeds.
`m5::MultiGpuWorkers` gives each selected device its own context, worker, and two job-slot streams.
The slot state machine lets the next job reach `Ready` while the current slot remains `Running`; a
CUDA error can synchronize and replace the complete worker. `AutoTuneCache` uses an atomic temp-file
rename and keys results by device identity, while `PowerClockOptions` passes explicit limits to
`nvidia-smi` without a shell.

## Turing sm_75 path

`cuda/sm75_mma.cu` and `crates/pearl-gpu/src/sm75.rs` implement the T4 path.
One warp computes an 8×8 output tile with signed INT8 m8n8k16 MMA. Each lane
packs four contiguous K elements of A and B-transpose into its operand registers,
using the [PTX fragment mapping](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m8n8k16).
Operands are zero-padded to 8×8; only logical cells enter the transcript fold.
Accumulators remain cumulative across rank boundaries, and the kernel hashes the
transcript with the official keyed BLAKE3 single-block root parameters.
Only accumulator scratch uses dynamic shared memory; no cp.async/TMA is used.
The adapter checks shape alignment, block limits and shared-memory limits.

The cubin is embedded with the other architectures and capability 7.5 selects it
in production dispatch. Run `bash tools/check_sm75_sass.sh` for compilation and
IMMA inspection, then `bash tools/run_sm75_acceptance.sh` on T4 for correctness
and benchmark acceptance. No hardware acceptance result is recorded yet.

### T4 resident job preparation

The production T4 path calls `prepare_job` once per job. CUDA computes the keyed
uniform noise factors and applies official CPU-generated sparse permutations to
the raw matrices in place. Full noised matrices remain on the device. One batch
(up to 1024 CTAs) gathers rows/columns by index; accumulator/digest output and index
buffers are reused. The previous padded-input path remains available to tests.
Job changes drop the previous resident allocation before allocating the next.
CPU commitments and final share verification remain unchanged.

`tools/verify_sm75_noise.py` verifies scalar noise arithmetic against the official
reference on CPU, while `sm75_resident_noise_and_mining_match_cpu_after_job_change`
requires a T4 and checks full/partial batches and changed seed material. No GPU
utilization improvement has been measured in the development environment.
