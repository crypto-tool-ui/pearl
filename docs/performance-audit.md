# Pearl performance audit

The requested H100 720 TH/s, RTX PRO 6000 Blackwell 420 TH/s and T4 20 TH/s
are targets, not measured results of this implementation. No software was
installed on the VPS. Rust, nvcc and an NVIDIA GPU are unavailable here;
compilation, CUDA correctness acceptance and hardware timing remain outstanding.
H100 production dispatch remains disabled pending WGMMA acceptance.

## Findings and changes

* Production accounting multiplied every scanned hash tile by full matrix M*N*K.
  It now counts tile height * tile width * effective K per tile. One production
  tile is 8*16*4096 = 524,288 MACs. A TH/s is 10^12 MAC/s, not twice that number.
  Existing CPU/GPU benchmark counters were corrected too. For M=N=131072 the
  old per-tile counter was inflated by 134,217,728. The user's 212 TH/s reading
  cannot be interpreted without knowing whether it came from this counter or
  from accepted pool shares.
* SM75 and SM120 now launch up to 128 independent tiles, preserving ordering and
  partial batches. Job cancellation is checked before and after a batch.
* Noise generation deduplicates the selected rows/columns across each batch;
  the official noise generator and commitment seeds are retained. Operands are
  still packed into the existing per-tile device layout.
* SM75/SM120 fold accumulators in registers using warp XOR reductions. Shared
  scratch holds one word per warp instead of the entire logical output matrix.
  Padding is excluded, and both block synchronization points are preserved.
* SM90 previously reset the running MMA accumulator every rank. It now initializes
  only at the first K step, retaining cumulative sums required by the V3 reference.

These changes do not implement a high-throughput GPU-resident GEMM miner. Per-batch
allocation, CPU noise, operand duplication, PCIe uploads, readback and synchronization
remain. A production T4 tile uploads (8+16)*4096 = 98,304 bytes for just 524,288 MACs.
Blackwell currently pads rows to 16, uploading 131,072 bytes. Batching amortizes
launches but cannot eliminate that transfer ceiling. The benchmark measures this
actual path, including preparation and transfers, rather than theoretical tensor throughput.

## Reference comparison

* `distributed-torch-flashatt-t4/CPPminer/docs/hashrate_calculation.md`: MAC units
  and wall-clock accounting. `src/cuda/cp_cutlass_gemm.cu` and
  `src/cuda/cutlass/cp_cutlass_gemm_types.h`: larger CUTLASS GEMM tiles, fused
  transcript/hit processing and architecture-specific MMA.
* `spark-pearl-miner/crates/spm-gpu`: GPU preparation, resident operands, large
  macrotiles and pipelined launches. Its GB10 measurements do not establish a
  result for RTX PRO 6000, H100 or T4.
* `IAminer`: useful tuning and GPU-resident preparation concepts; its Intel/SYCL
  measurements are not NVIDIA benchmarks. No GPL implementation was copied.
* `pearl/miner/pearl-gemm`: official Hopper GEMM pipeline and configuration
  heuristics. The CPU implementation remains the consensus oracle.

NVIDIA's [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
prioritizes reducing host transfers, exploiting concurrency and using TMA for
asynchronous data movement. Architecture limits must be checked separately against
[Turing](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html) and
[Blackwell](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html);
SM120 is not interchangeable with SM100.

The next substantial implementation step is resident noised matrices with a
macrotile GEMM computing many hash tiles per CTA, fused bound checking, a compact
hit queue, and reused asynchronous buffers. Preserve cumulative rank-fold ordering
and CPU proof validation. This requires correctness and profiling runs on each
architecture before selecting tile sizes, stages and launch geometry.

## Validation and measurement

Local checks: `git diff --check` passed. A Python CPU model checked 70 combinations
of logical/padded shapes for accumulator ownership, masking and warp XOR equality.
This does not validate CUDA instruction behavior or establish a speedup.

Added Rust tests compare batched packing with independently generated operands,
check zero padding and invalid inputs. GPU acceptance tests compare full and partial
batches, duplicates and reordered offsets against the CPU oracle. They have not run
on this VPS.

On an existing machine with Rust, CUDA and the target GPU already configured:

```sh
cargo test -p pearl-gpu --lib
PEARL_REQUIRE_GPU_ACCEPTANCE=1 cargo test -p pearl-gpu --features gpu sm75 -- --nocapture
PEARL_REQUIRE_GPU_ACCEPTANCE=1 cargo test -p pearl-gpu --features gpu sm120 -- --nocapture
PEARL_REQUIRE_GPU_ACCEPTANCE=1 cargo test -p pearl-gpu --features gpu sm90_wgmma -- --nocapture
cargo run --release -p pearl-gpu --example throughput -- sm75 0
cargo run --release -p pearl-gpu --example throughput -- sm120 0
cargo run --release -p pearl-gpu --example throughput -- sm90 0
```

Run the matching architecture commands on each GPU. The example first checks results
against the CPU oracle, warms up, then compares single-tile and batch wall-clock rates
for at least ten seconds each. It uses production hash-tile geometry and a small
128x128 committed matrix; it excludes full production matrix preparation and proof
construction, so it is not an end-to-end pool hashrate benchmark. SM90 remains a
single-tile experimental path. Record GPU model, clocks, power limit, CUDA/driver
versions and accepted/rejected pool shares alongside any performance claim.
