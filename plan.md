# Pearl GPU miner plan

The implementation is staged exactly at the requested milestones. A milestone is complete only when
its acceptance command passes. The next milestone does not start after a documentation-only result or
a CPU approximation of a GPU requirement.

## Repository and source policy

The new miner will live at the workspace root with the requested crates and `cuda/` layout. The
official Pearl checkout is `/home/user/pearl/pearl`, pinned for M1 to
`3fe226761a139a9652b8f28a6464a4bbc25986c8`; it is read-only protocol input. The existing
`spark-pearl-miner` checkout is an architecture reference only. No dev-fee code or wallet will be
copied.

Consensus types and verification are wrapped from the official reference in `pearl-ref`; consensus
parameters are not re-created from memory. The GPU host API will expose a versioned `GemmKernel`
trait with separate architecture and precision implementations. Network/job code will select a
supported version and pause cleanly on an unsupported hard fork.

## Milestones

### M1 — CPU reference and protocol lock

Create `crates/pearl-ref`, wrap the official V3 reference, add deterministic matrix/noise/tile/
transcript/digest helpers, and generate checked-in golden vectors under `tests/golden`. Add tests for
valid and invalid candidates and require official proof verification before a candidate is accepted.
Write the protocol citations in `docs/protocol.md`.

Acceptance: `cargo test -p pearl-ref`.

### M2 — Naive CUDA correctness oracle

Create `crates/pearl-gpu`, its cudarc loader and device-independent `build.rs`, plus `cuda/naive.cu`.
The naive kernel is an oracle only; it is never the production hot loop. GPU tests are gated by the
`gpu` feature. The cubin build path must work without a driver or device and must warn and skip when
`nvcc` is absent.

Acceptance with hardware: `cargo test -p pearl-gpu --features gpu` equals M1 golden vectors. Without
hardware: both requested cubin compilation commands succeed.

### M3 — sm_120 INT8 tensor-core kernel

Implement `cuda/sm120_mma.cu` with `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`, `cp.async`
multi-stage shared-memory movement, `ldmatrix` layout, and fused transcript/hash epilogue. Keep
the naive oracle and compare at least 10,000 randomized tiles. Verify SASS contains IMMA.

Acceptance: run `tools/run_sm120_acceptance.sh` on an RTX PRO 6000 Blackwell; it must complete the
golden comparison, 10,000 independently committed randomized INT7 input comparisons alternating
the official 4×8/rank32 shape and a full 16×16/rank128 pipeline shape against the naive oracle, and
the tiles/s plus effective TH/s benchmark. Run `tools/check_sm120_sass.sh` on the no-GPU CI host; it
must find IMMA, LDSM, and LDGSTS; the script also enforces the source instruction contract and
rejects `tcgen05`/TMA. The acceptance wrapper selects the detected sm_120a ordinal and fails if
tests would otherwise skip for missing cubins or driver setup. No TMA is used on sm_120a.

### M4 — sm_90 WGMMA/TMA kernel

The initial implementation is present in `cuda/sm90_wgmma.cu` with a host loader,
compute-capability dispatch, a dedicated TMA producer warp, a two-stage transaction-barrier
pipeline, fused transcript/digest, SASS gate, and H100 acceptance wrapper. It uses the same V3
INT8 trait metadata as M3 and leaves the sm_120 source untouched. M4 remains open pending the
H100 hardware acceptance test.

Acceptance: M3-equivalent randomized and benchmark checks plus WGMMA/QGMMA SASS.

### M5 — GPU-resident job setup and multi-GPU

The initial implementation is present in `cuda/tensor_hash.cu` and `crates/pearl-gpu/src/m5.rs`.
It computes keyed BLAKE3 chunk/parent levels on the GPU, derives the official job commitment, keeps
two job slots per device, supports one isolated worker/context per device, cancellation, restart,
autotune cache persistence, and explicit power/clock options.

Acceptance: the GPU self-test and benchmark remain bit-exact while exercising double buffering and
all selected devices.

### M6 — pool/gateway transport

The initial implementation is present in `crates/pearl-net`: `PoolProtocol` and the captured Kryptex
v1/v2 dialect, TLS Stratum session, up to three endpoint failover, reconnect/keepalive/stale-job
handling, share statistics, and official `getblocktemplate`/`submitblock` gateway RPC. `VerifiedShare`
is the only submit path and invokes the CPU reference verifier before encoding the proof.
No real-pool traffic is allowed until this milestone and an explicit testnet/local-gateway run are
complete.

Acceptance: local gateway or SimNet/testnet E2E with accepted verified shares, failover, stale-job
rejection, and zero locally invalid submissions.

### M7 — Rust/Node encrypted channel

The legitimate module is now created in `tools/proxy/crypto/index.js` because the user requested
that it be built from scratch. `pearl-net::crypto` implements the same X25519, HKDF-SHA256, and
AES-256-GCM contract; `docs/e2e-channel.md` records the exact handshake, framing, AAD, counters,
pinning, and rekey behavior. `tools/proxy/server.js` now terminates that channel and relays
unchanged NDJSON bytes over a configured Kryptex TCP/TLS connection. Fixed cross-language material
is in `tests/crypto-vectors/v1.json`; the Node suite includes a local fake-pool relay test.

Acceptance: Rust and Node vectors agree byte-for-byte and the integration test passes all failure
cases. No wire parameter is added without an explicit source in the supplied module.

### M8 — operations and release checks

The initial operations surface is present in `crates/pearl-miner`, `systemd/`, `Dockerfile`,
`tools/proxy/Dockerfile`, `README.md`, and `docs/m8-acceptance.md`: config/CLI parsing, offline
benchmark, GPU self-test, disposable GPU worker process, structured logs, an optional Prometheus
text endpoint, clock/power options, and V3-only `update required` behavior. CI still runs CPU-only
Rust and Node tests plus both cubin and SASS checks; hardware checks remain explicitly marked as
needing hardware.

Acceptance: no-GPU CI sequence passes, self-test refuses mismatches, and packaging starts only after
the local GPU self-test succeeds. Kryptex job admission and the correctness-first GPU tile loop are
now wired through `crates/pearl-miner/src/job.rs`; the worker materializes the captured fixed V3
work convention, scans tiles, CPU-verifies candidates, and submits only `VerifiedShare` values.

## Required CI sequence

The current workflow is `.github/workflows/ci.yml`. It runs the CPU-only Rust checks, the Node
proxy job when `tools/proxy/package.json` exists, and both CUDA 13.0 no-GPU cubin/SASS gates.
The operator-facing M3, M4, M5, and M6 procedures are documented in `docs/m3-acceptance.md`,
`docs/m4-acceptance.md`, `docs/m5-acceptance.md`, and `docs/m6-acceptance.md`.

```text
cargo build --release
cargo test
node.js proxy tests
nvcc -cubin -arch=sm_120a -o /tmp/k120.cubin cuda/sm120_mma.cu
cuobjdump -sass /tmp/k120.cubin
```

The current SASS gate fails if sm_120a has no IMMA. After M4, the workflow must also fail if sm_90a
has no WGMMA/QGMMA. M5 adds tensor-hash cubins for sm_90a and sm_120a. `cudarc` is configured with default features disabled and `std`,
`dynamic-loading`, and `cuda-13000`; no build script may require a CUDA device or an NVIDIA driver.

## Reporting and commit rule

Every milestone report will include the build command, test command, expected output, actual output,
and a `not verified (needs hardware)` list. After its acceptance test passes, commit with exactly
`M<n>: <summary>`. The workspace root Git repository was initialized at `/home/user/pearl`; the
official Pearl checkout and the existing `spark-pearl-miner` checkout remain separate and ignored.

## Current blockers

M3 implementation and host/build checks are present, but its acceptance remains open until an
RTX PRO 6000 Blackwell (compute capability 12.0) runs the M3 wrapper with the 10,000 randomized
comparisons and benchmark passing. This environment has the CUDA 13 toolkit but no NVIDIA driver
or device, so that runtime evidence cannot be collected here. M4 source/build checks are present,
but H100 bit-exact acceptance is also `needs hardware`. M5 source/build orchestration is present,
but GPU root equivalence, overlap, all-device execution, and benchmark remain `needs hardware`.
M6 source is present and the worker now consumes Kryptex jobs, but controlled gateway/testnet E2E,
TLS connectivity, failover, large-work-unit memory/throughput, and accepted verified-share evidence
remain `needs hardware`/external-service.

M1 is not blocked by the missing proxy module or pool choice. M6 is targeted at Kryptex, as
specified for this project; its authoritative job/share wire details will be pinned from public
documentation or a local gateway before any pool traffic is enabled. M7 source implementation is
present, but its Rust/Node test suites and integration acceptance have not been run because tests
were explicitly skipped. The earlier script identified as malware is not a source file, dependency,
or reference for this repository. M8 source packaging is present, but its
no-GPU build, systemd/container startup, self-test, metrics, and clean pause acceptance have not
been run because tests were explicitly skipped. V4 is not an M1 blocker because official
`master` has no V4 activation and INT8/V3 remains the current mainnet rule; a future job advertising
V4 will pause instead of being submitted.

The untracked `B200-Engine8-1440PH` package was inspected read-only. Its Kryptex V2 details are
covered by the Rust protocol regression test: V2 authorization omits the password field, submission
carries `hs`, and the compressed proof removes the trailing `moe=None` bincode byte. Its SM100a
TMA/TMEM/`tcgen05` kernel and B200 tuning are incompatible with the required `sm_120a` `mma.sync`
path and were not imported.
