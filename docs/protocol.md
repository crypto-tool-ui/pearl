# Pearl proof-of-useful-work protocol

This document records the protocol facts used by the miner. The source of truth is the local clone of
`https://github.com/pearl-research-labs/pearl` at official `master` commit
`3fe226761a139a9652b8f28a6464a4bbc25986c8` (2026-09-24). The proposed FP8 branch is recorded separately
where it affects dispatch; it is not used by the V3 reference.

## Current consensus target

The checked-out `master` reference has exactly three certificate versions: V1 dense, V2 MoE/dense, and
V3 with salted noise-seed derivation. `CertificateVersion` maps V1 and V2 to legacy seed derivation and
V3 to salted derivation in `pearl/zk-pow/src/ffi/plain_proof.rs:158-188`.

The only `MMAType` in this revision is `Int7xInt7ToInt32 = 0`,
`pearl/zk-pow/src/api/proof.rs:18-25`. Therefore M1 targets the dense V3/int7 path. INT8 is the
hardware representation of the exact int7 operands plus additive noise; it does not change the
consensus type.

The current mainnet parameters activate the earlier MoE, dense-only, rank-penalty, and salted-seed
rules at heights 71935, 91630, 96251, and 99000 respectively, with no FP8 field in this official
`master` source (`pearl/node/chaincfg/params.go:340-395`). The strict version selector returns V3 at
and after the salted-seed height, V2 at and after the MoE height, and V1 otherwise
(`pearl/node/chaincfg/params.go:320-337`).

The implementation will accept a job only when its advertised certificate version is known and
supported. A V4 or unknown version pauses mining with `update required`; it never submits a V3/INT8
share for that job. Every candidate share is checked by the official `verify_plain_proof` before it can
reach a pool or gateway (`pearl/zk-pow/src/api/verify.rs:96-133`).

## Job encoding and commitment

The incomplete header is 76 bytes: version, reversed previous-block hash, reversed Merkle root,
timestamp, and compact `nbits`, all other integer fields little endian
(`pearl/zk-pow/src/api/proof.rs:5-16,503-527`). The mining configuration is 52 bytes:

```text
common_dim u32 LE | rank u16 LE | mma_type u16 LE |
rows_pattern 6 bytes | cols_pattern 6 bytes |
MoE trailer: e u16 LE | top_k u16 LE | 28 zero bytes
```

This layout and its strict decoder are in `pearl/zk-pow/src/api/proof_utils.rs:430-500`. A dense
configuration has `e = 0` and `top_k = 0`. The job key is unkeyed BLAKE3 of the 76-byte serialized
header followed by the 52-byte serialized configuration
(`pearl/zk-pow/src/api/proof_utils.rs:348-353`).

For a dense V3 job, A is an `m × k` row-major int7 matrix and B is represented as `Bᵀ`, an `n × k`
row-major matrix. The reference miner generates both matrices in the inclusive range `[-64, 64]`,
transposes B, pads each byte buffer to a BLAKE3 1024-byte chunk boundary, and computes keyed roots
with the job key (`pearl/zk-pow/src/ffi/mine.rs:33-57`,
`pearl/zk-pow/src/ffi/mine.rs:381-390`). The padding operation is zero extension to a multiple of
1024 bytes (`pearl/pearl-blake3/src/merkle.rs:14-26`), and the Merkle tree uses keyed BLAKE3
(`pearl/pearl-blake3/src/merkle.rs:41-80`).

V3 first binds the A root to `m` and the B root to `n`:

```text
bound_a = BLAKE3(key=SEED_SALT_A, root_a || m as u32 LE || 28 zero bytes)
bound_b = BLAKE3(key=SEED_SALT_B, root_b || n as u32 LE || 28 zero bytes)
b_noise_seed = BLAKE3(job_key || bound_b)
a_noise_seed = BLAKE3(b_noise_seed || bound_a)
```

The exact salt bytes, 64-byte message construction, and the legacy-versus-salted selector are in
`pearl/zk-pow/src/api/seed.rs:7-59`; the seed chain is in
`pearl/zk-pow/src/api/proof_utils.rs:355-371`.

## Noise and noised GEMM

The current V3 noise rank must satisfy the official sanity checks: it is a power of two from 32
through 1024, divisible by the 16-element circuit tile; `k` is divisible by 64, at least 1024, at
most 2^16, at least `16*r`, and at most `4*r^2`
(`pearl/zk-pow/src/api/sanity_checks.rs:15-53`). The standard miner configuration uses the rank
that the received job specifies; it does not assume a rank or shape that the job did not commit.

Noise uses two zero-padded labels, `A_tensor` and `B_tensor`
(`pearl/zk-pow/src/circuit/pearl_noise.rs:19-30`). A noise hash is keyed BLAKE3 over a 64-byte
message: one 32-bit value `(index + 1)` is written at the selected four-byte slot and the label is
written in the final 32 bytes (`pearl/zk-pow/src/circuit/pearl_noise.rs:45-57`).

Uniform factors use every byte masked to seven bits and translated by 64, giving `[-32, 31]`
(`pearl/zk-pow/src/circuit/pearl_noise.rs:59-80`). Permutation factors consume little-endian
32-bit words from the slot-1 hash. For each word `x`,

```text
p = x & (r - 1)
q = p ^ (1 + high32((r - 1) * x))
```

The exact implementation, including hash grouping and pair generation, is
`pearl/zk-pow/src/circuit/pearl_noise.rs:82-115`. The A and B noise products are sparse permutation
matrix-vector products; the selected rows/columns are produced by
`compute_noise_for_indices` (`pearl/zk-pow/src/circuit/pearl_noise.rs:118-160`).

For each selected output cell, the reference adds the noise to the clean int7 value in `i32` and
then performs the dot product in `i32`. The official mining loop shows the noised operands and the
accumulation order (`pearl/zk-pow/src/ffi/mine.rs:59-109`); the verifier constructs the same noised
tiles and multiplies each 16-element slice (`pearl/zk-pow/src/circuit/chip/matmul/trace.rs:41-98`).
The CUDA path must preserve these values and the ascending K order exactly.

## Tile, transcript, digest, and difficulty

The committed row and column patterns define one jackpot tile. The official reference iterates the
pattern partitions in ascending order, maintains cumulative `i32` tile accumulators, and after each
rank-sized K slice XORs every accumulator as a `u32`. It then updates one of 16 transcript words:

```text
fold = XOR over the cumulative tile accumulators, interpreted as u32
slot = (slice_number - 1) mod 16
transcript[slot] = rotl32(transcript[slot], 13) XOR fold
```

This is the consensus loop in `pearl/zk-pow/src/circuit/chip/jackpot/helper.rs:9-34`, with the
reference miner's equivalent explicit loop in `pearl/zk-pow/src/ffi/mine.rs:86-109`. The 16 words
are serialized as 64 little-endian bytes and hashed by keyed BLAKE3 under the A noise seed
(`pearl/zk-pow/src/api/proof_utils.rs:1501-1505`). The digest is interpreted as a little-endian
256-bit integer for comparison (`pearl/zk-pow/src/api/sanity_checks.rs:146-156`).

The target is the compact `nbits` target multiplied by
`rows_pattern.size() * cols_pattern.size() * dot_product_length()`; the official implementation
does this in `pearl/zk-pow/src/api/sanity_checks.rs:182-230`. The rank penalty, when the node's
chain rules activate it, uses rank 128 as its base and scales by
`tile_size * (dot_product_length / rank) * 128`
(`pearl/zk-pow/src/api/sanity_checks.rs:159-196`). The miner will use the pool share target as
the `nbits_override` input to the same official verification path.

## Certificate V4 / FP8 status

The checked-out official `master` commit has no V4 type, no FP8 `MMAType`, and no `Fp8ForkHeight`.
The local remote ref `origin/fp8-scheme` is a proposal at commit
`284b147bf56bcc7e38c9f0bb87a56d61e9b7519d`; it is not the current `master` protocol.

That proposal defines `Fp8ForkHeight`; when nonzero and `height >= Fp8ForkHeight`, the required
certificate is V4, and V4 takes precedence over V3
(`pearl@origin/fp8-scheme/node/chaincfg/params.go:283-354`). Its mainnet parameter block still
omits `Fp8ForkHeight`, so the zero value disables the fork there
(`pearl@origin/fp8-scheme/node/chaincfg/params.go:357-395`). Its local regtest and SimNet examples
set the fork to height 1 (`pearl@origin/fp8-scheme/node/chaincfg/params.go:478-496,780-796`).

The proposal's V4 path is a materially different algorithm, not an INT8 encoding choice. It has
device-specific FP8 quantization and devices H100/B200 in
`pearl@origin/fp8-scheme/zk-pow/src/api/fp8/public_params.rs:120-225`, separate domain-separated
keys/seeds and a V4 jackpot key in
`pearl@origin/fp8-scheme/zk-pow/src/api/fp8/transcript.rs:57-187`, and FP8 E4M3/BF16 arithmetic in
`pearl@origin/fp8-scheme/zk-pow/src/api/fp8/dtype.rs:4-87`. The requested miner therefore keeps
`GemmKernel` behind a precision/version trait. The initial implementation is V3/int7/INT8 only;
the network layer will pause with `update required` when it observes V4 or an unknown version.
No V4 kernel or V4 protocol parameter will be invented from the proposal branch.

## Share safety invariant

The miner must construct a proof from the exact winning tile, call the official eligibility check,
then call `verify_plain_proof` with the job's certificate-derived seed mode and the pool share target
override. Only an `Ok(())` result may be submitted. A verification failure is a local computation
fault and must discard the candidate; it is never converted into a share submission.

## Kryptex job materialization

Kryptex `mining.notify` supplies the 76-byte header, target, job ID, height, and certificate
version; it does not carry A/B matrices or a serialized `MiningConfiguration`. The captured Kryptex
dialect therefore uses the fixed dense work convention recorded in the read-only reference at
`spark-pearl-miner/crates/spm-pow/src/lib.rs:13-105` and
`spark-pearl-miner/crates/spm-work/src/lib.rs:1-93`: `m=n=131072`, `k=4096`, `rank=128`, row
offsets `[0,8,16,24,32,40,48,56]`, column offsets
`[0,1,8,9,16,17,24,25,32,33,40,41,48,49,56,57]`, and dense V3/no MoE. The per-job fill seed is
`BLAKE3(job_key || "spm/fill/v1")`, reduced to the first eight little-endian bytes for the
SplitMix64 A/B fill streams, with domains `spm-a-01` and `spm-b-01`; the exact reference generator
is at `spark-pearl-miner/crates/spm-cpuref/src/problem.rs:1-105`.

The Rust implementation is `pearl_ref::Problem::from_kryptex_header`. It rejects malformed headers,
unknown certificate versions, and non-dense jobs before allocating work. This materialization is a
pool work convention; the consensus arithmetic, seeds, proof construction, and verifier still come
from the pinned official Pearl checkout cited above. The current correctness-first implementation
materializes both 1-GiB-scale matrices before the GPU loop, so production memory and throughput
remain acceptance items.

The pool-facing prefilter is `pearl_ref::share_bound`: it decodes the pool target through compact
`nbits`, takes the smaller of the original and compact-decoded targets, and applies the official
rank penalty. `pearl_net::VerifiedShare::from_verified_tile` still runs the complete official
verifier after this prefilter; the prefilter is never a submit authorization by itself.

The Kryptex socket runs in a network actor while the disposable GPU worker scans. Each new
`mining.notify` advances a generation and raises the cancellation flag before publishing the job.
The worker checks both values between tiles and before emitting a candidate, while the actor checks
the current job ID before writing `mining.submit`. This prevents a stale candidate from reaching
the pool after a job update.

## B200 Engine8 cross-check

The untracked `B200-Engine8-1440PH` package was inspected as a read-only implementation reference.
Its pool adapter confirms the V3 Kryptex V2 submit shape: object authorization with `wallet`,
`agent`, and `type: "v2"`; then `mining.submit.params` contains integer `hs`, `job_id`, and
`plain_proof`. The adapter removes the final bincode `moe=None` byte before gzip level 1/base64
encoding. `pearl-net::KryptexProtocol` implements this wire transformation for V2, while V1 remains
available as the captured subscribe/authorize/plain-proof mode. This is a transport compatibility
cross-check, not a change to Pearl consensus serialization.

The package's CUDA core targets `compute_100a`/`sm_100a` and uses SM100 MMA, TMA, TMEM, and
`tcgen05`; those instructions and its tuned `M/N/K/group` values are not used by the `sm_120a`
path. Its reported 1.440 PH/s evidence is explicitly local/offline throughput and excludes CPU
proof and network time. No prebuilt B200 binary, historical wallet, or B200 kernel is part of this
repository's mining path.

## Open protocol questions

These do not block M1, but they block the milestones that consume them:

1. The M7 crypto module and wire contract are now present in `tools/proxy/crypto/index.js` and
   `docs/e2e-channel.md`; its Rust/Node acceptance has not yet been run.
2. M6 Kryptex transport is now pinned to the captured dialect in
   `spark-pearl-miner/docs/protocol/kryptex.md:1-20`, using the public PRL endpoint guidance at
   `https://pool.kryptex.com/prl`: TLS port 8048, Stratum v1 subscribe/authorize arrays, object
   `mining.notify.params`, the pool-supplied target verbatim, and `plain_proof` base64(bincode)
   submits. The optional V2 mode uses object authorization, reports `hs`, and sends the B200-style
   gzip level 1 proof wire format; it is explicit in config rather than auto-guessed. M6 code
   refuses missing/unknown certificate versions and stale job IDs before any submit.
3. The target is fixed to the official V3/INT8 path for this miner. A future job advertising V4 or
   another unknown certificate version must pause with `update required`; it must not be interpreted
   as an INT8 job.
