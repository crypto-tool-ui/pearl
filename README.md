# Pearl V3 GPU miner

This repository targets Pearl certificate V3 with the official INT7 consensus arithmetic represented
by INT8 tensor-core operands. Runtime dispatch selects a resident, grouped `mma.sync`
mining kernel on T4 (sm_75), H100/H200 (sm_90a), or RTX PRO 6000 Blackwell (sm_120a).
Every device must pass the production-kernel CPU comparison at startup. The older
WGMMA implementation remains an experimental oracle, not the H100 mining path.
A job advertising V4 or an unknown certificate version is paused with `update required`.

Mining credentials are supplied only through `STATEAA`, encoded as
`hex(endpoint|wallet|worker|log[|password])`. The fourth field is a console toggle: `1`/`true`/`on`
enables logs; `0`/`false`/`off` disables them, even if `RUST_LOG` is set. It is not
a filename and no log file is created. Python tunnel logging is independent
(`LOG_LEVEL=ERROR` suppresses its informational snapshots).
Use the endpoint currently published by
[Kryptex](https://pool.kryptex.com/prl). The decoded value is never logged.
The optional fifth field is the pool password; it defaults to `x` when omitted.
Existing four-field configurations remain valid. V1 and V2 authorization both
send the password, including a static difficulty request such as `d=2097152`.

For `proxy/` + `tunel.py`, use `localhost:9000` (or `tunnel://localhost:9000`).
The miner sends a hex X25519 handshake, then hex AES-GCM frames through the tunnel.
`prl7://localhost:3000` selects the older binary protocol in `tools/proxy/`.
Explicit `stratum+ssl://` and `stratum+tcp://` endpoints connect directly to a pool.
The miner does not connect directly to a WebSocket URL; configure that URL in `tunel.py`.

## Build

Requires Rust 1.88+, a C/C++ toolchain (`build-essential` on Ubuntu), and the
existing `pearl/` source dependency. For a mining binary, put CUDA 13 `nvcc` on
PATH (or set `NVCC=/usr/local/cuda/bin/nvcc`) before building. Cubins for sm_75, sm_90a
and sm_120a are embedded in the executable, so copying the binary is sufficient.


```sh
cargo build --release
cargo test
```

If `nvcc` is unavailable, the CUDA build script emits a warning and skips cubin generation. The
CPU reference and protocol tests remain usable without an NVIDIA driver.

## Offline benchmark and GPU self-test

```sh
cargo run --release -p pearl-miner -- --benchmark
cp config.example.toml config.toml
cargo run --release -p pearl-miner -- --selftest
cargo run --release -p pearl-miner -- --gpu-benchmark --device 0
```

The self-test must pass before the daemon is started. It compares the production scan
against the CPU reference, including every lane, transcript wrap, target equality,
queue overflow replay, and job replacement, and exits on any mismatch.
`--gpu-benchmark` uses production-sized resident matrices and counts MAC/s; it includes
launch and result-count readback but excludes setup and CPU proofs (setup is printed separately).
The daemon uses a disposable `--gpu-worker` child so a
driver failure stops mining without allowing an unverified share to reach the network.

## Auto tuning

The first normal miner start benchmarks each configured GPU's production scan, then caches the
best stable launch batch and supported scan mode under `~/.pearl-miner/tuning/`. Later starts
verify that profile briefly and retune if measured throughput falls by more than 15%. Profiles
are keyed by GPU UUID, name, ordinal, compute capability, driver/toolkit versions, and miner
kernel version; each configured GPU has an independent profile. The search uses coarse,
focused, fine, and sustained stability stages. SM75, SM90, and SM120 use separate candidate
spaces. Candidate runs are isolated in disposable miner processes, so CUDA faults and hangs do
not take down the mining worker.

```sh
./target/release/pearl-miner --tune
./target/release/pearl-miner --tune-fast
./target/release/pearl-miner --retune
./target/release/pearl-miner --show-tuning --device 0
./target/release/pearl-miner --clear-tuning --device 0
./target/release/pearl-miner --no-tune
./target/release/pearl-miner --tune-target 1
```

Set `devices = [0, 1]` under `[mining]` in `config.toml` to tune and mine each GPU with its
own worker and profile; `--device N` selects one device for a single-GPU run.

Tuning increases first-start time. `--tune-fast` is for development; regular tuning defaults
to 8 seconds warmup, 20 seconds per candidate, and a 120 second stability run. Use
`--tune-warmup`, `--tune-duration`, `--tune-stability`, and `--tune-max-temp-c` to adjust the
measurement policy. Share correctness remains guarded by the miner's CPU verification before
submission.

## Run from environment

```sh
export STATEAA="$(printf '%s' 'localhost:9000|YOUR_WALLET|worker-1|1' | xxd -p -c 9999)"
./target/release/pearl-miner
```

To request difficulty 2097152 through the tunnel:

```sh
export STATEAA="$(printf '%s' 'localhost:9000|YOUR_WALLET|worker-1|1|d=2097152' | xxd -p -c 9999)"
./target/release/pearl-miner
```

For a direct pool connection, replace `localhost:9000` with your explicit
`stratum+ssl://HOST:PORT` endpoint. Restart the miner after changing `STATEAA`.

The normal mining command takes no pool, wallet, worker, or log arguments. The child GPU worker
inherits `STATEAA` automatically.

## Package and run

```sh
cargo build --release -p pearl-miner
sudo install -Dm755 target/release/pearl-miner /usr/local/bin/pearl-miner
sudo install -Dm644 config.example.toml /etc/pearl-miner/config.toml
sudo install -Dm644 systemd/pearl-miner.service /etc/systemd/system/pearl-miner.service
printf 'STATEAA=%s\n' "$(printf '%s' 'stratum+ssl://prl-us.kryptex.network:8048|YOUR_WALLET|worker-1|1' | xxd -p -c 9999)" | sudo tee /etc/pearl-miner/stateaa.env >/dev/null
sudo chmod 600 /etc/pearl-miner/stateaa.env
sudo systemctl daemon-reload
sudo systemctl enable --now pearl-miner
```

The GPU self-test runs before mining. Local transport tests do not establish live-pool share acceptance.

## Containers

```sh
docker build -t pearl-miner:cuda13 .
export STATEAA="$(printf '%s' 'stratum+ssl://prl-us.kryptex.network:8048|YOUR_WALLET|worker-1|1' | xxd -p -c 9999)"
docker run --rm --gpus all --env RUST_LOG=info --env STATEAA="$STATEAA" -v "$PWD/config.toml:/etc/pearl-miner/config.toml:ro" pearl-miner:cuda13
```

## Run with proxy/ and tunel.py

The backend must speak the Pearl/Kryptex Stratum dialect. Set its host and TCP
port explicitly; the proxy's bundled CSD default is not a Pearl pool setting.

```sh
# Proxy host (one terminal)
cd proxy
npm ci
TCP_TARGET_HOST=YOUR_PEARL_POOL TCP_TARGET_PORT=YOUR_TCP_PORT npm start
```

```sh
# Miner host (another terminal, repository root)
python3 -m venv .venv
.venv/bin/pip install -r requirements-tunnel.txt
WS_URL=ws://127.0.0.1:8443 LISTEN_HOST=127.0.0.1 LISTEN_PORT=9000 .venv/bin/python tunel.py
```

Use `wss://YOUR_PROXY_HOST/path` in `WS_URL` when your proxy is behind a TLS
reverse proxy. The proxy key is discovered during the handshake; this mode does
not pin server identity. Start the miner using the `STATEAA` example above.
If WebSocket disconnects, the tunnel closes TCP so the miner can negotiate fresh keys.

Run the local interoperability test (no GPU or real wallet required):

```sh
.venv/bin/python tools/test_tunnel.py
```

It checks the real Rust → Python → Node → TCP path, fragmented writes, a 200 KB
payload, reconnect with fresh keys, and Stratum authorization/job delivery.

## GitHub Actions builds and releases

The `build` workflow runs on pushes to `main`, tags matching `v*`, pull requests,
and manual dispatch. It checks out the upstream `pearl/` dependency at a pinned
commit and builds with Rust 1.88 and CUDA 13.0 for sm_75 / sm_90a / sm_120a.
Commit the root `Cargo.lock` together with the workflow and miner sources;
CI uses `--locked`.

Download `pearl-miner-linux-x64-cuda13` from the run's Artifacts section. It contains
the standalone `pearl-miner` binary, a `.tar.gz` archive and a `.sha256` checksum
covering both files, retained for 30 days. The archive
includes the executable, example config, Python tunnel, requirements and service file.
The Linux x64 binary requires glibc 2.35 or newer and a compatible NVIDIA driver.

After committing and pushing the source changes, publish a version with:

```sh
git tag v0.1.0
git push origin v0.1.0
```

Once CPU, proxy/tunnel and CUDA build checks pass, the workflow creates the GitHub
Release and attaches the standalone binary, archive and checksum. After downloading
the standalone binary, run `chmod +x pearl-miner` before executing it.
Only the release job has
`contents: write`; PR builds cannot publish. Runner checks do not exercise GPU
execution or establish real-pool share acceptance.

## Production GPU mining: T4, H100 and RTX PRO 6000 Blackwell

The miner embeds separate sm_75, sm_90a and sm_120a cubins. The new scan groups
128 logical proof tiles in a 128×128 block, reuses matrix operands through two
shared-memory stages, keeps transcripts in registers and checks targets on GPU.
T4 uses `m8n8k16` plus register prefetch; H100 and Blackwell use `m16n8k32` plus
`cp.async`. This is a new grouped MMA path; the old WGMMA kernel is not selected.

Matrices and official noise remain in VRAM throughout an attempt. Only target
hits return to a bounded CPU proof worker. The scan continues after finding a
share and starts a fresh matrix attempt after exhausting its work. Every share
still passes the official CPU verification before submission.

Run on each target GPU:

```sh
./target/release/pearl-miner --selftest --device 0
./target/release/pearl-miner --gpu-benchmark --device 0
PEARL_GPU_DEVICE=0 bash tools/run_production_acceptance.sh
```

Without a GPU, validate source arithmetic and cubins with:

```sh
python3 tools/verify_production_kernel.py
python3 tools/verify_device_setup.py
bash tools/check_production_sass.sh
```

The grouped path has CPU arithmetic and compilation checks. GPU correctness,
sustained throughput and pool acceptance require the hardware checks above;
no target hashrate is guaranteed. Blackwell/H100 generate matrices, hash commitments
and apply noise on the GPU; CPU matrices are regenerated only for candidate proofs.
T4 retains CPU preparation with one-attempt lookahead. CPU proof work remains a
potential bottleneck. See [implementation and validation
notes](docs/production-gpu.md) for references, constraints and test coverage.

Console hashrates use logical `h × w × k` MAC/s: 1 TH/s = 10¹² MAC/s. Current
scan rate excludes preparation. The INFO heartbeat aggregates confirmed work across
attempts over at least ten seconds, including preparation and waiting. Submitted
rates include that attempt's preparation and scan time. Benchmarks print setup time
separately. These rates are neither BLAKE3 hashes/s nor Tensor Core TOPS.
`STATEAA` log=0 still disables mining console output.

## Diagnosing missing shares

INFO logging prints a mining heartbeat every ten seconds (at the next attempt
boundary) and submitted/accepted/rejected/stale totals every thirty seconds.
Preparation and per-attempt completion messages are DEBUG only; use
`RUST_LOG=info` for normal operation. A GPU completion counter must confirm every
batch before it contributes to hashrate. Startup additionally constructs and
verifies an official proof from an actual GPU candidate and encodes it as V2,
using a local synthetic target that is never submitted to the pool.
Candidate detection, CPU proof start/end, submission and pool responses remain
immediate logs. Pool acknowledgement timeout is thirty seconds; errors stop the
worker and propagate a failure exit status so systemd can restart it. The metrics
endpoint does not export a fabricated zero accepted-share counter.

At difficulty 2097152 and rank 128, approximate mean intervals are 12.5 hours at
200 GH/s, 10.7 minutes at 14 TH/s, and 7.5 minutes at 20 TH/s. These assume steady
credited work without stale/rejected shares. Kryptex documents 2097152 as PRL's
default and minimum: setting it explicitly does not lower an already-minimum target.
See [Kryptex's difficulty documentation](https://pool.kryptex.com/ja/articles/vardiff-vs-static-diff-en).

Proof creation now builds the matrix Merkle trees once per candidate, verifies that
proof, and serializes the same verified object. CPU proof failures propagate as
worker errors instead of being mistaken for successful shutdown when the share
channel closes. Local encoding/verification and tunnel tests do not establish
live-pool share acceptance.
