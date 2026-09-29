# Pearl V3 GPU miner

This repository targets Pearl certificate V3 with the official INT7 consensus arithmetic represented
by INT8 tensor-core operands. Runtime dispatch selects sm_75 `mma.sync` on T4 or
sm_120a `mma.sync` on RTX PRO 6000 Blackwell. The sm_90a WGMMA kernel is built,
but H100/H200 mining dispatch remains blocked pending hardware acceptance. A job advertising V4 or an unknown certificate version is paused with `update required`.

Mining credentials are supplied only through `STATEAA`, encoded as
`hex(endpoint|wallet|worker|log)`. Use the endpoint currently published by
[Kryptex](https://pool.kryptex.com/prl). The decoded value is never logged.

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
```

The self-test must pass before the daemon is started. It compares one production-kernel tile with
the CPU reference and exits on any mismatch. The daemon uses a disposable `--gpu-worker` child so a
driver failure stops mining without allowing an unverified share to reach the network.

## Run from environment

```sh
export STATEAA="$(printf '%s' 'localhost:9000|YOUR_WALLET|worker-1|/var/log/pearl-miner.jsonl' | xxd -p -c 9999)"
./target/release/pearl-miner
```

The normal mining command takes no pool, wallet, worker, or log arguments. The child GPU worker
inherits `STATEAA` automatically.

## Package and run

```sh
cargo build --release -p pearl-miner
sudo install -Dm755 target/release/pearl-miner /usr/local/bin/pearl-miner
sudo install -Dm644 config.example.toml /etc/pearl-miner/config.toml
sudo install -Dm644 systemd/pearl-miner.service /etc/systemd/system/pearl-miner.service
printf 'STATEAA=%s\n' "$(printf '%s' 'stratum+ssl://prl-us.kryptex.network:8048|YOUR_WALLET|worker-1|/var/log/pearl-miner.jsonl' | xxd -p -c 9999)" | sudo tee /etc/pearl-miner/stateaa.env >/dev/null
sudo chmod 600 /etc/pearl-miner/stateaa.env
sudo systemctl daemon-reload
sudo systemctl enable --now pearl-miner
```

The GPU self-test runs before mining. Local transport tests do not establish live-pool share acceptance.

## Containers

```sh
docker build -t pearl-miner:cuda13 .
export STATEAA="$(printf '%s' 'stratum+ssl://prl-us.kryptex.network:8048|YOUR_WALLET|worker-1|/var/log/pearl-miner.jsonl' | xxd -p -c 9999)"
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

## NVIDIA T4 (sm_75)

The release binary embeds a separate Turing INT8 Tensor Core kernel using
`mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32`. Compute capability 7.5 selects
it automatically; the usual GPU self-test and CPU share verification still apply.
It loads operands into registers without cp.async or TMA, pads tiles to 8×8,
and retains the V3 cumulative transcript and fused keyed BLAKE3 digest.
The naive oracle and tensor-hash kernels also build for sm_75.

On a T4 host, build with CUDA 13, then run:

```sh
bash tools/check_sm75_sass.sh
bash tools/run_sm75_acceptance.sh
./target/release/pearl-miner --selftest --device 0
```

The acceptance script requires a real 7.5 GPU and fails on missing prerequisites.
It checks the golden tile, production dispatch, 10,000 randomized tiles against
both CPU and naive CUDA, and prints a tile benchmark. This path has been compiled
and its IMMA instruction verified; GPU correctness, hashrate and pool acceptance
remain unverified until those hardware checks run. This is an initial T4 path,
not a claim of maximum mining throughput.
