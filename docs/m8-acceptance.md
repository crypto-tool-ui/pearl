# M8 operations acceptance

The no-GPU checks are:

```sh
cargo build --release --locked
cargo test --locked
cargo run --release -p pearl-miner -- --benchmark
```

Expected output is a successful release build, passing CPU-only tests, and a line beginning
`CPU reference benchmark:`. On a GPU machine, run:

```sh
cargo run --release -p pearl-miner -- --config config.toml --selftest
systemctl start pearl-miner
```

The self-test must print a structured `GPU self-test passed` event before the service is allowed to
mine. A V4 or unknown job must produce `update required` and leave the worker paused. The M6
controlled gateway/Kryptex test remains a prerequisite for enabling pool submissions.

Actual acceptance is pending because the user requested that tests be skipped in this environment.
The following remain **not verified (needs hardware/external service)**: CUDA 13 release packaging,
GPU self-test on sm_90a and sm_120a, systemd startup, container startup with an NVIDIA runtime,
Prometheus endpoint operation, Kryptex TLS/failover, and an accepted share.
