# M5 GPU job setup acceptance

M5 moves the official keyed BLAKE3 matrix-root setup to `cuda/tensor_hash.cu`. The leaf kernel
hashes 1024-byte chunks with the official keyed flags and the parent kernel reduces the chaining
values level by level, applying `ROOT` only at the final chunk or parent. `GpuJobSetup` then uses
the official root binding and seed derivation from `pearl-ref`.

The runtime layer has two job slots per GPU, one CUDA worker/context per selected device, separate
streams for the slots, cancellation state, clean worker restart, autotune cache persistence, and
operator power/clock options through `nvidia-smi`.

The expected hardware acceptance command is:

```text
cargo test -p pearl-gpu --features gpu m5 -- --nocapture
```

It must compare GPU roots and derived commitments with the CPU reference, exercise both one-chunk
and multi-level Merkle inputs, build the next slot while the current slot is running, enumerate all
selected devices, and print the autotuned tile/stage choice and benchmark. A real GPU and driver
are required; no-GPU CI can only compile the cubins and Rust host layer.
