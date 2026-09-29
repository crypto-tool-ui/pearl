# M4 sm_90a acceptance

M4 targets the V3 dense `Int7xInt7ToInt32` path on Hopper `sm_90a`. The implementation uses
`wgmma.mma_async` with signed INT8 operands, four consumer warps per output tile, and a dedicated
producer warp. It uses two K32 shared-memory stages, TMA bulk copies completed by transaction
barriers, and a register accumulator followed by the fused transcript fold and keyed BLAKE3 digest.

The no-GPU SASS gate is:

```text
NVCC=/usr/local/cuda/bin/nvcc \
CUOBJDUMP=/usr/local/cuda/bin/cuobjdump \
tools/check_sm90_sass.sh
```

It must report INT8 IGMMA and bulk-copy UBLKCP in SASS and verify the source contract for WGMMA fence/commit/wait,
TMA bulk copies, transaction barriers, the sm_90a architecture guard, and the V3 fused digest.

Hardware acceptance requires an H100 or another compute-capability-9.0 device:

```text
tools/run_sm90_acceptance.sh
```

The wrapper selects the first compute-capability-9.0 device and runs the V3 golden vector, 10,000
independent comparisons against the naive CUDA oracle, and the benchmark. The benchmark prints
`sm_90 benchmark: <tiles>/s, <effective> effective TH/s`.

M4 is not accepted until the SASS gate and the hardware wrapper pass. A host without an NVIDIA
driver or device can verify only source/build setup; those checks remain `needs hardware`.
