# M3 sm_120a acceptance runbook

M3 targets the official V3 dense `Int7xInt7ToInt32` path on an NVIDIA RTX PRO 6000 Blackwell
(`sm_120a`). The naive CUDA kernel remains a correctness oracle; production dispatch selects the
tensor-core kernel only after the device reports compute capability 12.0.

Run the no-GPU SASS gate from the repository root:

```text
NVCC=/usr/local/cuda/bin/nvcc \
CUOBJDUMP=/usr/local/cuda/bin/cuobjdump \
tools/check_sm120_sass.sh
```

Expected SASS output contains nonzero counts for `IMMA`, `LDSM`, and `LDGSTS`. The source gate also
requires the exact signed INT8 MMA instruction, `cp.async` commit/wait groups, the two `ldmatrix`
forms, the `sm_120a` compile guard, and rejects `tcgen05` and TMA forms.

Run hardware acceptance on a host with the NVIDIA driver and an RTX PRO 6000 Blackwell:

```text
tools/run_sm120_acceptance.sh
```

The wrapper selects the first detected compute-capability-12.0 ordinal and sets
`PEARL_REQUIRE_GPU_ACCEPTANCE=1`. The test suite must then report:

- the V3 golden tile matches the CPU reference;
- 10,000 independently committed randomized comparisons match the naive oracle, alternating the
  official 4×8/rank32 shape and the 16×16/rank128 two-stage pipeline shape;
- the benchmark prints `sm_120 benchmark: <tiles>/s, <effective> effective TH/s` and identifies
  the selected device.

M3 acceptance is not complete until both the SASS gate and the hardware wrapper pass. A host without
an NVIDIA driver or device can verify only source/build setup and is recorded as `needs hardware`.
Both acceptance scripts are committed with executable mode and should be invoked from the repository
root.
