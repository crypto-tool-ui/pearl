# Profiling the production scan

This measures the current kernel on the actual GPU. It does not promise 420 TH/s
or change pool difficulty, proof arithmetic, clocks, or power limits. Stop other
GPU workloads, including the live miner, before comparing configurations.

From the extracted release directory:

```sh
python3 tools/profile_production.py --device 0
```

Python 3 and nvidia-smi are required. The default sweep runs nine combinations
of batch waves (4, 16, 64 CTAs per SM) and raster bands (4, 8, 16), each twice in
interleaved order. Each run performs mandatory correctness self-tests, prepares
full-size matrices and measures ten seconds of resident scanning. Allow several
minutes for the complete sweep. Output directories are timestamped and existing
results are never overwritten.

Results:

- `summary.json`: median/min/max resident TH/s for each configuration.
- `runs.json` and per-run logs: exact shape, launch count, preparation time and
  measured scan time. Device-written completion counters gate all reported work.
- `telemetry.csv`: physical GPU identity, utilization, temperature, power and clocks.
- `gpu-before.txt`, `gpu-after.txt`: driver/device details and clock-event information.

CUDA_VISIBLE_DEVICES changes miner ordinals. Telemetry intentionally captures all
physical GPUs, with UUIDs, so the selected device can be mapped correctly.
No wallet or STATEAA contents are collected or passed to benchmark processes.
Do not interpret resident scan TH/s as live-pool effective throughput.

If NVIDIA Nsight Systems and Compute are already installed, capture both:

```sh
python3 tools/profile_production.py --device 0 --profile all
```

This first runs the unprofiled sweep, then profiles its fastest observed setting.
Nsight Systems captures CUDA API/kernel/copy timing in `timeline.nsys-rep` and
`nsys.log`. Nsight Compute captures one matching scan invocation in `kernel.ncu-rep`
and `ncu.log`, skipping 128 matching launches to bypass startup test cases. Inspect
the captured grid against `groups_per_launch` in runs.json to confirm a production
batch. Profiler-instrumented hashrates are excluded from the sweep summary because
instrumentation and replay distort wall-clock throughput.

Missing Nsight tools or counter permission errors stop the optional capture with
an explicit error; completed baseline files remain available. The script does not
install tools or modify driver permissions. Consult the GPU administrator for
counter access when necessary.

To isolate a configuration, for example:

```sh
python3 tools/profile_production.py --device 0 --waves 16 --bands 8 --repeats 2 --profile nsys
```

The miner accepts `PEARL_SCAN_WAVES=1..64` and `PEARL_SCAN_BAND=1..32` for controlled
experiments. Unset defaults remain 16 waves, band 4 on T4 or 8 on H100/Blackwell.
Selection is not persisted automatically. Validate any winning setting with a
sustained mining session and real accepted shares before deploying it broadly.

NVIDIA references:
- https://docs.nvidia.com/nsight-systems/UserGuide/
- https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
