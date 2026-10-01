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

The miner accepts `SCAN_WAVES=1..128` and `SCAN_BAND=1..128` for controlled
experiments. Unset defaults remain 16 waves, band 4 on T4 or 8 on H100/Blackwell.
The sweep exports its highest-median setting to `best.env`; it is not applied automatically. Validate any winning setting with a
sustained mining session and real accepted shares before deploying it broadly.

NVIDIA references:
- https://docs.nvidia.com/nsight-systems/UserGuide/
- https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html

## RTX PRO 6000 Blackwell parameter sweep

Use a miner rebuilt from this revision: earlier binaries reject bands above 32 or
waves above 64. No CUDA source or GitHub Actions changes are needed. This tool uses
Python's standard library and the existing driver tools; it installs nothing.

```sh
python3 tools/profile_production.py --binary ./pearl-miner --device 0 \
  --preset blackwell --repeats 3 --output blackwell-tuning
```

This tests waves 16/32/64/128 against bands 8/32/64/128, including the unchanged
16/8 baseline. It runs 48 measurements of ten seconds plus preparation/self-tests.
Larger waves amortize host launch/readback costs; a larger band changes reuse of
A/B rows in L2. Neither parameter is an occupancy guarantee or a universal optimum.
At K=4096, a band of 128 covers 64 MiB of A rows (excluding B and other cache users).
The launch limit rises from 16,384 to 32,768 groups to accommodate the wider sweep.
Larger launches can delay response to job changes; check live stale-share behavior.

Inspect `summary.json` and telemetry, then apply the measured configuration in the
same shell that starts the miner:

```sh
. ./blackwell-tuning/best.env
# Start pearl-miner with your usual arguments in this shell.
```

The exported setting is selected from resident-scan medians, not a promised mining
rate. The reported 261.562 TH/s average versus 280.040 TH/s scan suggests about 6.6%
of that interval is outside scan throughput, assuming comparable measurement windows.
Even eliminating this gap would not reach 420 TH/s: the kernel scan itself needs
roughly 50% more throughput. If this sweep plateaus, profile kernel utilization,
register spills and memory traffic rather than continuing to increase launch size.
See NVIDIA's [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
for SM120 occupancy and memory limits, which differ from SM100.

Local tooling checks (no GPU): `python3 tools/test_profile_production.py` exercises
all 32 default Blackwell sweep runs with a fake benchmark, verifies winner export,
and checks the raster formula for full and partial bands. It does not measure speed.

## Matrix-size experiment (rebuilt binary required)

Live mining and `--gpu-benchmark` now share `MATRIX_M`, `MATRIX_N`,
`MATRIX_K`. Defaults remain 131072/131072/4096; rank remains 128 and the
8x16 proof pattern is unchanged. M/N must be multiples of 128 and at most 2^24;
K must be a multiple of 128 in 2048..65536. Additional generator/group-index
limits are checked before allocation. The original verifier permits these shapes;
pool acceptance must still be checked with real shares.

Build all cubins and the Rust host together: the noise kernel now uses a 64-bit
cell count and index. Mixing old cubins with the new host is unsupported.

First compare shapes at the same waves/band, on an idle target GPU:

```sh
python3 tools/profile_production.py --binary ./pearl-miner --waves 16 --bands 8 --repeats 3 --output shape-1g
python3 tools/profile_production.py --binary ./pearl-miner --matrix-m 262144 --matrix-n 262144 --matrix-k 4096 --waves 16 --bands 8 --repeats 3 --output shape-2g
python3 tools/profile_production.py --binary ./pearl-miner --matrix-m 262144 --matrix-n 262144 --matrix-k 8192 --waves 16 --bands 8 --repeats 3 --output shape-4g
python3 tools/profile_production.py --binary ./pearl-miner --matrix-m 262144 --matrix-n 262144 --matrix-k 16384 --waves 16 --bands 8 --repeats 3 --output shape-8g
```

The GiB names refer only to A+B INT8 storage, not total VRAM. GPU generation/hash
scratch and noise buffers add memory. CPU proof construction materializes matrices
and Merkle data, so host RAM and proof latency must also be measured. A faster
resident scan is not sufficient if proof processing falls behind or shares become stale.

The profile script verifies the actual M/N/K reported by the binary, records the
shape in `manifest.json`, and includes it in `best.env`. An old binary that ignores
the requested dimensions is rejected. After comparing summaries, run the Blackwell
parameter sweep with the selected `--matrix-m/--matrix-n/--matrix-k`, then source
that run's `best.env` in the shell used to start live mining. The shape flags on the
profile script explicitly override inherited matrix environment variables.

Local validation: `python3 tools/test_profile_production.py` and
`python3 tools/test_noise_indexing.py`. The latter exercises the actual noise
kernel arithmetic with sparse CPU storage at the 4 GiB boundary, not CUDA ABI or
scheduling. Rust recipe tests cover default compatibility, large deferred shapes
and invalid dimensions; run `cargo test -p pearl-ref` on an existing Rust host.
