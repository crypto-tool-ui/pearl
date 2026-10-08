#!/usr/bin/env python3
"""Sweep a small set of matrix shapes with the miner's resident GPU benchmark."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


SHAPES = [
    (131072, 131072, 32768),
    (131072, 131072, 40960),
    (131072, 131072, 49152),
    (196608, 196608, 16384),
    (196608, 196608, 24576),
    (262144, 262144, 8192),
    (262144, 262144, 12288),
    (262144, 262144, 16384),
    (262144, 262144, 20480),
    (327680, 327680, 8192),
    (393216, 393216, 4096),
    (393216, 393216, 8192),
    (524288, 524288, 4096),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("./app"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gpu-benchmark", action="store_true",
                        help="accepted for compatibility; each shape already runs this mode")
    parser.add_argument("--seconds", type=int, default=5,
                        help="screening duration when supported by this miner version")
    parser.add_argument("--target", type=float, default=400.0, help="stop at this TH/s")
    parser.add_argument("--output", type=Path, default=Path("shape-search"))
    parser.add_argument("--shapes", nargs=3, type=int, action="append", metavar=("M", "N", "K"),
                        help="optional custom shape; may be repeated and replaces defaults")
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"miner binary is missing or not executable: {binary}")
    if args.seconds < 1 or args.device < 0:
        parser.error("seconds >= 1 and device >= 0 are required")

    help_result = subprocess.run([str(binary), "--help"], text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    supports_duration = "--gpu-benchmark-seconds" in help_result.stdout
    if not supports_duration:
        print("Miner has no --gpu-benchmark-seconds option; using its built-in GPU benchmark duration.")

    shapes = args.shapes or SHAPES
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop("STATEAA", None)
    rows = []
    for index, (m, n, k) in enumerate(shapes, 1):
        if m % 128 or n % 128 or not 2048 <= k <= 65536 or k % 128:
            print(f"skip invalid shape {m}x{n}x{k}")
            continue
        shape = f"{m}x{n}x{k}"
        run_env = dict(env, MATRIX_M=str(m), MATRIX_N=str(n), MATRIX_K=str(k))
        command = [str(binary), "--gpu-benchmark", "--device", str(args.device)]
        if supports_duration:
            command += ["--gpu-benchmark-seconds", str(args.seconds)]
        print(f"[{index}/{len(shapes)}] {shape}", flush=True)
        result = subprocess.run(command, env=run_env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        log = args.output / f"{shape}.log"
        log.write_text(result.stdout)
        record = next((json.loads(line.removeprefix("PEARL_BENCHMARK_JSON "))
                       for line in result.stdout.splitlines()
                       if line.startswith("PEARL_BENCHMARK_JSON ")), None)
        if result.returncode or record is None:
            rows.append({"shape": shape, "status": "failed", "th_s": "", "log": str(log)})
            print(f"  failed; see {log}")
            continue
        if (record.get("m"), record.get("n"), record.get("k")) != (m, n, k):
            rows.append({"shape": shape, "status": "unsupported_shape", "th_s": "", "log": str(log)})
            print(f"  miner did not run requested MATRIX_M/N/K; see {log}")
            continue
        th_s = float(record["th_s"])
        rows.append({"shape": shape, "status": "ok", "th_s": th_s, "log": str(log)})
        print(f"  {th_s:.3f} TH/s | VRAM details in {log}")
        if th_s >= args.target:
            print(f"Reached target {args.target:g} TH/s with {shape}")
            break

    rows.sort(key=lambda row: float(row["th_s"]) if row["th_s"] else -1, reverse=True)
    summary = args.output / "summary.csv"
    summary.write_text("shape,status,th_s,log\n" + "".join(
        f'{row["shape"]},{row["status"]},{row["th_s"]},{row["log"]}\n' for row in rows))
    print(f"Best: {rows[0]['shape']} at {rows[0]['th_s']} TH/s" if rows and rows[0]["th_s"]
          else "No successful GPU benchmark. Check per-shape logs.")
    print(f"Results: {summary}")
    return 0 if any(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
