#!/usr/bin/env python3
import json
import re
import statistics
import sys

tag, log_path, sample_path, before_path, after_path = sys.argv[1:]
log = open(log_path, encoding="utf-8", errors="replace").read()
match = re.search(r"PEARL_BENCHMARK_JSON\s+(\{.*\})", log)
if not match:
    raise SystemExit(f"{tag}: benchmark JSON missing")
bench = json.loads(match.group(1))
mem_match = re.search(r"PEARL_VRAM_JSON\s+(\{.*\})", log)
cuda_used_mib = (json.loads(mem_match.group(1))["used_bytes"] / (1024 * 1024)) if mem_match else None
samples = []
for line in open(sample_path, encoding="utf-8", errors="replace"):
    fields = [x.strip() for x in line.split(",")]
    if len(fields) >= 3:
        try:
            samples.append(tuple(float(x) for x in fields[:3]))
        except ValueError:
            pass
power = statistics.mean(x[0] for x in samples) if samples else 0.0
clock = statistics.mean(x[1] for x in samples) if samples else 0.0
vram = cuda_used_mib if cuda_used_mib is not None else max((x[2] for x in samples), default=0.0)
def power_cap_state(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    vals = re.findall(r"SW Power (?:Capping|Cap)\s*:\s*(\w+)", text, re.I)
    return vals[-1].lower() if vals else "unknown"
print("\t".join(str(x) for x in (
    tag, bench["mem_mode"], bench["groups_per_launch"], bench["th_s"],
    bench["tiles"] / bench["scan_seconds"], vram, power, clock,
    bench["th_s"] / power if power > 0 else 0,
    power_cap_state(before_path), power_cap_state(after_path))))
