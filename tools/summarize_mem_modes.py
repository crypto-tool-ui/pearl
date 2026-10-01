#!/usr/bin/env python3
import csv
import statistics
import sys
from collections import defaultdict

runs = defaultdict(list)
power_states = defaultdict(list)
with open(sys.argv[1], newline="") as f:
    for row in csv.reader(f, delimiter="\t"):
        if len(row) != 11:
            continue
        tag, mode, groups, rate, tiles, vram, power, clock, efficiency, cap0, cap1 = row
        runs[(mode, groups)].append(tuple(map(float, (rate, tiles, vram, power, clock, efficiency))))
        power_states[(mode, groups)].append((cap0, cap1))
print("mode groups | MAC/s T (min/median/max) | tiles/s (min/median/max) | VRAM MiB max | W avg | SM MHz avg | T MAC/s/W median | power-cap state changes")
for (mode, groups), values in sorted(runs.items()):
    cols = list(zip(*values))
    rate = cols[0]
    print(f"{mode:9s} {groups:5s} | {min(rate):.2f}/{statistics.median(rate):.2f}/{max(rate):.2f} | "
          f"{min(cols[1]):.0f}/{statistics.median(cols[1]):.0f}/{max(cols[1]):.0f} | "
          f"{max(cols[2]):.0f} | {statistics.mean(cols[3]):.1f} | {statistics.mean(cols[4]):.0f} | "
          f"{statistics.median(cols[5]):.3f} | "
          f"{sum(a != b for a, b in power_states[(mode, groups)])}/{len(values)} changed")
