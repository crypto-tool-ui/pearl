#!/usr/bin/env python3
"""Summarize tuner scan medians alongside nvidia-smi power telemetry."""
import csv
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
rows = []
for run_dir in sorted(root.glob('scan-*W')):
    limit = int(run_dir.name.removeprefix('scan-').removesuffix('W'))
    summary = json.loads((run_dir / 'summary.json').read_text())
    best = summary[0]
    readings = []
    with (run_dir / 'telemetry.csv').open(newline='') as source:
        for row in csv.DictReader(source):
            if row.get('index', '').strip() != '0':
                continue
            key = next((k for k in row if k.startswith('power.draw')), None)
            try:
                readings.append(float(row[key].split()[0]))
            except (KeyError, ValueError, IndexError):
                continue
    power = statistics.median(readings) if readings else float('nan')
    rows.append((limit, best['waves'], best['band'], best['median_th_s'], power,
                 best['median_th_s'] / power if power > 0 else float('nan')))
if not rows:
    raise SystemExit(f'no completed scan-*W results in {root}')
path = root / 'power-summary.csv'
with path.open('w', newline='') as output:
    writer = csv.writer(output)
    writer.writerow(['limit_W', 'waves', 'band', 'median_T_MAC_s', 'median_power_W', 'T_MAC_s_per_W'])
    writer.writerows(rows)
print(f'Wrote {path}')
for row in rows:
    print('%dW waves=%d band=%d rate=%.3f T_MAC/s power=%.1fW efficiency=%.4f T_MAC/s/W' % row)
