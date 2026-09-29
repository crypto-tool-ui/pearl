#!/usr/bin/env python3
"""Offline production scan sweep and optional NVIDIA Nsight capture. No pool access."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys


def run(command, path, env, timeout=300):
    with path.open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env=env, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'Command exited {result.returncode}; inspect {path}')
    return path.read_text()


def benchmark_result(output):
    records = [json.loads(line.removeprefix('PEARL_BENCHMARK_JSON '))
               for line in output.splitlines() if line.startswith('PEARL_BENCHMARK_JSON ')]
    if len(records) != 1 or records[0]['th_s'] <= 0 or records[0]['tiles'] <= 0:
        raise RuntimeError('Missing/invalid benchmark JSON; use the bundled miner')
    return records[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path(__file__).resolve().parent.parent / 'pearl-miner')
    parser.add_argument('--device', type=int, default=0, help='CUDA ordinal as seen by miner')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--waves', nargs='+', type=int, default=[4, 16, 64])
    parser.add_argument('--bands', nargs='+', type=int, default=[4, 8, 16])
    parser.add_argument('--profile', choices=['none', 'nsys', 'ncu', 'all'], default='none')
    parser.add_argument('--output', type=Path, default=Path('pearl-profile-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    args = parser.parse_args()
    if args.repeats < 1 or args.device < 0 or any(not 1 <= x <= 64 for x in args.waves) or any(not 1 <= x <= 32 for x in args.bands):
        parser.error('repeats >= 1, device >= 0, waves 1..64, bands 1..32 required')
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f'miner not found: {binary}; pass --binary /path/to/pearl-miner')
    if not shutil.which('nvidia-smi'):
        parser.error('nvidia-smi is required on the target GPU machine')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'manifest.json').write_text(json.dumps({
        'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'device': args.device, 'waves': args.waves, 'bands': args.bands,
        'repeats': args.repeats, 'profile': args.profile,
        'started_utc': datetime.now(timezone.utc).isoformat()
    }, indent=2) + '\n')
    env = os.environ.copy()
    env.pop('STATEAA', None)
    env['RUST_LOG'] = 'info'
    run(['nvidia-smi', '-q'], args.output / 'gpu-before.txt', env)
    tasks = [(w, b, r) for w in args.waves for b in args.bands for r in range(args.repeats)]
    random.Random(20260929).shuffle(tasks)  # Interleave repeats to reduce order/temperature bias.
    print(f'{len(tasks)} runs, ten seconds of measured scanning per run plus self-tests.', flush=True)
    query = 'timestamp,index,name,uuid,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem'
    records = []
    with (args.output / 'telemetry.csv').open('w') as telemetry:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=' + query, '--format=csv', '-l', '1'],
                                   stdout=telemetry, stderr=subprocess.STDOUT, env=env)
        try:
            for idx, (waves, band, repeat) in enumerate(tasks, 1):
                child_env = dict(env, PEARL_SCAN_WAVES=str(waves), PEARL_SCAN_BAND=str(band))
                label = f'w{waves}-b{band}-r{repeat + 1}'
                print(f'[{idx}/{len(tasks)}] {label}', flush=True)
                output = run([str(binary), '--gpu-benchmark', '--device', str(args.device)],
                             args.output / (label + '.log'), child_env)
                record = benchmark_result(output)
                record.update(waves=waves, band=band, repeat=repeat + 1)
                records.append(record)
                (args.output / 'runs.json').write_text(json.dumps(records, indent=2) + '\n')
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
    summary = []
    for waves in args.waves:
        for band in args.bands:
            rates = [r['th_s'] for r in records if r['waves'] == waves and r['band'] == band]
            summary.append(dict(waves=waves, band=band, median_th_s=statistics.median(rates),
                                min_th_s=min(rates), max_th_s=max(rates)))
    summary.sort(key=lambda r: r['median_th_s'], reverse=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    best = summary[0]
    print('Fastest observed resident scan (not live mining): ' + json.dumps(best), flush=True)
    child_env = dict(env, PEARL_SCAN_WAVES=str(best['waves']), PEARL_SCAN_BAND=str(best['band']))
    command = [str(binary), '--gpu-benchmark', '--device', str(args.device)]
    if args.profile in ('nsys', 'all'):
        if not shutil.which('nsys'):
            raise RuntimeError('nsys not installed; baseline results saved. Install NVIDIA Nsight Systems to capture timeline.')
        run(['nsys', 'profile', '--trace=cuda', '--sample=none', '--stats=true',
             '--output', str(args.output / 'timeline')] + command,
            args.output / 'nsys.log', child_env, timeout=600)
    if args.profile in ('ncu', 'all'):
        if not shutil.which('ncu'):
            raise RuntimeError('ncu not installed; baseline results saved. Install NVIDIA Nsight Compute to capture counters.')
        # Skip the small startup tests; inspect captured grid dimensions against runs.json.
        # Profiling replays distort wall-clock timing: never add these results to summary.json.
        run(['ncu', '--set', 'full', '--kernel-name', 'regex:pearl_production_scan',
             '--launch-skip', '128', '--launch-count', '1',
             '--export', str(args.output / 'kernel')] + command,
            args.output / 'ncu.log', child_env, timeout=900)
    run(['nvidia-smi', '-q'], args.output / 'gpu-after.txt', env)
    print(f'Results saved to {args.output.resolve()}')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
