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
    parser.add_argument('--matrix-m', type=int, default=131072)
    parser.add_argument('--matrix-n', type=int, default=131072)
    parser.add_argument('--matrix-k', type=int, default=4096)
    parser.add_argument('--preset', choices=['default', 'blackwell'], default='default')
    parser.add_argument('--waves', nargs='+', type=int)
    parser.add_argument('--bands', nargs='+', type=int)
    parser.add_argument('--profile', choices=['none', 'nsys', 'ncu', 'all'], default='none')
    parser.add_argument('--output', type=Path, default=Path('pearl-profile-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    args = parser.parse_args()
    args.waves = args.waves or ([16, 32, 64, 128] if args.preset == 'blackwell' else [4, 16, 64])
    args.bands = args.bands or ([8, 32, 64, 128] if args.preset == 'blackwell' else [4, 8, 16])
    if args.repeats < 1 or args.device < 0 or any(not 1 <= x <= 128 for x in args.waves) or any(not 1 <= x <= 128 for x in args.bands):
        parser.error('repeats >= 1, device >= 0, waves 1..128, bands 1..128 required')
    if (any(x <= 0 or x > 2**24 or x % 128 for x in (args.matrix_m, args.matrix_n))
            or not 2048 <= args.matrix_k <= 65536 or args.matrix_k % 128):
        parser.error('M/N must be positive multiples of 128 <= 2^24; K=2048..65536, multiple of 128')
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f'miner not found: {binary}; pass --binary /path/to/pearl-miner')
    if not shutil.which('nvidia-smi'):
        parser.error('nvidia-smi is required on the target GPU machine')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'manifest.json').write_text(json.dumps({
        'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'device': args.device, 'preset': args.preset, 'waves': args.waves, 'bands': args.bands,
        'repeats': args.repeats, 'profile': args.profile,
        'm': args.matrix_m, 'n': args.matrix_n, 'k': args.matrix_k,
        'started_utc': datetime.now(timezone.utc).isoformat()
    }, indent=2) + '\n')
    env = os.environ.copy()
    env.pop('STATEAA', None)
    env['RUST_LOG'] = 'info'
    env.update(MATRIX_M=str(args.matrix_m), MATRIX_N=str(args.matrix_n),
               MATRIX_K=str(args.matrix_k))
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
                child_env = dict(env, SCAN_WAVES=str(waves), SCAN_BAND=str(band))
                label = f'w{waves}-b{band}-r{repeat + 1}'
                print(f'[{idx}/{len(tasks)}] {label}', flush=True)
                output = run([str(binary), '--gpu-benchmark', '--device', str(args.device)],
                             args.output / (label + '.log'), child_env)
                record = benchmark_result(output)
                if (record.get('m'), record.get('n'), record.get('k')) != (args.matrix_m, args.matrix_n, args.matrix_k):
                    raise RuntimeError('Benchmark shape mismatch; rebuild miner with matrix parameter support')
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
    # Only numeric, validated parameters enter this shell file. Never include credentials.
    (args.output / 'best.env').write_text(
        '# Highest median resident scan in this sweep; validate during live mining.\n'
        f'export SCAN_WAVES={best["waves"]}\n'
        f'export SCAN_BAND={best["band"]}\n'
        f'export MATRIX_M={args.matrix_m}\n'
        f'export MATRIX_N={args.matrix_n}\n'
        f'export MATRIX_K={args.matrix_k}\n')
    baseline = next((r for r in summary if r['waves'] == 16 and r['band'] == 8), None)
    if baseline:
        gain = 100 * (best['median_th_s'] / baseline['median_th_s'] - 1)
        print(f'Compared with waves=16 band=8: {gain:+.2f}% median resident scan', flush=True)
    print('Fastest observed resident scan (not live mining): ' + json.dumps(best), flush=True)
    child_env = dict(env, SCAN_WAVES=str(best['waves']), SCAN_BAND=str(best['band']))
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
