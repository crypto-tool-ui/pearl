#!/usr/bin/env python3
"""Exercise parameter selection and export without CUDA or third-party packages."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProfileTests(unittest.TestCase):
    def test_blackwell_sweep_exports_measured_winner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            smi = root / 'nvidia-smi'
            smi.write_text('#!/bin/sh\nexit 0\n')
            smi.chmod(0o755)
            miner = root / 'miner'
            miner.write_text(f'#!{sys.executable}\n' + '''import json, os
assert 'STATEAA' not in os.environ
w = int(os.environ['SCAN_WAVES'])
b = int(os.environ['SCAN_BAND'])
rate = 350 if (w, b) == (64, 32) else 280
print('PEARL_BENCHMARK_JSON ' + json.dumps({'tiles': 128, 'th_s': rate, 'm': int(os.environ['MATRIX_M']), 'n': int(os.environ['MATRIX_N']), 'k': int(os.environ['MATRIX_K'])}))
''')
            miner.chmod(0o755)
            output = root / 'results'
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'], STATEAA='test')
            result = subprocess.run([
                sys.executable, str(ROOT / 'tools/profile_production.py'),
                '--binary', str(miner), '--preset', 'blackwell', '--output', str(output),
                '--matrix-m', '262144', '--matrix-n', '262144', '--matrix-k', '16384',
            ], env=env, text=True, capture_output=True, check=True)
            runs = json.loads((output / 'runs.json').read_text())
            self.assertEqual(len(runs), 32)
            self.assertEqual({r['waves'] for r in runs}, {16, 32, 64, 128})
            self.assertEqual({r['band'] for r in runs}, {8, 32, 64, 128})
            exported = (output / 'best.env').read_text()
            self.assertIn('export SCAN_WAVES=64\n', exported)
            self.assertIn('export SCAN_BAND=32\n', exported)
            self.assertIn('export MATRIX_K=16384\n', exported)
            self.assertIn('export MATRIX_M=262144\n', exported)
            self.assertIn('+25.00%', result.stdout)
            self.assertNotIn('STATEAA', exported)

    def test_expanded_raster_is_bijective_including_tail_bands(self):
        # The unchanged production.cu raster formula must cover every group once.
        for rows in (1, 2, 33, 129, 1024):
            cols = 17
            for band in (1, 8, 32, 64, 128):
                mapped = set()
                for index in range(rows * cols):
                    first = index // (band * cols) * band
                    band_rows = min(band, rows - first)
                    local = index - index // (band * cols) * band * cols
                    row, col = first + local % band_rows, local // band_rows
                    self.assertLess(row, rows)
                    self.assertLess(col, cols)
                    mapped.add((row, col))
                self.assertEqual(len(mapped), rows * cols)


if __name__ == '__main__':
    unittest.main()
