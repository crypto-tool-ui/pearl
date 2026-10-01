#!/usr/bin/env python3
"""Execute the noise kernel's index arithmetic with sparse CPU storage, including 4 GiB.
Not a CUDA launch/ABI test. Requires only an existing C++ compiler.
"""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'cuda/sm75_mma.cu').read_text().split('extern "C" __global__ void pearl_sm75_apply_noise(')[1]
source = 'void pearl_sm75_apply_noise(' + source
source = source.replace('int8_t* matrix', 'Sparse& matrix').replace('const int8_t* uniform', 'Sparse& uniform')
preamble = '''
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <map>
#include <vector>
using Sparse = std::map<uint64_t, int8_t>;
struct Dim { uint32_t x; } blockIdx{}, blockDim{256}, threadIdx{};
'''
main = '''
int main() {
  constexpr uint32_t k = 16384, rank = 128;
  std::vector<uint32_t> perm(k * 2);
  for (uint32_t c = 0; c < k; ++c) perm[c * 2 + 1] = 1;
  for (uint64_t index : std::vector<uint64_t>{0, (1ull << 32) - 1, 1ull << 32, (1ull << 32) + 1}) {
    Sparse matrix, noise;
    matrix[index] = 10;
    noise[(index / k) * rank] = 7;
    noise[(index / k) * rank + 1] = -3;
    blockIdx.x = uint32_t(index / 256);
    threadIdx.x = uint32_t(index % 256);
    pearl_sm75_apply_noise(matrix, index + 1, k, rank, noise, perm.data());
    assert(matrix.size() == 1 && matrix[index] == 20);
    pearl_sm75_apply_noise(matrix, index, k, rank, noise, perm.data());
    assert(matrix.size() == 1 && matrix[index] == 20);
  }
}
'''
with tempfile.TemporaryDirectory(prefix='pearl-noise-index-') as temp:
    cpp, exe = Path(temp) / 'test.cpp', Path(temp) / 'test'
    cpp.write_text(preamble + source + main)
    subprocess.run(shlex.split(os.environ.get('CXX', 'c++')) + ['-std=c++17', str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
print('PASS: actual noise index arithmetic at 0, 2^32-1, 2^32, 2^32+1 and bounds')
