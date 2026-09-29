#!/usr/bin/env python3
"""Exercise the actual CUDA noise routines on CPU against official Rust vectors.
This validates arithmetic/seed layout, not CUDA scheduling or Tensor Core execution.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'cuda/sm75_mma.cu').read_text()
# Compile only the scalar preparation functions; MMA still needs real GPU tests.
helpers = source[:source.index('__device__ __forceinline__ void mma_int8')]
kernels = source[source.index('extern "C" __global__ void pearl_sm75_uniform'):]
preamble = r'''
#include <iostream>
#include <vector>
#define __device__
#define __constant__
#define __global__
#define __forceinline__ inline
struct Dim { unsigned x; } blockIdx, blockDim, threadIdx;
inline unsigned __funnelshift_r(unsigned a, unsigned b, unsigned s) {
  return (a >> s) | (b << (32-s));
}
'''
main = r'''
int main() {
  unsigned rows,k,rank,is_b; std::cin >> rows >> k >> rank >> is_b;
  uint8_t seed[32]; int value;
  for (auto &x: seed) { std::cin >> value; x=value; }
  std::vector<uint32_t> perm(2*k);
  for (auto &x: perm) std::cin >> x;
  std::vector<int8_t> matrix(rows*k), uniform(rows*rank);
  for (auto &x: matrix) { std::cin >> value; x=value; }
  blockDim.x=1; threadIdx.x=0;
  for (blockIdx.x=0; blockIdx.x<rows*rank/32; ++blockIdx.x)
    pearl_sm75_uniform(uniform.data(), rows*rank/32, seed, is_b);
  for (blockIdx.x=0; blockIdx.x<rows*k; ++blockIdx.x)
    pearl_sm75_apply_noise(matrix.data(),rows*k,k,rank,uniform.data(),perm.data());
  for (auto x: uniform) std::cout << int(x) << ' ';
  for (auto x: matrix) std::cout << int(x) << ' ';
}
'''
cases = json.loads(subprocess.check_output(['cargo','run','--release','--locked','--quiet','-p','pearl-ref','--example','sm75_noise_vectors'],cwd=ROOT))
with tempfile.TemporaryDirectory() as temp:
    cpp, exe = Path(temp)/'noise.cpp', Path(temp)/'noise'
    cpp.write_text(preamble+helpers+'\n}\n'+kernels+main)
    subprocess.run(shlex.split(os.environ.get('CXX', 'c++'))+['-std=c++17','-O2',str(cpp),'-o',str(exe)],check=True)
    for case in cases:
        values = [case['rows'],case['k'],case['rank'],case['is_b']]+case['seed']+case['permutation']+case['raw']
        actual = list(map(int,subprocess.check_output([str(exe)], input=' '.join(map(str,values)).encode()).split()))
        assert actual == case['uniform']+case['expected'], f"noise mismatch rank={case['rank']} side={case['is_b']}"
print(f'T4 noise arithmetic: {len(cases)} official reference cases passed')
