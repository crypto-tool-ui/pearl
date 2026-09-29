#!/usr/bin/env python3
"""Execute CUDA generation/hash source on CPU against Rust matrix/commitment vectors.
This checks arithmetic, not device scheduling or hardware performance.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
ROOT = Path(__file__).resolve().parents[1]
generator = (ROOT / 'cuda/production.cu').read_text().split('// Exact stateless form')[1]
generator = generator[generator.index('extern "C"'):]
hasher = (ROOT / 'cuda/tensor_hash.cu').read_text()
preamble = r'''
#include <cstdint>
#include <iostream>
#include <vector>
#define __device__
#define __constant__
#define __global__
#define __forceinline__ inline
struct Dim { unsigned x; } blockIdx, blockDim{1}, threadIdx{0};
'''
main = r'''
int main() {
  uint64_t seed, domain; unsigned len;
  std::cin >> seed >> domain >> len;
  uint8_t key[32]; unsigned v;
  for(auto &x:key) { std::cin >> v; x=v; }
  std::vector<uint64_t> data(len/8 + 1, 0xdeadbeefULL);
  for(blockIdx.x=0;blockIdx.x<=len/8;++blockIdx.x)
    pearl_generate_matrix(seed, domain, len/8, data.data());
  if(data.back()!=0xdeadbeefULL) return 2;
  const auto *bytes=reinterpret_cast<const uint8_t*>(data.data());
  for(unsigned i=0;i<len;++i) std::cout<<unsigned(bytes[i])<<' ';
  unsigned count=len/1024;
  std::vector<uint8_t> level(count*32);
  for(blockIdx.x=0;blockIdx.x<count;++blockIdx.x)
    pearl_tensor_hash_leaf_cv(bytes,count,key,count==1,level.data());
  while(count>1) {
    std::vector<uint8_t> next(((count+1)/2)*32);
    for(blockIdx.x=0;blockIdx.x<(count+1)/2;++blockIdx.x)
      pearl_tensor_hash_parent_cv(level.data(),count,key,count==2,next.data());
    count=(count+1)/2; level=std::move(next);
  }
  for(auto x:level) std::cout<<unsigned(x)<<' ';
}
'''
cases = json.loads(subprocess.check_output(['cargo','run','--release','--locked','--quiet','-p','pearl-ref','--example','device_setup_vectors'],cwd=ROOT))
with tempfile.TemporaryDirectory(prefix='pearl-device-setup-') as temp:
    cpp, exe = Path(temp)/'setup.cpp', Path(temp)/'setup'
    cpp.write_text(preamble+generator+hasher+main)
    subprocess.run(shlex.split(os.environ.get('CXX','c++'))+['-std=c++17','-O2',str(cpp),'-o',str(exe)],check=True)
    for case in cases:
        values=[case['seed'],case['domain'],len(case['bytes'])]+case['key']
        actual=list(map(int,subprocess.check_output([str(exe)],input=' '.join(map(str,values)).encode()).split()))
        assert actual==case['bytes']+case['root'], f"matrix/root mismatch: seed={case['seed']} bytes={len(case['bytes'])}"
print(f'Device generation and Merkle roots: {len(cases)} Rust reference cases passed')
