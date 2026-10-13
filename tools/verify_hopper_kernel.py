#!/usr/bin/env python3
"""Run the production H100 source with CPU TMA-layout/WGMMA-fragment emulation.

Checks arithmetic and descriptor layout, not GPU async timing or scheduler behavior.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

from verify_production_kernel import ROOT, helpers, preamble, main as check_main

source = (ROOT / 'cuda/production_sm90.cu').read_text()
source = source.replace('#include "production.cu"', helpers).replace('#include <cuda.h>', '')


def replace_body(name, body):
    global source
    match = re.search(r'__device__ __forceinline__ [^\n]*\b' + name + r'\(', source)
    assert match, name
    begin = source.index('{', match.end())
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    source = source[:begin] + '{\n' + body + '\n}' + source[end:]


# Preserve h_descriptor itself: WGMMA reads the descriptor's actual encoded fields.
replace_body('h_shared', 'return unsigned(static_cast<const uint8_t*>(p) - emu_stage_base) + 1024;')
replace_body('h_init', '*bar = 0;')
replace_body('h_enqueue', r'''
  if (kk == 0) emu_stage_base = stage;
  *token = ++*bar;
  h_copy(stage, a, kk, row, bar);
  h_copy(stage + H_OPERAND_BYTES, b, kk, col, bar);
''')
replace_body('h_ready', 'if (*bar != token) throw std::runtime_error("wrong TMA stage token");')
replace_body('h_copy', r'''
  if (row + 128 > map->rows || k + 64 > map->k)
    throw std::runtime_error("TMA box outside matrix");
  const size_t base = dst - emu_stage_base;
  if (base % 512) throw std::runtime_error("TMA SW64 stage alignment");
  for (unsigned r = 0; r < 128; ++r) for (unsigned c = 0; c < 64; ++c) {
    const size_t address = base + r * 64 + c;
    const size_t swizzled = address ^ (((address >> 7) & 3) << 4);
    emu_stage_base[swizzled] = uint8_t(map->data[(row + r) * map->k + k + c]);
  }
''')
replace_body('h_mma', r'''
  auto read = [](uint64_t desc, unsigned row, unsigned col) -> int {
    if ((desc >> 62) != 2 || ((desc >> 16) & 0x3fff) != 1 ||
        ((desc >> 32) & 0x3fff) != 32 || ((desc >> 49) & 7) != 0)
      throw std::runtime_error("wrong WGMMA K-major SW64 descriptor");
    const size_t start = (desc & 0x3fff) << 4;
    // PTX K-major canonical layout: row stride 64B, SBO=512B, Swizzle<2,4,3>.
    const size_t address = start + (row / 8) * 512 + (row % 8) * 64 + col;
    const size_t swizzled = address ^ (((address >> 7) & 3) << 4);
    return int8_t(emu_stage_base[swizzled - 1024]);
  };
  const unsigned lane = threadIdx.x % 32, warp = (threadIdx.x % 128) / 32;
  for (unsigned j = 0; j < 64; ++j) {
    const unsigned row = warp * 16 + lane / 4 + ((j % 4) / 2) * 8;
    const unsigned col = (j / 4) * 8 + (lane % 4) * 2 + j % 2;
    uint32_t sum = uint32_t(d[j]);
    for (unsigned q = 0; q < 32; ++q) sum += read(a, row, q) * read(b, col, q);
    d[j] = int32_t(sum);
  }
''')
replace_body('h_fence', '')
replace_body('h_commit_wait', '')
source, count = re.subn(r'asm volatile\("fence.proxy.async.shared::cta;" ::: "memory"\);', '', source)
assert count == 1
assert 'asm volatile' not in source
source = source.replace('void pearl_production_scan(', 'void pearl_production_scan_hopper(')
extra = r'''
#define __grid_constant__
struct CUtensorMap { const int8_t* data; unsigned rows, k; };
uint8_t* emu_stage_base;
'''
wrapper = r'''
void pearl_production_scan(const int8_t* a, const int8_t* bt,
    unsigned m, unsigned n, unsigned k, unsigned rank,
    unsigned begin, unsigned band, const uint8_t* seed, const uint32_t* bound,
    unsigned capacity, uint32_t* count, uint32_t* hits) {
  pearl_production_scan_hopper(a,bt,m,n,k,rank,begin,band,seed,bound,capacity,count,hits,
                              CUtensorMap{a,m,k}, CUtensorMap{bt,n,k});
}
'''
# Add one-stage drain (K=rank=64), rank=64, zero capacity, and inclusive target equality.
check_main = check_main.replace('{128u,2176u,4096u}', '{64u,128u,2176u,4096u}')
check_main = check_main.replace('band=4,rank=128', 'band=4,rank=k==64?64:128')
check_main = check_main.replace('if(k==128) {', r'''
    if(k==128) {
      std::copy(hits.begin()+18, hits.begin()+26, bound);
      run(128);
      bool equal=false;
      for(unsigned j=0;j<count;++j)
        equal |= std::equal(bound,bound+8,hits.data()+j*26+18);
      if(!equal) throw std::runtime_error("target equality was rejected");
      std::fill(bound,bound+8,0xffffffffu);
      std::fill(hits.begin(),hits.end(),0xdeadbeefu);
      run(0);
      if(count!=128||hits[0]!=0xdeadbeefu) throw std::runtime_error("zero capacity queue");
''')
with tempfile.TemporaryDirectory(prefix='pearl-hopper-emulator-') as temp:
    cpp, exe = Path(temp) / 'hopper.cpp', Path(temp) / 'hopper'
    cpp.write_text(preamble + extra + source + wrapper + check_main)
    subprocess.run(shlex.split(os.environ.get('CXX', 'c++')) + [
        '-std=c++20', '-O2', '-pthread', '-D__CUDA_ARCH__=900',
        '-D__CUDA_ARCH_FEAT_SM90_ALL=1', str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=240)
