#!/usr/bin/env python3
"""Execute the actual production CUDA control flow on CPU with PTX-fragment emulation.

Checks staging, lane mapping, cumulative transcripts, target filtering and queue bounds.
This is NOT a CUDA scheduler/race test or a hardware performance benchmark.
Requires C++20/pthreads. No CUDA device or third-party Python package is needed.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
helpers = (ROOT / "cuda/sm75_mma.cu").read_text().split("__device__ __forceinline__ void mma_int8")[0] + "\n}\n"
source = (ROOT / "cuda/production.cu").read_text().split("// Exact stateless form")[0].replace('#include "sm75_mma.cu"', helpers)
source, replacements = re.subn(
    r'const unsigned address = static_cast<unsigned>\(__cvta_generic_to_shared\(out \+ smem_word\(row, kw\)\)\);\s*'
    r'asm volatile\("cp.async.cg.shared.global[^;]*;"[^;]*;',
    'std::memcpy(out + smem_word(row, kw), src, 16);', source)
assert replacements == 1
source = re.sub(r'asm volatile\("cp.async.(?:commit_group|wait_group 0);" ::: "memory"\);', '', source)
source, replacements = re.subn(
    r'asm volatile\("mma.sync.aligned.m16n8k32[\s\S]*?"r"\(bf\[1\]\)\);',
    'emulate16(acc[2*i][j], acc[2*i+1][j], af[i], bf);', source)
assert replacements == 1

preamble = r'''
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
#define PEARL_PORTABLE_MMA 1
#define __device__
#define __constant__
#define __global__
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))
#define __forceinline__ inline
#define __launch_bounds__(...)
using std::min;
struct Dim { unsigned x; } blockIdx;
thread_local Dim threadIdx;
struct alignas(16) uint4 { uint32_t x,y,z,w; };
constexpr unsigned EMU_THREADS = __CUDA_ARCH__ == 750 ? 512 : 256;
std::barrier cta_barrier(EMU_THREADS);
std::array<std::unique_ptr<std::barrier<>>,EMU_THREADS/32> warp_barriers;
struct Fragment { uint32_t a[4],b[2]; } fragments[EMU_THREADS];
inline void __syncthreads() { cta_barrier.arrive_and_wait(); }
inline unsigned atomicAdd(unsigned* p, unsigned v) { return std::atomic_ref(*p).fetch_add(v); }
inline unsigned __funnelshift_r(unsigned a, unsigned b, unsigned s) { return (a>>s)|(b<<(32-s)); }
inline int byte(uint32_t x,unsigned i) { return static_cast<int8_t>(x>>(8*i)); }
inline void mma_int8(int32_t (&d)[2],uint32_t a,uint32_t b) {
  const unsigned t=threadIdx.x, lane=t%32, base=t/32*32;
  fragments[t].a[0]=a; fragments[t].b[0]=b;
  auto &bar=*warp_barriers[t/32]; bar.arrive_and_wait();
  for(unsigned v=0;v<2;++v) {
    uint32_t sum=uint32_t(d[v]);
    for(unsigned k=0;k<16;++k)
      sum+=byte(fragments[base+(lane/4)*4+k/4].a[0],k%4)*
           byte(fragments[base+(2*(lane%4)+v)*4+k/4].b[0],k%4);
    d[v]=int32_t(sum);
  }
  bar.arrive_and_wait();
}
inline void emulate16(int32_t (&lo)[2],int32_t (&hi)[2],const uint32_t (&a)[4],const uint32_t (&b)[2]) {
  const unsigned t=threadIdx.x,lane=t%32,base=t/32*32;
  std::copy(a,a+4,fragments[t].a); std::copy(b,b+2,fragments[t].b);
  auto &bar=*warp_barriers[t/32]; bar.arrive_and_wait();
  for(unsigned r=0;r<2;++r) for(unsigned v=0;v<2;++v) {
    int32_t &d=r?hi[v]:lo[v]; uint32_t sum=uint32_t(d);
    for(unsigned k=0;k<32;++k)
      sum+=byte(fragments[base+(lane/4)*4+(k%16)/4].a[r+2*(k/16)],k%4)*
           byte(fragments[base+(2*(lane%4)+v)*4+(k%16)/4].b[k/16],k%4);
    d=int32_t(sum);
  }
  bar.arrive_and_wait();
}
'''
main = r'''
int main() {
  for(auto &b:warp_barriers) b=std::make_unique<std::barrier<>>(32);
  // Two stages, residual transcript rotation, wrap at slot 16, and a nonzero
  // begin in a partial raster band, and the full 32-slice production K.
  for(unsigned k: {128u,2176u,4096u}) {
    const unsigned m=256,n=384,begin=5,band=4,rank=128;
    std::mt19937 rng(17+k); std::vector<int8_t>a(m*k),b(n*k);
    for(auto &x:a) x=int(rng()%255)-127;
    for(auto &x:b) x=int(rng()%255)-127;
    uint8_t seed[32]; for(auto &x:seed) x=rng();
    uint32_t bound[8]; std::fill(bound,bound+8,0xffffffffu);
    std::vector<uint32_t> hits(128*26+16,0xdeadbeefu); unsigned counters[2]={}; unsigned &count=counters[0];
    auto run=[&](unsigned capacity) {
      count=0; counters[1]=0; blockIdx.x=0;
      std::vector<std::thread> threads;
      for(unsigned t=0;t<EMU_THREADS;++t) threads.emplace_back([&,t] {
        threadIdx.x=t;
        pearl_production_scan(a.data(),b.data(),m,n,k,rank,begin,band,seed,bound,capacity,&count,hits.data());
      });
      for(auto &t:threads)t.join();
      if(counters[1]!=128) throw std::runtime_error("device work counter mismatch");
    };
    run(128);
    if(count!=128) throw std::runtime_error("missing hits");
    std::array<bool,128> seen={};
    for(unsigned slot=0;slot<count;++slot) {
      const auto* rec=hits.data()+slot*26;
      const unsigned r=rec[0],c=rec[1];
      if(r<128||r>=200||c<256||c>=327||c%2) throw std::runtime_error("raster coordinates");
      const unsigned warp=(r-128)/64*2+(c-256)/64;
      const unsigned lane=(r%64)*4+(c%64)/2;
      if(lane>=32||seen[warp*32+lane]) throw std::runtime_error("duplicate/wrong lane");
      seen[warp*32+lane]=true;
      int32_t acc[8][16]={}; uint32_t tr[16]={};
      for(unsigned s=0;s<k/rank;++s) {
        uint32_t fold=0;
        for(unsigned i=0;i<8;++i)for(unsigned j=0;j<16;++j) {
          for(unsigned q=s*rank;q<(s+1)*rank;++q)
            acc[i][j]+=int(a[(r+i*8)*k+q])*int(b[(c+(j/2)*8+j%2)*k+q]);
          fold^=uint32_t(acc[i][j]);
        }
        tr[s%16]=(tr[s%16]<<13|tr[s%16]>>19)^fold;
      }
      if(!std::equal(tr,tr+16,rec+2)) throw std::runtime_error("scalar transcript mismatch");
      uint8_t digest[32]; blake3_jackpot_hash(tr,seed,digest);
      if(std::memcmp(digest,rec+18,32)) throw std::runtime_error("digest mismatch");
    }
    if(hits[128*26]!=0xdeadbeefu) throw std::runtime_error("queue overrun");
    if(k==128) {
      std::fill(hits.begin(),hits.end(),0xdeadbeefu);
      run(3);
      if(count!=128||hits[3*26]!=0xdeadbeefu) throw std::runtime_error("bounded queue failure");
      std::fill(bound,bound+8,0u); run(128);
      if(count!=0) throw std::runtime_error("target filter failure");
    }
    std::cout<<"source emulator SM="<<__CUDA_ARCH__<<" K="<<k<<": 128 scalar transcripts matched\n"<<std::flush;
  }
}
'''
def run_checks():
    with tempfile.TemporaryDirectory(prefix="pearl-production-emulator-") as temp:
        variants = [("staged", source, "")]
        direct = source.replace(
            'asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(value) : "l"(address));',
            'value = *reinterpret_cast<const uint32_t*>(address);')
        assert direct != source
        direct_define = "#define PEARL_DIRECT_GLOBAL 1\n"
        variants.append(("direct", direct, direct_define))
        variants.append(("direct-padded", direct,
                         direct_define + "#define PEARL_DIRECT_SMEM_PAD 1024\n"))
        for name, kernel, defines in variants:
            cpp = Path(temp) / f"kernel-{name}.cpp"
            cpp.write_text(defines + preamble + kernel + main)
            arches = (750, 1200) if name in ("staged", "direct") else (1200,)
            for arch in arches:  # H100 production is checked by verify_hopper_kernel.py.
                exe = Path(temp) / f"kernel-{name}-{arch}"
                subprocess.run(shlex.split(os.environ.get("CXX", "c++")) + [
                    "-std=c++20", "-O2", "-pthread", f"-D__CUDA_ARCH__={arch}", str(cpp), "-lstdc++", "-o", str(exe)], check=True)
                subprocess.run([str(exe)], check=True, timeout=240)


if __name__ == "__main__":
    run_checks()
