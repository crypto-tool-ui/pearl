// Blackwell V3 scan: signed INT8 MMA fed by a TMA/mbarrier producer-consumer pipeline.
// The 128x128 CTA geometry and 8x16 proof transcript remain the root miner's contract.
#define PEARL_PORTABLE_MMA 1
#include "sm75_mma.cu"
#include <cuda.h>
#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1200
#error "production_sm120_tma.cu requires sm_120"
#endif

namespace {
constexpr unsigned BM = 128, BN = 128, BK = 64;
constexpr unsigned MMA_WARPS = 4, THREADS = 256;
constexpr unsigned STAGES = 4;
constexpr unsigned A_BYTES = BM * BK, B_BYTES = BN * BK;
constexpr unsigned STAGE_BYTES = A_BYTES + B_BYTES;
constexpr unsigned BARRIERS_BYTES = STAGES * 16;
constexpr unsigned COMMAND_BYTES = STAGES * 4;
constexpr unsigned SMEM_BYTES = STAGES * STAGE_BYTES + BARRIERS_BYTES + COMMAND_BYTES;
static_assert(SMEM_BYTES < 99 * 1024, "TMA pipeline exceeds Blackwell shared-memory budget");

__device__ __forceinline__ unsigned shmem(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void bar_init(unsigned bar, unsigned arrivals) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(arrivals) : "memory");
}
__device__ __forceinline__ void bar_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void bar_arrive(unsigned bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void bar_expect_tx(unsigned bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bar_wait(unsigned bar, unsigned phase) {
  asm volatile("{ .reg .pred done; L_%=: mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1; @!done bra L_%=; }"
      :: "r"(bar), "r"(phase) : "memory");
}
__device__ __forceinline__ void tma_load(unsigned dst, const CUtensorMap* map,
                                         unsigned k, unsigned row, unsigned bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      :: "r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(k), "r"(row), "r"(bar) : "memory");
}
__device__ __forceinline__ void ldmatrix4(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
}
__device__ __forceinline__ void mma_s8(int32_t (&d)[4], const uint32_t (&a)[4],
                                       uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t rotl13(uint32_t x) { return (x << 13) | (x >> 19); }
template <unsigned BY>
__device__ __forceinline__ void rotate_right(uint32_t (&t)[16]) {
  uint32_t shifted[16];
#pragma unroll
  for (unsigned i = 0; i < 16; ++i) shifted[i] = t[(i + 16 - BY) & 15];
#pragma unroll
  for (unsigned i = 0; i < 16; ++i) t[i] = shifted[i];
}
} // namespace

extern "C" __global__ __launch_bounds__(THREADS, 1)
void pearl_production_scan_sm120_tma(
    const int8_t* a, const int8_t* bt, unsigned m, unsigned n, unsigned k, unsigned rank,
    unsigned begin, unsigned band, const uint8_t* seed, const uint32_t* bound,
    unsigned capacity, uint32_t* count, uint32_t* hits,
    const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_b) {
  extern __shared__ __align__(1024) uint8_t raw[];
  const unsigned raw_addr = shmem(raw);
  uint8_t* stages = raw + (((raw_addr + 1023u) & ~1023u) - raw_addr);
  const unsigned barriers = shmem(stages + STAGES * STAGE_BYTES);
  const unsigned empty = barriers + STAGES * 8;
  const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned index = begin + blockIdx.x;
  const unsigned grid_n = n / BN;
  const unsigned first_row = (index / (band * grid_n)) * band;
  const unsigned band_rows = min(band, m / BM - first_row);
  const unsigned local = index - (index / (band * grid_n)) * band * grid_n;
  const unsigned row0 = (first_row + local % band_rows) * BM;
  const unsigned col0 = (local / band_rows) * BN;

  if (threadIdx.x == 0) {
#pragma unroll
    for (unsigned s = 0; s < STAGES; ++s) {
      bar_init(barriers + s * 8, 1);
      bar_init(empty + s * 8, MMA_WARPS);
    }
    bar_fence();
  }
  __syncthreads();

  // The producer warpgroup fills the ring while four MMA warps consume it.
  if (warp >= MMA_WARPS) {
    if (warp == MMA_WARPS && lane == 0) {
      unsigned stage = 0, phase = 0;
      for (unsigned kk = 0; kk < k; kk += BK) {
        bar_wait(empty + stage * 8, phase ^ 1u);
        const unsigned full = barriers + stage * 8;
        const unsigned dst = shmem(stages + stage * STAGE_BYTES);
        bar_expect_tx(full, STAGE_BYTES);
        tma_load(dst, &map_a, kk, row0, full);
        tma_load(dst + A_BYTES, &map_b, kk, col0, full);
        if (++stage == STAGES) { stage = 0; phase ^= 1u; }
      }
    }
    return;
  }

  // Each warp owns a 64x64 tile. A lane owns one 8x16 proof's 128 accumulators.
  const unsigned wm = warp >> 1, wn = warp & 1u;
  const unsigned swz = (lane & 7u) >> 1;
  const unsigned arow = wm * 64 + (lane & 7u) + ((lane >> 3) & 1u) * 8;
  const unsigned brow = wn * 64 + (lane & 7u) + (lane >> 4) * 8;
  const unsigned aoff[2] = { arow * BK + ((((lane >> 4) + 0u) ^ swz) << 4),
                              arow * BK + ((((lane >> 4) + 2u) ^ swz) << 4) };
  const unsigned boff[2] = { A_BYTES + brow * BK + (((((lane >> 3) & 1u) + 0u) ^ swz) << 4),
                             A_BYTES + brow * BK + (((((lane >> 3) & 1u) + 2u) ^ swz) << 4) };
  int32_t acc[4][8][4] = {};
  uint32_t transcript[16] = {};
  unsigned stage = 0, phase = 0;
  for (unsigned kk = 0; kk < k; kk += BK) {
    bar_wait(barriers + stage * 8, phase);
    const unsigned sb = shmem(stages + stage * STAGE_BYTES);
#pragma unroll
    for (unsigned ks = 0; ks < 2; ++ks) {
      uint32_t af[4][4], bf[8][2];
#pragma unroll
      for (unsigned i = 0; i < 4; ++i)
        ldmatrix4(af[i][0], af[i][1], af[i][2], af[i][3], sb + aoff[ks] + i * 16 * BK);
#pragma unroll
      for (unsigned jp = 0; jp < 4; ++jp)
        ldmatrix4(bf[2 * jp][0], bf[2 * jp][1], bf[2 * jp + 1][0], bf[2 * jp + 1][1],
                  sb + boff[ks] + jp * 16 * BK);
#pragma unroll
      for (unsigned i = 0; i < 4; ++i)
#pragma unroll
        for (unsigned j = 0; j < 8; ++j)
          mma_s8(acc[i][j], af[i], bf[j][0], bf[j][1]);
    }
    __syncwarp();
    if (lane == 0) bar_arrive(empty + stage * 8);
    if (++stage == STAGES) { stage = 0; phase ^= 1u; }
    if ((kk + BK) % rank == 0) {
      uint32_t fold = 0;
#pragma unroll
      for (unsigned i = 0; i < 4; ++i)
#pragma unroll
        for (unsigned j = 0; j < 8; ++j)
#pragma unroll
          for (unsigned v = 0; v < 4; ++v) fold ^= static_cast<uint32_t>(acc[i][j][v]);
      const uint32_t head = rotl13(transcript[0]) ^ fold;
#pragma unroll
      for (unsigned i = 0; i < 15; ++i) transcript[i] = transcript[i + 1];
      transcript[15] = head;
    }
  }
  const unsigned rotation = (k / rank) & 15u;
  if (rotation & 1u) rotate_right<1>(transcript);
  if (rotation & 2u) rotate_right<2>(transcript);
  if (rotation & 4u) rotate_right<4>(transcript);
  if (rotation & 8u) rotate_right<8>(transcript);
  if (warp == 0 && lane == 0) atomicAdd(count + 1, 128u);
  const unsigned t_rows = row0 + wm * 64 + (lane >> 2);
  const unsigned t_cols = col0 + wn * 64 + (lane & 3u) * 2;
  uint8_t digest[32];
  blake3_jackpot_hash(transcript, seed, digest);
  bool wins = true;
#pragma unroll
  for (int j = 7; j >= 0; --j) {
    const uint32_t word = load_u32_le(digest + j * 4);
    if (word != bound[j]) { wins = word < bound[j]; break; }
  }
  if (wins) {
    const unsigned hit = atomicAdd(count, 1u);
    if (hit < capacity) {
      uint32_t* dst = hits + size_t(hit) * 26;
      dst[0] = t_rows; dst[1] = t_cols;
#pragma unroll
      for (unsigned j = 0; j < 16; ++j) dst[2 + j] = transcript[j];
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) dst[18 + j] = load_u32_le(digest + j * 4);
    }
  }
}

// Same device-side SplitMix generator as production.cu; kept in this cubin because the
// production adapter loads preparation kernels from the scan module too.
extern "C" __global__ void pearl_generate_matrix(
    uint64_t seed, uint64_t domain, unsigned words, uint64_t* out) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= words) return;
  uint64_t x = (seed ^ domain) + (uint64_t(i) + 1) * 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x ^= x >> 31;
  const uint64_t sign = ~x & 0x4040404040404040ULL;
  out[i] = (x & 0x3f3f3f3f3f3f3f3fULL) | sign | (sign << 1);
}
