// Dense V3 mining: one 128x128 CTA contains 128 independent 8x16 proof tiles.
// Each lane owns its complete cumulative transcript. No cross-lane reduction.
// sm75: m8n8k16 and register-prefetched loads; sm90/sm120: m16n8k32 + cp.async.
// Written against the existing Pearl fragment contract; no third-party kernel code.
#define PEARL_PORTABLE_MMA 1
#include "sm75_mma.cu"

namespace {
constexpr unsigned BM = 128, BN = 128, BK = 64, THREADS = 128;
constexpr unsigned WORDS = (BM + BN) * BK / 4;
constexpr unsigned RECORD_WORDS = 26;

__device__ __forceinline__ unsigned smem_word(unsigned row, unsigned kw) {
  // Spread adjacent rows across banks while retaining aligned 16-byte copies.
  return row * 16 + (kw ^ ((row & 3) * 4));
}

__device__ __forceinline__ uint32_t operand(const uint32_t* s, unsigned row, unsigned byte) {
  return s[smem_word(row, byte / 4)];
}

__device__ __forceinline__ void stage_load(
    const int8_t* a, const int8_t* b, unsigned k, unsigned row0, unsigned col0,
    unsigned kk, uint32_t* out, uint4 (&prefetch)[8]) {
#pragma unroll
  for (unsigned q = 0; q < 8; ++q) {
    const unsigned vector = threadIdx.x + q * THREADS;
    const unsigned row = vector / 4, kw = (vector % 4) * 4;
    const int8_t* src = row < BM ? a + size_t(row0 + row) * k + kk + kw * 4
                                : b + size_t(col0 + row - BM) * k + kk + kw * 4;
#if __CUDA_ARCH__ >= 800
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(out + smem_word(row, kw)));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address), "l"(src) : "memory");
#else
    prefetch[q] = *reinterpret_cast<const uint4*>(src);
#endif
  }
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void stage_ready(uint32_t* out, const uint4 (&prefetch)[8]) {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#else
#pragma unroll
  for (unsigned q = 0; q < 8; ++q) {
    const unsigned vector = threadIdx.x + q * THREADS;
    *reinterpret_cast<uint4*>(out + smem_word(vector / 4, (vector % 4) * 4)) = prefetch[q];
  }
#endif
  __syncthreads();
}
}

extern "C" __global__ __launch_bounds__(128, 1)
void pearl_production_scan(const int8_t* a, const int8_t* bt,
    unsigned m, unsigned n, unsigned k, unsigned rank,
    unsigned begin, unsigned band, const uint8_t* seed, const uint32_t* bound,
    unsigned capacity, uint32_t* count, uint32_t* hits) {
  __shared__ __align__(16) uint32_t stages[2][WORDS];
  // Host admits complete 128x128 CTAs and K/rank multiples of 64 only.
  const unsigned index = begin + blockIdx.x;
  const unsigned grid_m = m / BM, grid_n = n / BN;
  const unsigned first_row = (index / (band * grid_n)) * band;
  const unsigned band_rows = min(band, grid_m - first_row);
  const unsigned local = index - (index / (band * grid_n)) * band * grid_n;
  const unsigned row0 = (first_row + local % band_rows) * BM;
  const unsigned col0 = (local / band_rows) * BN;
  const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const unsigned wr = (warp / 2) * 64, wc = (warp % 2) * 64;
  const unsigned group = lane / 4, pair = lane % 4;
  int32_t acc[8][8][2] = {};
  uint32_t transcript[16] = {};
  uint4 prefetch[8];
  stage_load(a, bt, k, row0, col0, 0, stages[0], prefetch);
  stage_ready(stages[0], prefetch);
  for (unsigned kk = 0; kk < k; kk += BK) {
    const unsigned current = (kk / BK) & 1;
    if (kk + BK < k) stage_load(a, bt, k, row0, col0, kk + BK, stages[current ^ 1], prefetch);
    const uint32_t* s = stages[current];
#if __CUDA_ARCH__ >= 800
#pragma unroll
    for (unsigned sk = 0; sk < BK; sk += 32) {
      uint32_t af[4][4], bf[8][2];
#pragma unroll
      for (unsigned i = 0; i < 4; ++i) {
        af[i][0] = operand(s, wr + i * 16 + group, sk + pair * 4);
        af[i][1] = operand(s, wr + i * 16 + group + 8, sk + pair * 4);
        af[i][2] = operand(s, wr + i * 16 + group, sk + pair * 4 + 16);
        af[i][3] = operand(s, wr + i * 16 + group + 8, sk + pair * 4 + 16);
      }
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) {
        bf[j][0] = operand(s, BM + wc + j * 8 + group, sk + pair * 4);
        bf[j][1] = operand(s, BM + wc + j * 8 + group, sk + pair * 4 + 16);
      }
#pragma unroll
      for (unsigned i = 0; i < 4; ++i)
#pragma unroll
        for (unsigned j = 0; j < 8; ++j)
          asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(acc[2*i][j][0]), "+r"(acc[2*i][j][1]),
              "+r"(acc[2*i+1][j][0]), "+r"(acc[2*i+1][j][1])
            : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]),
              "r"(bf[j][0]), "r"(bf[j][1]));
    }
#else
#pragma unroll
    for (unsigned sk = 0; sk < BK; sk += 16) {
      uint32_t af[8], bf[8];
#pragma unroll
      for (unsigned i = 0; i < 8; ++i) {
        af[i] = operand(s, wr + i * 8 + group, sk + pair * 4);
        bf[i] = operand(s, BM + wc + i * 8 + group, sk + pair * 4);
      }
#pragma unroll
      for (unsigned i = 0; i < 8; ++i)
#pragma unroll
        for (unsigned j = 0; j < 8; ++j) mma_int8(acc[i][j], af[i], bf[j]);
    }
#endif
    if ((kk + BK) % rank == 0) {
      uint32_t fold[8] = {};
#pragma unroll
      for (unsigned j = 0; j < 8; ++j)
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) fold[j] ^= uint32_t(acc[i][j][0]) ^ uint32_t(acc[i][j][1]);
      uint32_t x = 0;
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) x ^= fold[j];
      // Rotate the register queue to avoid dynamically indexed local-memory arrays.
      const uint32_t head = (transcript[0] << 13 | transcript[0] >> 19) ^ x;
#pragma unroll
      for (unsigned i = 0; i < 15; ++i) transcript[i] = transcript[i + 1];
      transcript[15] = head;
    }
    __syncthreads(); // all readers finished before either stage can be reused
    if (kk + BK < k) stage_ready(stages[current ^ 1], prefetch);
  }
  // Production rank=128/K=4096 has no residual rotation; tests also cover shorter K.
  for (unsigned r = (k / rank) & 15; r != 0; --r) {
    const uint32_t last = transcript[15];
#pragma unroll
    for (int i = 15; i > 0; --i) transcript[i] = transcript[i - 1];
    transcript[0] = last;
  }
  uint8_t digest[32];
  blake3_jackpot_hash(transcript, seed, digest);
  bool wins = true;
#pragma unroll
  for (int i = 7; i >= 0; --i) {
    const uint32_t word = load_u32_le(digest + i * 4);
    if (word != bound[i]) { wins = word < bound[i]; break; }
  }
  if (wins) {
    const unsigned slot = atomicAdd(count, 1u);
    if (slot < capacity) {
      uint32_t* dst = hits + size_t(slot) * RECORD_WORDS;
      dst[0] = row0 + wr + group;
      dst[1] = col0 + wc + pair * 2;
#pragma unroll
      for (unsigned i = 0; i < 16; ++i) dst[2+i] = transcript[i];
#pragma unroll
      for (unsigned i = 0; i < 8; ++i) dst[18+i] = load_u32_le(digest + i*4);
    }
  }
}
