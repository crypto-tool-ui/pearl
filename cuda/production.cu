// Dense V3 mining: one 128x128 CTA contains 128 independent 8x16 proof tiles.
// T4 merges partial proofs at each milestone; newer GPUs merge after GEMM.
// sm75: m8n8k16 + register prefetch; sm120: m16n8k32 + cp.async.
// H100 uses production_sm90.cu with shared helpers and a separate scan kernel.
// Written against the existing Pearl fragment contract; no third-party kernel code.
#define PEARL_PORTABLE_MMA 1
#include "sm75_mma.cu"

#ifndef PEARL_DIRECT_SMEM_PAD
#define PEARL_DIRECT_SMEM_PAD 0
#endif

#ifndef PEARL_HOPPER_PRODUCTION
namespace {
constexpr unsigned BM = 128, BN = 128, BK = 64;
#if __CUDA_ARCH__ == 750
constexpr unsigned THREADS = 512, ACC_ROWS = 2;
#else
constexpr unsigned THREADS = 256, ACC_ROWS = 4;
#endif
#if !defined(PEARL_DIRECT_GLOBAL)
#if __CUDA_ARCH__ == 750
constexpr unsigned STAGES = 1;
#else
constexpr unsigned STAGES = 2;
#endif
#endif
constexpr unsigned MERGE_WARPS = 8 / ACC_ROWS;
static_assert(THREADS / 32 * ACC_ROWS * 8 * 64 == BM * BN);
static_assert((BM + BN) * BK % (16 * THREADS) == 0);
#if !defined(PEARL_DIRECT_GLOBAL)
constexpr unsigned LOADS = (BM + BN) * BK / 16 / THREADS;
constexpr unsigned WORDS = (BM + BN) * BK / 4;
#endif
constexpr unsigned RECORD_WORDS = 26;
#if !defined(PEARL_DIRECT_GLOBAL)
#if __CUDA_ARCH__ == 750
static_assert(WORDS * 4 + 16 * 128 * 4 + THREADS * 4 <= 32768,
              "two CTAs must fit the Turing shared-memory budget");
#else
static_assert(THREADS * 16 <= STAGES * WORDS, "transcript merge fits staging storage");
#endif

__device__ __forceinline__ unsigned smem_word(unsigned row, unsigned kw) {
  // Spread adjacent rows across banks while retaining aligned 16-byte copies.
  return row * 16 + (kw ^ ((row & 3) * 4));
}

__device__ __forceinline__ uint32_t operand(const uint32_t* s, unsigned row, unsigned byte) {
  return s[smem_word(row, byte / 4)];
}
#endif

#if defined(PEARL_DIRECT_GLOBAL)
__device__ __forceinline__ uint32_t global_operand(
    const int8_t* matrix, unsigned row, unsigned k, unsigned ld) {
  uint32_t value;
  const int8_t* address = matrix + size_t(row) * ld + k;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(value) : "l"(address));
  return value;
}
#endif

#if !defined(PEARL_DIRECT_GLOBAL)
__device__ __forceinline__ void stage_load(
    const int8_t* a, const int8_t* b, unsigned k, unsigned row0, unsigned col0,
    unsigned kk, uint32_t* out, uint4 (&prefetch)[LOADS]) {
#pragma unroll
  for (unsigned q = 0; q < LOADS; ++q) {
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

__device__ __forceinline__ void stage_ready(uint32_t* out, const uint4 (&prefetch)[LOADS]) {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#else
#pragma unroll
  for (unsigned q = 0; q < LOADS; ++q) {
    const unsigned vector = threadIdx.x + q * THREADS;
    *reinterpret_cast<uint4*>(out + smem_word(vector / 4, (vector % 4) * 4)) = prefetch[q];
  }
#endif
  __syncthreads();
}
#endif
}

#if defined(PEARL_DIRECT_GLOBAL) && __CUDA_ARCH__ == 750
extern "C" __global__ __launch_bounds__(THREADS, 1)
#else
extern "C" __global__ __launch_bounds__(THREADS, 2)
#endif
void pearl_production_scan(const int8_t* a, const int8_t* bt,
    unsigned m, unsigned n, unsigned k, unsigned rank,
    unsigned begin, unsigned band, const uint8_t* seed, const uint32_t* bound,
    unsigned capacity, uint32_t* count, uint32_t* hits) {
#if defined(PEARL_DIRECT_GLOBAL) && __CUDA_ARCH__ >= 800
  static_assert(THREADS == 256 && MERGE_WARPS == 2,
                "direct-global proof merge assumes eight 256-thread warps");
  constexpr unsigned DIRECT_MERGE_WORDS = THREADS / MERGE_WARPS * 16 + 16 + PEARL_DIRECT_SMEM_PAD / 4;
  static_assert(DIRECT_MERGE_WORDS * sizeof(uint32_t) == 8256 + PEARL_DIRECT_SMEM_PAD,
                "direct-global transcript scratch layout changed");
  __shared__ uint32_t merge_storage[DIRECT_MERGE_WORDS];
#elif !defined(PEARL_DIRECT_GLOBAL)
  __shared__ __align__(256) uint32_t stages[STAGES][WORDS];
#endif
#if __CUDA_ARCH__ == 750
  __shared__ uint32_t proof_transcripts[16][128];
  __shared__ uint32_t folds[THREADS];
  if (threadIdx.x < 128) {
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) proof_transcripts[i][threadIdx.x] = 0;
  }
#endif
  // Host admits complete 128x128 CTAs and K/rank multiples of 64 only.
  const unsigned index = begin + blockIdx.x;
  const unsigned grid_m = m / BM, grid_n = n / BN;
  const unsigned first_row = (index / (band * grid_n)) * band;
  const unsigned band_rows = min(band, grid_m - first_row);
  const unsigned local = index - (index / (band * grid_n)) * band * grid_n;
  const unsigned row0 = (first_row + local % band_rows) * BM;
  const unsigned col0 = (local / band_rows) * BN;
  const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const unsigned wr = (warp / 2) * (ACC_ROWS * 8), wc = (warp % 2) * 64;
  const unsigned group = lane / 4, pair = lane % 4;
  const unsigned part = (warp / 2) % MERGE_WARPS;
  int32_t acc[ACC_ROWS][8][2] = {};
#if __CUDA_ARCH__ >= 800
  uint32_t transcript[16] = {};
#else
  const unsigned proof = (warp / (2 * MERGE_WARPS)) * 64 + (warp & 1) * 32 + lane;
#endif
#if !defined(PEARL_DIRECT_GLOBAL)
  uint4 prefetch[LOADS];
  stage_load(a, bt, k, row0, col0, 0, stages[0], prefetch);
  stage_ready(stages[0], prefetch);
#endif
#if defined(PEARL_DIRECT_GLOBAL)
#pragma unroll 2
#endif
  for (unsigned kk = 0; kk < k; kk += BK) {
#if !defined(PEARL_DIRECT_GLOBAL)
    const unsigned current = (kk / BK) % STAGES;
    if (kk + BK < k) stage_load(a, bt, k, row0, col0, kk + BK, stages[(current + 1) % STAGES], prefetch);
    const uint32_t* s = stages[current];
#endif
#if __CUDA_ARCH__ >= 800
#pragma unroll
    for (unsigned sk = 0; sk < BK; sk += 32) {
      uint32_t af[2][4];
#pragma unroll
      for (unsigned i = 0; i < 2; ++i) {
#if defined(PEARL_DIRECT_GLOBAL)
        const unsigned arow = row0 + wr + i * 16 + group;
        af[i][0] = global_operand(a, arow, kk + sk + pair * 4, k);
        af[i][1] = global_operand(a, arow + 8, kk + sk + pair * 4, k);
        af[i][2] = global_operand(a, arow, kk + sk + pair * 4 + 16, k);
        af[i][3] = global_operand(a, arow + 8, kk + sk + pair * 4 + 16, k);
#else
        af[i][0] = operand(s, wr + i * 16 + group, sk + pair * 4);
        af[i][1] = operand(s, wr + i * 16 + group + 8, sk + pair * 4);
        af[i][2] = operand(s, wr + i * 16 + group, sk + pair * 4 + 16);
        af[i][3] = operand(s, wr + i * 16 + group + 8, sk + pair * 4 + 16);
#endif
      }
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) {
        uint32_t bf[2];
#if defined(PEARL_DIRECT_GLOBAL)
        const unsigned brow = col0 + wc + j * 8 + group;
        bf[0] = global_operand(bt, brow, kk + sk + pair * 4, k);
        bf[1] = global_operand(bt, brow, kk + sk + pair * 4 + 16, k);
#else
        bf[0] = operand(s, BM + wc + j * 8 + group, sk + pair * 4);
        bf[1] = operand(s, BM + wc + j * 8 + group, sk + pair * 4 + 16);
#endif
#pragma unroll
        for (unsigned i = 0; i < 2; ++i)
          asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(acc[2*i][j][0]), "+r"(acc[2*i][j][1]),
              "+r"(acc[2*i+1][j][0]), "+r"(acc[2*i+1][j][1])
            : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]),
              "r"(bf[0]), "r"(bf[1]));
      }
    }
#else
#pragma unroll
    for (unsigned sk = 0; sk < BK; sk += 16) {
      uint32_t af[ACC_ROWS];
#pragma unroll
      for (unsigned i = 0; i < ACC_ROWS; ++i) {
#if defined(PEARL_DIRECT_GLOBAL)
        af[i] = global_operand(a, row0 + wr + i * 8 + group, kk + sk + pair * 4, k);
#else
        af[i] = operand(s, wr + i * 8 + group, sk + pair * 4);
#endif
      }
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) {
#if defined(PEARL_DIRECT_GLOBAL)
        const uint32_t bf = global_operand(bt, col0 + wc + j * 8 + group,
                                           kk + sk + pair * 4, k);
#else
        const uint32_t bf = operand(s, BM + wc + j * 8 + group, sk + pair * 4);
#endif
#pragma unroll
        for (unsigned i = 0; i < ACC_ROWS; ++i) mma_int8(acc[i][j], af[i], bf);
      }
    }
#endif
    if ((kk + BK) % rank == 0) {
      uint32_t fold[8] = {};
#pragma unroll
      for (unsigned j = 0; j < 8; ++j)
#pragma unroll
        for (unsigned i = 0; i < ACC_ROWS; ++i) fold[j] ^= uint32_t(acc[i][j][0]) ^ uint32_t(acc[i][j][1]);
      uint32_t x = 0;
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) x ^= fold[j];
#if __CUDA_ARCH__ == 750
      // Merge each milestone so only 128 full transcripts occupy shared memory.
      folds[threadIdx.x] = x;
      __syncthreads();
      if (part == 0) {
#pragma unroll
        for (unsigned p = 1; p < MERGE_WARPS; ++p) x ^= folds[threadIdx.x + p * 64];
        const unsigned slot = ((kk + BK) / rank - 1) & 15;
        const uint32_t prev = proof_transcripts[slot][proof];
        proof_transcripts[slot][proof] = (prev << 13 | prev >> 19) ^ x;
      }
#else
      // Rotate the register queue to avoid dynamically indexed local-memory arrays.
      const uint32_t head = (transcript[0] << 13 | transcript[0] >> 19) ^ x;
#pragma unroll
      for (unsigned i = 0; i < 15; ++i) transcript[i] = transcript[i + 1];
      transcript[15] = head;
#endif
    }
#if defined(PEARL_DIRECT_GLOBAL) && __CUDA_ARCH__ == 750
    __syncthreads(); // protect the Turing reduction scratch before the next rank slice
#elif !defined(PEARL_DIRECT_GLOBAL)
    __syncthreads(); // all readers finished before either stage can be reused
    if (kk + BK < k) stage_ready(stages[(current + 1) % STAGES], prefetch);
#endif
  }
#if __CUDA_ARCH__ == 750
  if (threadIdx.x == 0) atomicAdd(count + 1, 128u);
  if (part != 0) return;
  uint32_t transcript[16];
#pragma unroll
  for (unsigned i = 0; i < 16; ++i) transcript[i] = proof_transcripts[i][proof];
#else
  // Production rank=128/K=4096 has no residual rotation; tests also cover shorter K.
  for (unsigned r = (k / rank) & 15; r != 0; --r) {
    const uint32_t last = transcript[15];
#pragma unroll
    for (int i = 15; i > 0; --i) transcript[i] = transcript[i - 1];
    transcript[0] = last;
  }
  // XOR/rotation are linear. Merge two partial proofs after GEMM.
  // Input staging is dead here; reuse its allocation for transcript merging.
#if defined(PEARL_DIRECT_GLOBAL)
  uint32_t* merged = merge_storage;
#else
  uint32_t* merged = reinterpret_cast<uint32_t*>(stages);
#endif
  if (part != 0) {
#pragma unroll
    for (unsigned i = 0; i < 16; ++i)
#if defined(PEARL_DIRECT_GLOBAL)
      merged[(threadIdx.x / (MERGE_WARPS * 64) * 64 + threadIdx.x % 64) * 16 + i] = transcript[i];
#else
      merged[threadIdx.x * 16 + i] = transcript[i];
#endif
  }
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(count + 1, 128u);
  if (part != 0) return;
#pragma unroll
  for (unsigned p = 1; p < MERGE_WARPS; ++p)
#pragma unroll
    for (unsigned i = 0; i < 16; ++i)
#if defined(PEARL_DIRECT_GLOBAL)
      transcript[i] ^= merged[(threadIdx.x / (MERGE_WARPS * 64) * 64 + threadIdx.x % 64) * 16 + i];
#else
      transcript[i] ^= merged[(threadIdx.x + p * 64) * 16 + i];
#endif
#endif
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

#endif // PEARL_HOPPER_PRODUCTION

// Exact stateless form of pearl_ref::fill_int7: one SplitMix word per thread.
extern "C" __global__ void pearl_generate_matrix(
    uint64_t seed, uint64_t domain, unsigned words, uint64_t* out) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= words) return;
  uint64_t x = (seed ^ domain) + (uint64_t(i) + 1) * 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x ^= x >> 31;
  // Eight signed INT7 bytes at once: bit 6 of x selects the sign after -64.
  const uint64_t sign = ~x & 0x4040404040404040ULL;
  out[i] = (x & 0x3f3f3f3f3f3f3f3fULL) | sign | (sign << 1);
}
