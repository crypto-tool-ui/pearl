// H100 production: 128x128x64 CTA, two 64x128 INT8 WGMMA warpgroups.
#define PEARL_HOPPER_PRODUCTION 1
#include "production.cu"
#include <cuda.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ != 900 || !defined(__CUDA_ARCH_FEAT_SM90_ALL))
#error "production_sm90.cu requires sm_90a"
#endif

namespace {
constexpr unsigned H_THREADS = 256, H_BK = 64;
constexpr unsigned H_OPERAND_BYTES = 128 * H_BK;
constexpr unsigned H_STAGE_BYTES = 2 * H_OPERAND_BYTES;
static_assert(2 * H_STAGE_BYTES + (16 * 128 + 2 * H_THREADS) * sizeof(uint32_t) +
              4 * sizeof(uint64_t) <= 48 * 1024, "static shared memory needs no opt-in");

__device__ __forceinline__ unsigned h_shared(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint64_t h_descriptor(const uint8_t* p) {
  // K-major SW64: 64 bytes/row, 512 bytes between groups of eight rows.
  // Each operand/stage starts on a 512B boundary, so the swizzle base is zero.
  return ((uint64_t(h_shared(p)) >> 4) & 0x3fffULL) |
      (1ULL << 16) | (32ULL << 32) | (2ULL << 62);
}

__device__ __forceinline__ void h_init(uint64_t* bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(h_shared(bar)) : "memory");
}

__device__ __forceinline__ void h_copy(
    uint8_t* dst, const CUtensorMap* map, unsigned k, unsigned row, uint64_t* bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];" ::
      "r"(h_shared(dst)), "l"(map), "r"(k), "r"(row), "r"(h_shared(bar)) : "memory");
}

__device__ __forceinline__ void h_enqueue(
    uint8_t* stage, uint64_t* bar, uint64_t* token,
    const CUtensorMap* a, const CUtensorMap* b, unsigned kk, unsigned row, unsigned col) {
  uint64_t state;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;" : "=l"(state) :
      "r"(h_shared(bar)), "r"(H_STAGE_BYTES) : "memory");
  *token = state;
  h_copy(stage, a, kk, row, bar);
  h_copy(stage + H_OPERAND_BYTES, b, kk, col, bar);
}

__device__ __forceinline__ void h_ready(uint64_t* bar, uint64_t token) {
  unsigned done;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.acquire.cta.shared::cta.b64 p, [%1], %2; "
        "selp.b32 %0, 1, 0, p; }" : "=r"(done) : "r"(h_shared(bar)), "l"(token) : "memory");
  } while (!done);
}

__device__ __forceinline__ void h_mma(int32_t (&d)[64], uint64_t a, uint64_t b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, 1;"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
        "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
        "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
        "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]),
        "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]),
        "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]),
        "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]),
        "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63])
      : "l"(a), "l"(b) : "memory");
}

// Operand constraints also stop the compiler moving accumulator accesses across
// the async fence/wait; a memory clobber alone does not constrain register arithmetic.
__device__ __forceinline__ void h_fence(int32_t (&d)[64]) {
  asm volatile("wgmma.fence.sync.aligned;"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
        "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
        "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
        "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]),
        "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]),
        "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]),
        "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]),
        "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63]) :: "memory");
}
__device__ __forceinline__ void h_commit_wait(int32_t (&d)[64]) {
  asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
        "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]),
        "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]),
        "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31]),
        "+r"(d[32]), "+r"(d[33]), "+r"(d[34]), "+r"(d[35]), "+r"(d[36]), "+r"(d[37]), "+r"(d[38]), "+r"(d[39]),
        "+r"(d[40]), "+r"(d[41]), "+r"(d[42]), "+r"(d[43]), "+r"(d[44]), "+r"(d[45]), "+r"(d[46]), "+r"(d[47]),
        "+r"(d[48]), "+r"(d[49]), "+r"(d[50]), "+r"(d[51]), "+r"(d[52]), "+r"(d[53]), "+r"(d[54]), "+r"(d[55]),
        "+r"(d[56]), "+r"(d[57]), "+r"(d[58]), "+r"(d[59]), "+r"(d[60]), "+r"(d[61]), "+r"(d[62]), "+r"(d[63]) :: "memory");
}
} // namespace

extern "C" __global__ __launch_bounds__(H_THREADS, 2)
void pearl_production_scan(const int8_t* a, const int8_t* bt,
    unsigned m, unsigned n, unsigned k, unsigned rank,
    unsigned begin, unsigned band, const uint8_t* seed, const uint32_t* bound,
    unsigned capacity, uint32_t* count, uint32_t* hits,
    const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_b) {
  __shared__ __align__(1024) uint8_t stages[2][H_STAGE_BYTES];
  __shared__ uint64_t barriers[2], tokens[2];
  __shared__ uint32_t transcripts[16][128], folds[2][H_THREADS];
  if (threadIdx.x < 128) {
#pragma unroll
    for (unsigned j = 0; j < 16; ++j) transcripts[j][threadIdx.x] = 0;
  }
  const unsigned tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wg = tid >> 7;
  const unsigned index = begin + blockIdx.x;
  const unsigned grid_m = m / 128, grid_n = n / 128;
  const unsigned first_row = (index / (band * grid_n)) * band;
  const unsigned band_rows = min(band, grid_m - first_row);
  const unsigned local = index - (index / (band * grid_n)) * band * grid_n;
  const unsigned row0 = (first_row + local % band_rows) * 128;
  const unsigned col0 = (local / band_rows) * 128;
  if (tid == 0) {
    h_init(&barriers[0]);
    h_init(&barriers[1]);
    // Publish barrier initialization to the TMA async proxy before issuing copies.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    h_enqueue(stages[0], &barriers[0], &tokens[0], &map_a, &map_b, 0, row0, col0);
    if (k > H_BK)
      h_enqueue(stages[1], &barriers[1], &tokens[1], &map_a, &map_b, H_BK, row0, col0);
  }
  __syncthreads();
  int32_t d[64] = {};
  for (unsigned kk = 0; kk < k; kk += H_BK) {
    const unsigned slot = (kk / H_BK) & 1;
    h_ready(&barriers[slot], tokens[slot]);
    const uint8_t* as = stages[slot] + wg * 64 * H_BK;
    const uint8_t* bs = stages[slot] + H_OPERAND_BYTES;
    h_fence(d);
    h_mma(d, h_descriptor(as), h_descriptor(bs));
    h_mma(d, h_descriptor(as + 32), h_descriptor(bs + 32));
    h_commit_wait(d);
    if ((kk + H_BK) % rank == 0) {
#pragma unroll
      for (unsigned half = 0; half < 2; ++half) {
        uint32_t fold = 0;
#pragma unroll
        for (unsigned j = 0; j < 32; ++j) fold ^= uint32_t(d[half * 32 + j]);
        folds[half][tid] = fold;
      }
      __syncthreads();
      if (warp % 4 == 0) {
#pragma unroll
        for (unsigned half = 0; half < 2; ++half) {
          uint32_t fold = 0;
#pragma unroll
          for (unsigned w = 0; w < 4; ++w) fold ^= folds[half][tid + w * 32];
          const unsigned proof = wg * 64 + half * 32 + lane;
          const unsigned slot = ((kk + H_BK) / rank - 1) & 15;
          const uint32_t prev = transcripts[slot][proof];
          transcripts[slot][proof] = (prev << 13 | prev >> 19) ^ fold;
        }
      }
    }
    // Every warpgroup must finish WGMMA reads before TMA may overwrite this stage.
    __syncthreads();
    if (tid == 0 && kk + 2 * H_BK < k)
      h_enqueue(stages[slot], &barriers[slot], &tokens[slot], &map_a, &map_b,
                kk + 2 * H_BK, row0, col0);
    __syncthreads(); // publish the next barrier token to every consumer
  }
  if (tid == 0) atomicAdd(count + 1, 128u);
  if (warp % 4 != 0) return;
#pragma unroll
  for (unsigned half = 0; half < 2; ++half) {
    uint32_t tr[16];
#pragma unroll
    for (unsigned j = 0; j < 16; ++j) tr[j] = transcripts[j][wg * 64 + half * 32 + lane];
    uint8_t digest[32];
    blake3_jackpot_hash(tr, seed, digest);
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
        dst[0] = row0 + wg * 64 + lane / 4;
        dst[1] = col0 + half * 64 + (lane % 4) * 2;
#pragma unroll
        for (unsigned j = 0; j < 16; ++j) dst[2 + j] = tr[j];
#pragma unroll
        for (unsigned j = 0; j < 8; ++j) dst[18 + j] = load_u32_le(digest + j * 4);
      }
    }
  }
}
