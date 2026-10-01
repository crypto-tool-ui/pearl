// M4: V3/int7 Pearl tile GEMM for sm_90a.
//
// One CTA contains one producer warp followed by one or more 128-thread
// warpgroups. Each consumer warpgroup owns a 64x8 output tile and uses WGMMA
// with signed int8 operands and int32 accumulation. The producer warp issues
// TMA bulk copies into two shared-memory stages; the stage layout is the
// K-major interleave layout described by the Hopper PTX ISA.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 900
#error "sm90_wgmma.cu must be compiled for sm_90a"
#endif

#include <stdint.h>

namespace {

constexpr uint32_t kMmaM = 64;
constexpr uint32_t kMmaN = 8;
constexpr uint32_t kMmaK = 32;
constexpr uint32_t kRowsPerCore = 8;
constexpr uint32_t kBytesPerCoreRow = 16;
constexpr uint32_t kPackedChunkBytes = kRowsPerCore * kBytesPerCoreRow;
constexpr uint32_t kMaxWarpgroups = 8;
constexpr uint32_t kPipelineStages = 2;
constexpr uint32_t kAStageBytes = kMmaM * kMmaK;
constexpr uint32_t kBStageBytes = kMmaN * kMmaK;
constexpr uint32_t kStageBytes = kAStageBytes + kBStageBytes;
static_assert(kPackedChunkBytes == 128, "the K-major core must occupy one 128B chunk");
static_assert(kStageBytes == 2304, "the WGMMA K32 stage size changed");

// The jackpot BLAKE3 invocation is the official V3 single-block keyed root hash.
__device__ __constant__ const uint32_t kBlake3Iv[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
};
__device__ __constant__ const uint8_t kBlake3Schedule[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};
constexpr uint32_t kBlake3Flags = 1u | 2u | 8u | 16u;
constexpr uint32_t kBlake3BlockBytes = 64u;
static_assert(kBlake3Flags == 27u, "the keyed single-block root flags must remain unchanged");
static_assert(kBlake3BlockBytes == 64u, "the jackpot message must remain one block");

__device__ __forceinline__ uint32_t shared_address(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t global_address(const void* ptr) {
  return static_cast<uint64_t>(__cvta_generic_to_global(ptr));
}

// PTX K-major canonical layout is ((8,m),(T,2k)):((1T,SBO),(1,LBO)).
// For signed 8-bit data T=16. A K32 panel is two 128B core chunks; rows are
// interleaved inside each chunk and the next row group starts at 256B.
__device__ __forceinline__ uint32_t packed_offset(uint32_t row, uint32_t k_byte) {
  const uint32_t row_group = row / kRowsPerCore;
  const uint32_t row_in_group = row % kRowsPerCore;
  const uint32_t k_chunk = k_byte / kBytesPerCoreRow;
  const uint32_t k_in_chunk = k_byte % kBytesPerCoreRow;
  return row_group * (kMmaK * kRowsPerCore) + k_chunk * kPackedChunkBytes +
      row_in_group * kBytesPerCoreRow + k_in_chunk;
}

__device__ __forceinline__ uint64_t make_k_major_desc(const uint8_t* matrix) {
  constexpr uint32_t lbo = kPackedChunkBytes;
  constexpr uint32_t sbo = kMmaK * kRowsPerCore;
  const uint64_t start = static_cast<uint64_t>(shared_address(matrix));
  return ((start >> 4) & 0x3fffULL) |
      (((static_cast<uint64_t>(lbo) >> 4) & 0x3fffULL) << 16) |
      (((static_cast<uint64_t>(sbo) >> 4) & 0x3fffULL) << 32);
}

__device__ __forceinline__ void mbarrier_init(uint64_t* barrier) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(shared_address(barrier)));
}

__device__ __forceinline__ uint64_t mbarrier_arrive_expect_tx(
    uint64_t* barrier, uint32_t bytes) {
  uint64_t token;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;"
               : "=l"(token)
               : "r"(shared_address(barrier)), "r"(bytes));
  return token;
}

__device__ __forceinline__ void tma_copy_16(
    uint8_t* dst, const int8_t* src, uint64_t* barrier) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], 16, [%2];"
      :
      : "r"(shared_address(dst)), "l"(global_address(src)),
        "r"(shared_address(barrier))
      : "memory");
}

__device__ __forceinline__ bool mbarrier_try_wait(
    uint64_t* barrier, uint64_t token) {
  uint32_t ready;
  asm volatile(
      "{ .reg .pred p; "
      "mbarrier.try_wait.acquire.cta.shared::cta.b64 p, [%1], %2; "
      "selp.b32 %0, 1, 0, p; }"
      : "=r"(ready)
      : "r"(shared_address(barrier)), "l"(token)
      : "memory");
  return ready != 0;
}

__device__ __forceinline__ void wait_for_tma(uint64_t* barrier, uint64_t token) {
  while (!mbarrier_try_wait(barrier, token)) asm volatile("nanosleep.u32 32;");
}

__device__ __forceinline__ void wgmma_zero(
    int32_t d[4], uint64_t desc_a, uint64_t desc_b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8 "
      "{%0, %1, %2, %3}, %4, %5, 0;"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "l"(desc_a), "l"(desc_b));
}

__device__ __forceinline__ void wgmma_accumulate(
    int32_t d[4], uint64_t desc_a, uint64_t desc_b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8 "
      "{%0, %1, %2, %3}, %4, %5, 1;"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "l"(desc_a), "l"(desc_b));
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_wait() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
}

__device__ __forceinline__ uint32_t load_u32_le(const uint8_t* bytes) {
  return static_cast<uint32_t>(bytes[0]) |
      (static_cast<uint32_t>(bytes[1]) << 8) |
      (static_cast<uint32_t>(bytes[2]) << 16) |
      (static_cast<uint32_t>(bytes[3]) << 24);
}

__device__ __forceinline__ void store_u32_le(uint8_t* bytes, uint32_t value) {
  bytes[0] = static_cast<uint8_t>(value);
  bytes[1] = static_cast<uint8_t>(value >> 8);
  bytes[2] = static_cast<uint8_t>(value >> 16);
  bytes[3] = static_cast<uint8_t>(value >> 24);
}

__device__ __forceinline__ void blake3_g(
    uint32_t state[16], uint32_t a, uint32_t b, uint32_t c, uint32_t d,
    uint32_t x, uint32_t y) {
  state[a] = state[a] + state[b] + x;
  state[d] = __funnelshift_r(state[d] ^ state[a], state[d] ^ state[a], 16);
  state[c] = state[c] + state[d];
  state[b] = __funnelshift_r(state[b] ^ state[c], state[b] ^ state[c], 12);
  state[a] = state[a] + state[b] + y;
  state[d] = __funnelshift_r(state[d] ^ state[a], state[d] ^ state[a], 8);
  state[c] = state[c] + state[d];
  state[b] = __funnelshift_r(state[b] ^ state[c], state[b] ^ state[c], 7);
}

__device__ __forceinline__ void blake3_round(
    uint32_t state[16], const uint32_t message[16], uint32_t round) {
  const uint8_t* schedule = kBlake3Schedule[round];
  blake3_g(state, 0, 4, 8, 12, message[schedule[0]], message[schedule[1]]);
  blake3_g(state, 1, 5, 9, 13, message[schedule[2]], message[schedule[3]]);
  blake3_g(state, 2, 6, 10, 14, message[schedule[4]], message[schedule[5]]);
  blake3_g(state, 3, 7, 11, 15, message[schedule[6]], message[schedule[7]]);
  blake3_g(state, 0, 5, 10, 15, message[schedule[8]], message[schedule[9]]);
  blake3_g(state, 1, 6, 11, 12, message[schedule[10]], message[schedule[11]]);
  blake3_g(state, 2, 7, 8, 13, message[schedule[12]], message[schedule[13]]);
  blake3_g(state, 3, 4, 9, 14, message[schedule[14]], message[schedule[15]]);
}

__device__ __forceinline__ void blake3_jackpot_hash(
    const uint32_t transcript[16], const uint8_t key[32], uint8_t digest[32]) {
  uint32_t state[16];
  uint32_t message[16];
  for (uint32_t i = 0; i < 8; ++i) {
    state[i] = load_u32_le(key + i * 4);
    state[i + 8] = i < 4 ? kBlake3Iv[i] : 0;
  }
  state[12] = 0;
  state[13] = 0;
  state[14] = kBlake3BlockBytes;
  state[15] = kBlake3Flags;
  for (uint32_t i = 0; i < 16; ++i) message[i] = transcript[i];
  for (uint32_t round = 0; round < 7; ++round) blake3_round(state, message, round);
  for (uint32_t i = 0; i < 8; ++i) store_u32_le(digest + i * 4, state[i] ^ state[i + 8]);
}

__device__ __forceinline__ void enqueue_stage(
    const int8_t* a, const int8_t* bt, uint8_t* stage, uint64_t* barrier,
    uint64_t* token_slot, uint32_t warp_m, uint32_t warp_n, uint32_t k_base,
    uint32_t physical_k) {
  *token_slot = mbarrier_arrive_expect_tx(barrier, kStageBytes);
  uint8_t* a_stage = stage;
  uint8_t* b_stage = stage + kAStageBytes;
  for (uint32_t row = 0; row < kMmaM; ++row) {
    for (uint32_t chunk = 0; chunk < 2; ++chunk) {
      const uint32_t k = k_base + chunk * kBytesPerCoreRow;
      tma_copy_16(a_stage + packed_offset(row, chunk * kBytesPerCoreRow),
                   a + (warp_m * kMmaM + row) * physical_k + k, barrier);
    }
  }
  for (uint32_t row = 0; row < kMmaN; ++row) {
    for (uint32_t chunk = 0; chunk < 2; ++chunk) {
      const uint32_t k = k_base + chunk * kBytesPerCoreRow;
      tma_copy_16(b_stage + packed_offset(row, chunk * kBytesPerCoreRow),
                   bt + (warp_n * kMmaN + row) * physical_k + k, barrier);
    }
  }
}

}  // namespace

extern "C" __global__ void pearl_sm90_wgmma_tile(
    const int8_t* a, const int8_t* bt,
    uint32_t logical_h, uint32_t logical_w,
    uint32_t padded_h, uint32_t padded_w,
    uint32_t physical_k, uint32_t effective_k, uint32_t rank,
    uint32_t* transcript, const uint8_t* a_noise_seed, uint8_t* digest) {
  const uint32_t thread = threadIdx.x;
  if (padded_h == 0 || padded_w == 0) return;
  const uint32_t tiles_m = padded_h / kMmaM;
  const uint32_t tiles_n = padded_w / kMmaN;
  const uint32_t warpgroups = tiles_m * tiles_n;
  if (warpgroups == 0 || warpgroups > kMaxWarpgroups || rank == 0 ||
      rank > effective_k || (rank & (kMmaK - 1)) != 0 ||
      (physical_k & (kMmaK - 1)) != 0 ||
      (effective_k & (kMmaK - 1)) != 0 || effective_k > physical_k) return;

  // The producer is a full warp placed after the consumer warpgroups. Keeping
  // it outside the consumer warpgroups preserves the four-warp alignment
  // required by WGMMA while allowing one warp to own all TMA issue work.
  const uint32_t warp_id = thread / 32;
  const uint32_t lane = thread & 31;
  const uint32_t consumer_lane = thread & 127;
  const bool producer = warp_id == warpgroups * 4;
  const uint32_t warpgroup = thread / 128;
  if (!producer && warpgroup >= warpgroups) return;

  extern __shared__ uint8_t shared[];
  uint64_t* barriers = reinterpret_cast<uint64_t*>(shared);
  uint64_t* tokens = barriers + warpgroups * kPipelineStages;
  uint8_t* stages = reinterpret_cast<uint8_t*>(tokens + warpgroups * kPipelineStages);
  int32_t* accum = reinterpret_cast<int32_t*>(stages + warpgroups * kPipelineStages * kStageBytes);
  if (producer && lane == 0) {
    for (uint32_t group = 0; group < warpgroups; ++group) {
      mbarrier_init(&barriers[group * kPipelineStages]);
      mbarrier_init(&barriers[group * kPipelineStages + 1]);
    }
  }
  __syncthreads();

  int32_t d[4] = {0, 0, 0, 0};
  const uint32_t k_steps = effective_k / kMmaK;
  if (producer && lane == 0) {
    for (uint32_t group = 0; group < warpgroups; ++group) {
      const uint32_t warp_m = group / tiles_n;
      const uint32_t warp_n = group % tiles_n;
      enqueue_stage(a, bt,
                    stages + group * kPipelineStages * kStageBytes,
                    barriers + group * kPipelineStages,
                    tokens + group * kPipelineStages,
                    warp_m, warp_n, 0, physical_k);
    }
  }
  __syncthreads();

  for (uint32_t step = 0; step < k_steps; ++step) {
    const uint32_t current = step & 1;
    if (producer && lane == 0 && step + 1 < k_steps) {
      const uint32_t next = (step + 1) & 1;
      for (uint32_t group = 0; group < warpgroups; ++group) {
        const uint32_t warp_m = group / tiles_n;
        const uint32_t warp_n = group % tiles_n;
        enqueue_stage(a, bt,
                      stages + (group * kPipelineStages + next) * kStageBytes,
                      barriers + group * kPipelineStages + next,
                      tokens + group * kPipelineStages + next,
                      warp_m, warp_n, (step + 1) * kMmaK, physical_k);
      }
    }
    __syncthreads();
    if (!producer) {
      uint64_t* group_barriers = barriers + warpgroup * kPipelineStages;
      uint64_t* group_tokens = tokens + warpgroup * kPipelineStages;
      wait_for_tma(&group_barriers[current], group_tokens[current]);
    }
    __syncthreads();

    if (!producer) {
      uint8_t* group_stages = stages + warpgroup * kPipelineStages * kStageBytes;
      const uint8_t* current_stage = group_stages + current * kStageBytes;
      const uint64_t desc_a = make_k_major_desc(current_stage);
      const uint64_t desc_b = make_k_major_desc(current_stage + kAStageBytes);
      wgmma_fence();
      if (step == 0) wgmma_zero(d, desc_a, desc_b);
      else wgmma_accumulate(d, desc_a, desc_b);
      wgmma_commit_wait();
    }
    // Do not let the producer reuse the stage or barrier until every WGMMA
    // consumer has completed its async group.
    __syncthreads();

    if (((step + 1) % (rank / kMmaK)) == 0) {
      if (!producer) {
        int32_t* group_accum = accum + warpgroup * kMmaM * kMmaN;
        const uint32_t row_base = (warpgroup / tiles_n) * kMmaM;
        const uint32_t col_base = (warpgroup % tiles_n) * kMmaN;
        const uint32_t warp = consumer_lane / 32;
        const uint32_t lane_in_warp = consumer_lane & 31;
        const uint32_t row = row_base + warp * 16 + lane_in_warp / 4;
        const uint32_t col = col_base + (lane_in_warp & 3) * 2;
        group_accum[(row - row_base) * kMmaN + (col - col_base)] = d[0];
        group_accum[(row - row_base) * kMmaN + (col - col_base) + 1] = d[1];
        group_accum[(row - row_base + 8) * kMmaN + (col - col_base)] = d[2];
        group_accum[(row - row_base + 8) * kMmaN + (col - col_base) + 1] = d[3];
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        uint32_t fold = 0;
        for (uint32_t group = 0; group < warpgroups; ++group) {
          const uint32_t group_row = (group / tiles_n) * kMmaM;
          const uint32_t group_col = (group % tiles_n) * kMmaN;
          for (uint32_t r = 0; r < kMmaM; ++r) {
            for (uint32_t c = 0; c < kMmaN; ++c) {
              if (group_row + r < logical_h && group_col + c < logical_w) {
                fold ^= static_cast<uint32_t>(accum[group * kMmaM * kMmaN + r * kMmaN + c]);
              }
            }
          }
        }
        const uint32_t slot = (step / (rank / kMmaK)) & 15;
        transcript[slot] = (transcript[slot] << 13 | transcript[slot] >> 19) ^ fold;
      }
      __syncthreads();
      // Preserve cumulative MMA state across rank boundaries (V3 transcript).
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) blake3_jackpot_hash(transcript, a_noise_seed, digest);
}
