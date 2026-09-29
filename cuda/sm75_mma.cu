// Turing / T4: signed INT8 Tensor Core GEMM, with the Pearl V3 cumulative transcript.
// PTX m8n8k16 fragments: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m8n8k16
// Direct register loads avoid cp.async/TMA, which are unavailable on sm_75.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 750
#error "sm75_mma.cu must be compiled for sm_75"
#endif
#include <stdint.h>
#include <stddef.h>
namespace {
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
constexpr uint32_t kBlake3Flags = 1u | 2u | 8u | 16u;  // CHUNK_START|CHUNK_END|ROOT|KEYED_HASH
constexpr uint32_t kBlake3BlockBytes = 16u * sizeof(uint32_t);
static_assert(kBlake3Flags == 27u, "the keyed single-block root flags must remain unchanged");
static_assert(kBlake3BlockBytes == 64u, "the jackpot message must remain one 64-byte block");

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
  for (uint32_t i = 0; i < 16; ++i) {
    message[i] = transcript[i];
  }
  for (uint32_t round = 0; round < 7; ++round) {
    blake3_round(state, message, round);
  }
  for (uint32_t i = 0; i < 8; ++i) {
    store_u32_le(digest + i * 4, state[i] ^ state[i + 8]);
  }
}


__device__ __forceinline__ void mma_int8(int32_t d[2], uint32_t a, uint32_t b) {
  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
               "{%0, %1}, {%2}, {%3}, {%0, %1};"
               : "+r"(d[0]), "+r"(d[1]) : "r"(a), "r"(b));
}
}  // namespace

extern "C" __global__ void pearl_sm75_mma_tile(
    const int8_t* a, const int8_t* bt,
    uint32_t logical_h, uint32_t logical_w,
    uint32_t padded_h, uint32_t padded_w,
    uint32_t physical_k, uint32_t effective_k, uint32_t rank,
    uint32_t* transcript, const uint8_t* a_noise_seed, uint8_t* digest) {
  a += static_cast<size_t>(blockIdx.x) * padded_h * physical_k;
  bt += static_cast<size_t>(blockIdx.x) * padded_w * physical_k;
  transcript += blockIdx.x * 16;
  digest += blockIdx.x * 32;
  // Uniform checks: every thread in this CTA reaches every block barrier.
  const uint32_t warps = (padded_h / 8) * (padded_w / 8);
  if (!logical_h || !logical_w || logical_h > padded_h || logical_w > padded_w ||
      padded_h % 8 || padded_w % 8 || !warps || warps > 32 ||
      blockDim.x != warps * 32 || !rank || rank % 16 ||
      physical_k % 16 || !effective_k || effective_k > physical_k || effective_k % rank) return;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t row_base = (warp / (padded_w / 8)) * 8;
  const uint32_t col_base = (warp % (padded_w / 8)) * 8;
  const uint32_t group = lane >> 2;
  const uint32_t thread = lane & 3;
  extern __shared__ int32_t accum[];
  int32_t d[2] = {0, 0};
  // Keep d cumulative across rank slices; each boundary folds the running product.
  for (uint32_t slice = 0; slice < effective_k; slice += rank) {
    for (uint32_t kk = slice; kk < slice + rank; kk += 16) {
      const uint32_t af = load_u32_le(reinterpret_cast<const uint8_t*>(
          a + (row_base + group) * physical_k + kk + thread * 4));
      // B is column-major in MMA; the host supplies contiguous rows of B transpose.
      const uint32_t bf = load_u32_le(reinterpret_cast<const uint8_t*>(
          bt + (col_base + group) * physical_k + kk + thread * 4));
      mma_int8(d, af, bf);
    }
    const uint32_t row = row_base + group;
    for (uint32_t i = 0; i < 2; ++i) {
      const uint32_t col = col_base + thread * 2 + i;
      if (row < logical_h && col < logical_w) accum[row * logical_w + col] = d[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t fold = 0;
      for (uint32_t cell = 0; cell < logical_h * logical_w; ++cell)
        fold ^= static_cast<uint32_t>(accum[cell]);
      const uint32_t slot = (slice / rank) & 15;
      transcript[slot] = (transcript[slot] << 13 | transcript[slot] >> 19) ^ fold;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) blake3_jackpot_hash(transcript, a_noise_seed, digest);
}
