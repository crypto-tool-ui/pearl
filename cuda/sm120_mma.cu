// M3: V3/int7 Pearl tile GEMM for sm_120a.
//
// The host supplies the official noised int8 operands.  A tile is padded to 16 rows and
// 8 columns by the host, while the logical dimensions still control the transcript fold.
// Every 32-wide K step is one mma.sync.aligned.m16n8k32 operation.  The accumulator fragment
// remains in registers; the only intermediate store is to shared memory for the fused
// transcript fold.  No full tile is written back to global memory.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1200
#error "sm120_mma.cu must be compiled for sm_120a"
#endif

#include <stdint.h>

namespace {

constexpr uint32_t kMmaM = 16;
constexpr uint32_t kMmaN = 8;
constexpr uint32_t kMmaK = 32;
constexpr uint32_t kMaxWarps = 32;
// Each logical row occupies one 128-byte swizzle span. The operands only use the first
// two 16-byte chunks of that span, but the full span is reserved so the software layout
// follows the documented 128B/16B-chunk mapping without TMA.
constexpr uint32_t kSwizzleSpanBytes = 128;
constexpr uint32_t kSwizzleChunkBytes = 16;
constexpr uint32_t kSwizzleChunks = kSwizzleSpanBytes / kSwizzleChunkBytes;
static_assert(kSwizzleChunks == 8,
              "the 128-byte swizzle span must contain eight 16-byte chunks");
constexpr uint32_t kAStageBytes = kMmaM * kSwizzleSpanBytes;
constexpr uint32_t kBRawStageBytes = kMmaN * kSwizzleSpanBytes;
constexpr uint32_t kStageBytes = kAStageBytes + kBRawStageBytes;

// These constants and the seven-round compression schedule mirror the BLAKE3 implementation
// used by the official Pearl reference (pearl-blake3 -> blake3).  The jackpot message is exactly
// one 64-byte block, so no tree parent or second output block is involved here.
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

__device__ __forceinline__ uint32_t shared_address(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// CUDA documents the 128B swizzle as an XOR of the eight 16-byte chunks in a row,
// offset by the row number and the shared-memory base alignment. cp.async stores to
// this address and ldmatrix uses the inverse (the same XOR) address below.
__device__ __forceinline__ uint32_t swizzled_chunk(
    const uint8_t* base, uint32_t row, uint32_t logical_chunk) {
  const uint32_t base_offset =
      (shared_address(base) / kSwizzleSpanBytes) % kSwizzleChunks;
  const uint32_t physical_chunk =
      ((row + base_offset) % kSwizzleChunks) ^ logical_chunk;
  return physical_chunk * kSwizzleChunkBytes;
}

__device__ __forceinline__ void cp_async_16(void* dst, const int8_t* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
               :
               : "r"(shared_address(dst)), "l"(src));
}

__device__ __forceinline__ void commit_async() {
  asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void mma_int8(
    int32_t d[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
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

// ldmatrix uses the same warp fragment contract as the m16n8k32 MMA instruction.  A is
// represented as a 16x16 matrix of 16-bit carriers (two int8 values per carrier). B is the
// row-major B^T view supplied by Pearl, represented as an 8x16 carrier matrix and loaded with
// the transpose form so the MMA fragment sees column-major B.
__device__ __forceinline__ void load_fragments(
    const uint8_t* a_smem, const uint8_t* b_smem, uint32_t lane,
    uint32_t a[4], uint32_t b[2]) {
  const uint16_t* a_half = reinterpret_cast<const uint16_t*>(a_smem);
  const uint16_t* b_half = reinterpret_cast<const uint16_t*>(b_smem);
  // PTX ldmatrix supplies x4 row addresses from lane groups 0..7, 8..15, 16..23,
  // and 24..31, and x2.trans supplies its two matrices from groups 0..7 and 8..15.
  // Every lane still executes both aligned instructions; the addresses below are valid for
  // all lanes, including the lanes whose x2 address is ignored by the instruction.
  // x4 consumes four 8x8 carrier matrices. The mma.m16n8k32 A fragment's register
  // order is top-left, bottom-left, top-right, bottom-right (a0..a3). Therefore
  // lane groups 0..7, 8..15, 16..23, and 24..31 address those matrices in that order.
  const uint32_t matrix_group = lane >> 3;
  const uint32_t matrix_row = (lane & 7) + (matrix_group & 1) * 8;
  const uint32_t matrix_col = (matrix_group >> 1) * 8;
  const uint16_t* a_ptr = a_half + matrix_row * (kSwizzleSpanBytes / 2) +
      swizzled_chunk(a_smem, matrix_row, matrix_col / 8) / 2;
  // The transposed B load uses the two K halves in register order: b0 covers K[0:16]
  // and b1 covers K[16:32]. The address groups below name those halves in B^T storage.
  const uint32_t b_row = lane & 7;
  const uint32_t b_col = ((lane >> 3) & 1) * 8;
  const uint16_t* b_ptr = b_half + b_row * (kSwizzleSpanBytes / 2) +
      swizzled_chunk(b_smem, b_row, b_col / 8) / 2;

  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(shared_address(a_ptr)));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(shared_address(b_ptr)));
}

__device__ __forceinline__ void enqueue_stage(
    const int8_t* a, const int8_t* bt, uint8_t* stage,
    uint32_t warp_m, uint32_t warp_n, uint32_t k_base, uint32_t physical_k,
    uint32_t lane) {
  uint8_t* a_stage = stage;
  uint8_t* b_stage = stage + kAStageBytes;

  const uint32_t a_byte = lane * 16;
  const uint32_t a_row = a_byte / kMmaK;
  const uint32_t a_col = a_byte % kMmaK;
  cp_async_16(
      a_stage + a_row * kSwizzleSpanBytes + swizzled_chunk(a_stage, a_row, a_col / 16),
      a + (warp_m * kMmaM + a_row) * physical_k + k_base + a_col);

  // The B operand is eight rows of B^T, each with 32 contiguous K bytes.  Only lanes 0..15
  // issue these sixteen-byte copies: each destination chunk has one writer, while every lane
  // still commits the same async-copy group below.
  if (lane < 16) {
    const uint32_t b_byte = lane * 16;
    const uint32_t b_row = b_byte / kMmaK;
    const uint32_t b_col = b_byte % kMmaK;
    cp_async_16(
        b_stage + b_row * kSwizzleSpanBytes + swizzled_chunk(b_stage, b_row, b_col / 16),
        bt + (warp_n * kMmaN + b_row) * physical_k + k_base + b_col);
  }
}

}  // namespace

extern "C" __global__ void pearl_sm120_mma_tile(
    const int8_t* a,
    const int8_t* bt,
    uint32_t logical_h,
    uint32_t logical_w,
    uint32_t padded_h,
    uint32_t padded_w,
    uint32_t physical_k,
    uint32_t effective_k,
    uint32_t rank,
    uint32_t tile_count,
    uint32_t* transcript,
    const uint8_t* a_noise_seed,
    uint8_t* digest) {
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t tile = blockIdx.x;
  if (tile >= tile_count) return;
  if (padded_h == 0 || padded_w == 0) {
    return;
  }
  const uint32_t tiles_m = (padded_h + kMmaM - 1) / kMmaM;
  const uint32_t tiles_n = (padded_w + kMmaN - 1) / kMmaN;
  const uint32_t warps = tiles_m * tiles_n;
  const uint32_t warp_m = warp / tiles_n;
  const uint32_t warp_n = warp % tiles_n;
  if (warp_m >= tiles_m || warp_n >= tiles_n || warps == 0 || warps > kMaxWarps ||
      rank == 0 || rank > effective_k ||
      (rank & 31) != 0 || (physical_k & 31) != 0 || (effective_k & 31) != 0 ||
      effective_k > physical_k) {
    return;
  }

  extern __shared__ uint8_t shared[];
  uint8_t* stages = shared;
  uint32_t* warp_folds = reinterpret_cast<uint32_t*>(shared + 2 * warps * kStageBytes);
  uint8_t* warp_stage0 = stages + warp * kStageBytes * 2;
  uint8_t* warp_stage1 = warp_stage0 + kStageBytes;
  a += static_cast<uint64_t>(tile) * padded_h * physical_k;
  bt += static_cast<uint64_t>(tile) * padded_w * physical_k;
  transcript += static_cast<uint64_t>(tile) * 16;
  digest += static_cast<uint64_t>(tile) * 32;

  int32_t d[4] = {0, 0, 0, 0};
  for (uint32_t slice_begin = 0; slice_begin < effective_k; slice_begin += rank) {
    const uint32_t k_steps = rank / kMmaK;
    enqueue_stage(a, bt, warp_stage0, warp_m, warp_n, slice_begin, physical_k, lane);
    commit_async();
    if (k_steps > 1) {
      enqueue_stage(
          a, bt, warp_stage1, warp_m, warp_n, slice_begin + kMmaK, physical_k, lane);
      commit_async();
    }

    for (uint32_t step = 0; step < k_steps; ++step) {
      // At most two groups are live. Waiting for one group leaves the successor in
      // flight while the current stage is consumed. The final stage has no successor.
      if (step + 1 < k_steps) {
        asm volatile("cp.async.wait_group 1;");
      } else {
        asm volatile("cp.async.wait_group 0;");
      }
      __syncthreads();
      const uint32_t stage_index = step & 1;
      uint8_t* current = stage_index == 0 ? warp_stage0 : warp_stage1;
      uint32_t a_frag[4];
      uint32_t b_frag[2];
      load_fragments(current, current + kAStageBytes, lane, a_frag, b_frag);
      mma_int8(d, a_frag, b_frag);
      __syncthreads();

      // The stage just consumed is now free. Reuse it for the K step after the
      // successor, keeping the two-stage pipeline full without overwriting an
      // outstanding async copy.
      if (step + 2 < k_steps) {
        enqueue_stage(
            a,
            bt,
            current,
            warp_m,
            warp_n,
            slice_begin + (step + 2) * kMmaK,
            physical_k,
            lane);
        commit_async();
      }
    }

    const uint32_t row_base = warp_m * kMmaM;
    const uint32_t col_base = warp_n * kMmaN;
    const uint32_t group = lane >> 2;
    const uint32_t thread = lane & 3;
    uint32_t lane_fold = 0;
    for (uint32_t i = 0; i < 4; ++i) {
      const uint32_t row = row_base + group + (i / 2) * 8;
      const uint32_t col = col_base + thread * 2 + (i & 1);
      if (row < logical_h && col < logical_w) {
        lane_fold ^= static_cast<uint32_t>(d[i]);
      }
    }
    // All lanes participate; padded cells contribute zero.
    for (uint32_t delta = 16; delta > 0; delta >>= 1)
      lane_fold ^= __shfl_down_sync(0xffffffffu, lane_fold, delta);
    if (lane == 0) warp_folds[warp] = lane_fold;
    __syncthreads();

    if (threadIdx.x == 0) {
      uint32_t fold = 0;
      for (uint32_t cell = 0; cell < warps; ++cell) {
        fold ^= warp_folds[cell];
      }
      const uint32_t slot = (slice_begin / rank) & 15;
      transcript[slot] = (transcript[slot] << 13 | transcript[slot] >> 19) ^ fold;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    blake3_jackpot_hash(transcript, a_noise_seed, digest);
  }
}
