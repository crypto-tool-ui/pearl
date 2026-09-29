// M5: GPU-resident keyed BLAKE3 chunk and Merkle reduction for V3 job setup.
//
// The layout and flags mirror pearl-blake3/src/{hasher,merkle}.rs. The first
// kernel hashes 1024-byte chunks to chaining values; the second combines
// chaining values level by level and applies ROOT only to the final parent.
// This file contains no consensus parameters beyond the official BLAKE3
// constants and the 1024-byte chunk size.

#include <stdint.h>
#include <stddef.h>

namespace {

constexpr uint32_t kChunkBytes = 1024;
constexpr uint32_t kBlockBytes = 64;
constexpr uint32_t kBlocksPerChunk = kChunkBytes / kBlockBytes;
constexpr uint32_t kDigestBytes = 32;
constexpr uint32_t kChunkStart = 1;
constexpr uint32_t kChunkEnd = 2;
constexpr uint32_t kParent = 4;
constexpr uint32_t kRoot = 8;
constexpr uint32_t kKeyedHash = 16;

__device__ __constant__ const uint32_t kIv[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
};

__device__ __constant__ const uint8_t kSchedule[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

__device__ __forceinline__ uint32_t load32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) |
      (static_cast<uint32_t>(p[1]) << 8) |
      (static_cast<uint32_t>(p[2]) << 16) |
      (static_cast<uint32_t>(p[3]) << 24);
}

__device__ __forceinline__ void store32(uint8_t* p, uint32_t v) {
  p[0] = static_cast<uint8_t>(v);
  p[1] = static_cast<uint8_t>(v >> 8);
  p[2] = static_cast<uint8_t>(v >> 16);
  p[3] = static_cast<uint8_t>(v >> 24);
}

__device__ __forceinline__ uint32_t rotr(uint32_t v, uint32_t n) {
  return (v >> n) | (v << (32 - n));
}

__device__ __forceinline__ void g(
    uint32_t s[16], uint32_t a, uint32_t b, uint32_t c, uint32_t d,
    uint32_t x, uint32_t y) {
  s[a] = s[a] + s[b] + x;
  s[d] = rotr(s[d] ^ s[a], 16);
  s[c] += s[d];
  s[b] = rotr(s[b] ^ s[c], 12);
  s[a] = s[a] + s[b] + y;
  s[d] = rotr(s[d] ^ s[a], 8);
  s[c] += s[d];
  s[b] = rotr(s[b] ^ s[c], 7);
}

__device__ __forceinline__ void compress(
    const uint32_t cv[8], const uint32_t message[16], uint64_t counter,
    uint32_t block_len, uint32_t flags, uint32_t out[16]) {
  for (uint32_t i = 0; i < 8; ++i) out[i] = cv[i];
  for (uint32_t i = 0; i < 4; ++i) out[i + 8] = kIv[i];
  out[12] = static_cast<uint32_t>(counter);
  out[13] = static_cast<uint32_t>(counter >> 32);
  out[14] = block_len;
  out[15] = flags;
  for (uint32_t round = 0; round < 7; ++round) {
    const uint8_t* p = kSchedule[round];
    g(out, 0, 4, 8, 12, message[p[0]], message[p[1]]);
    g(out, 1, 5, 9, 13, message[p[2]], message[p[3]]);
    g(out, 2, 6, 10, 14, message[p[4]], message[p[5]]);
    g(out, 3, 7, 11, 15, message[p[6]], message[p[7]]);
    g(out, 0, 5, 10, 15, message[p[8]], message[p[9]]);
    g(out, 1, 6, 11, 12, message[p[10]], message[p[11]]);
    g(out, 2, 7, 8, 13, message[p[12]], message[p[13]]);
    g(out, 3, 4, 9, 14, message[p[14]], message[p[15]]);
  }
}

__device__ __forceinline__ void cv_from_output(const uint32_t state[16], uint32_t cv[8]) {
  for (uint32_t i = 0; i < 8; ++i) cv[i] = state[i] ^ state[i + 8];
}

__device__ __forceinline__ void key_words(const uint8_t* key, uint32_t words[8]) {
  for (uint32_t i = 0; i < 8; ++i) words[i] = load32(key + i * 4);
}

__device__ __forceinline__ void hash_chunk(
    const uint8_t* data, uint64_t chunk_index, const uint8_t* key,
    bool root_output, uint8_t* output) {
  uint32_t cv[8];
  key_words(key, cv);
  uint32_t message[16];
  uint32_t state[16];
  for (uint32_t block = 0; block < kBlocksPerChunk; ++block) {
    const uint8_t* block_data = data + block * kBlockBytes;
    for (uint32_t i = 0; i < 16; ++i) message[i] = load32(block_data + i * 4);
    uint32_t flags = kKeyedHash;
    if (block == 0) flags |= kChunkStart;
    if (block + 1 == kBlocksPerChunk) {
      flags |= kChunkEnd;
      if (root_output) flags |= kRoot;
    }
    compress(cv, message, chunk_index, kBlockBytes, flags, state);
    if (block + 1 != kBlocksPerChunk) cv_from_output(state, cv);
  }
  if (root_output) {
    for (uint32_t i = 0; i < 8; ++i) store32(output + i * 4, state[i] ^ state[i + 8]);
  } else {
    for (uint32_t i = 0; i < 8; ++i) store32(output + i * 4, state[i] ^ state[i + 8]);
  }
}

__device__ __forceinline__ void parent_hash(
    const uint8_t* left, const uint8_t* right, const uint8_t* key,
    bool root_output, uint8_t* output) {
  uint32_t cv[8];
  uint32_t message[16];
  uint32_t state[16];
  key_words(key, cv);
  for (uint32_t i = 0; i < 8; ++i) message[i] = load32(left + i * 4);
  for (uint32_t i = 0; i < 8; ++i) message[i + 8] = load32(right + i * 4);
  compress(cv, message, 0, kDigestBytes,
           kKeyedHash | kParent | (root_output ? kRoot : 0), state);
  for (uint32_t i = 0; i < 8; ++i) store32(output + i * 4, state[i] ^ state[i + 8]);
}

}  // namespace

extern "C" __global__ void pearl_tensor_hash_leaf_cv(
    const uint8_t* data, uint32_t chunk_count, const uint8_t* key,
    uint32_t root_output, uint8_t* output) {
  const uint32_t chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= chunk_count) return;
  hash_chunk(data + static_cast<size_t>(chunk) * kChunkBytes, chunk, key,
             root_output != 0, output + static_cast<size_t>(chunk) * kDigestBytes);
}

extern "C" __global__ void pearl_tensor_hash_parent_cv(
    const uint8_t* input, uint32_t input_count, const uint8_t* key,
    uint32_t root_output, uint8_t* output) {
  const uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t output_count = (input_count + 1) / 2;
  if (pair >= output_count) return;
  const uint32_t left_index = pair * 2;
  uint8_t* dst = output + static_cast<size_t>(pair) * kDigestBytes;
  const uint8_t* left = input + static_cast<size_t>(left_index) * kDigestBytes;
  if (left_index + 1 >= input_count) {
    for (uint32_t i = 0; i < kDigestBytes; ++i) dst[i] = left[i];
    return;
  }
  parent_hash(left, left + kDigestBytes, key,
              root_output != 0 && input_count == 2, dst);
}
