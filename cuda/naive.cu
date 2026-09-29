// M2 correctness oracle. It intentionally uses one CUDA thread per tile and no tensor-core
// instruction. M3/M4 replace only the implementation behind GemmKernel; this oracle stays as the
// bit-exact device reference.

#include <stdint.h>

extern "C" __global__ void pearl_naive_tile(
    const int8_t* a,
    const int8_t* bt,
    uint32_t h,
    uint32_t w,
    uint32_t k,
    uint32_t rank,
    int32_t* accum,
    uint32_t* transcript) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || rank == 0 || rank > k) return;

  for (uint32_t slice_end = rank; slice_end <= k; slice_end += rank) {
    const uint32_t slice_begin = slice_end - rank;
    for (uint32_t row = 0; row < h; ++row) {
      for (uint32_t col = 0; col < w; ++col) {
        int32_t sum = 0;
        for (uint32_t l = slice_begin; l < slice_end; ++l) {
          const int32_t av = static_cast<int32_t>(a[row * k + l]);
          const int32_t bv = static_cast<int32_t>(bt[col * k + l]);
          sum += av * bv;
        }
        accum[row * w + col] += sum;
      }
    }

    uint32_t fold = 0;
    for (uint32_t cell = 0; cell < h * w; ++cell) {
      fold ^= static_cast<uint32_t>(accum[cell]);
    }
    const uint32_t slot = (slice_end / rank - 1) % 16;
    transcript[slot] = (transcript[slot] << 13 | transcript[slot] >> 19) ^ fold;
  }
}
