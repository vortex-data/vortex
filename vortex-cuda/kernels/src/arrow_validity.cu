// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"

#include <stdint.h>

namespace {

// Transform up to 8 input bytes into a zero-extended 64-bit word:
//
//   [ b0 ][ b1 ][ b2 ] | end  ->  [ b0 ][ b1 ][ b2 ][ 00 ][ 00 ][ 00 ][ 00 ][ 00 ]
__device__ uint64_t load_input_word(const uint8_t *const input, int64_t word_idx, uint64_t input_bytes) {
    if (word_idx < 0) {
        return 0;
    }
    const uint64_t byte_idx = static_cast<uint64_t>(word_idx) * sizeof(uint64_t);
    if (byte_idx >= input_bytes) {
        return 0;
    }
    if (byte_idx + sizeof(uint64_t) <= input_bytes) {
        return reinterpret_cast<const uint64_t *>(input)[word_idx];
    }
    // Trailing partial word: assemble byte-by-byte to avoid reading past the buffer.
    uint64_t word = 0;
    for (uint64_t i = byte_idx; i < input_bytes; i++) {
        word |= static_cast<uint64_t>(input[i]) << ((i - byte_idx) * 8);
    }
    return word;
}

// Build one output word for sliced validity. The row bits are the same, but
// row 0 may live at a different bit position in the source and Arrow bitmaps.
// For example, `input_offset = 5` and `arrow_offset = 0` shifts row0 from bit 5
// in the input bitmap to bit 0 in the Arrow bitmap.
//
//   input bitmap:  [ . ][ . ][ . ][ . ][ . ][ row0 ][ row1 ][ row2 ]....
//                                            ^ input_offset
//   Arrow bitmap:  [ row0 ][ row1 ][ row2 ]....
//                     ^ arrow_offset
//
// Padding bits are cleared so word-sized validity readers can safely over-read.
__device__ uint64_t repack_word(const uint8_t *const input,
                                uint64_t word_idx,
                                int64_t shift,
                                uint64_t arrow_offset,
                                uint64_t validity_bits,
                                uint64_t input_bytes) {
    const uint64_t word_start = word_idx * 64;

    // Bits before Arrow's array offset are padding from the consumer's point of view.
    // Tail bits beyond len + offset stay zero so word-at-a-time mask readers are safe.
    uint64_t mask = ~uint64_t {0};
    if (word_start < arrow_offset) {
        const uint64_t lead = arrow_offset - word_start;
        mask = lead >= 64 ? 0 : mask << lead;
    }
    const uint64_t remaining = validity_bits - word_start;
    if (remaining < 64) {
        mask &= (uint64_t {1} << remaining) - 1;
    }
    if (mask == 0) {
        return 0;
    }

    // Each output bit `b` reads source bit `b + shift`.
    // `>> 6` floors for negative positions, unlike `/ 64` which truncates toward zero.
    const int64_t source_bit_start = static_cast<int64_t>(word_start) + shift;
    const int64_t source_word = source_bit_start >> 6;
    const uint32_t source_bit = static_cast<uint32_t>(source_bit_start & 63);

    const uint64_t lo = load_input_word(input, source_word, input_bytes);
    if (source_bit == 0) {
        return lo & mask;
    }
    const uint64_t hi = load_input_word(input, source_word + 1, input_bytes);
    return ((lo >> source_bit) | (hi << (64 - source_bit))) & mask;
}

constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t FULL_WARP_MASK = 0xffffffff;

// First reduction step for the count kernel: sum one value per lane so each
// warp produces a single partial count.
//
//   lanes:  [a][b][c][d]... -> lane 0: a+b+c+d+...
__device__ uint64_t warp_sum(uint64_t value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(FULL_WARP_MASK, value, offset);
    }
    return value;
}

// Mask one bitmap byte down to actual rows. This keeps null counting from
// including Arrow offset padding or trailing padding bits.
//
//   byte bits:  [ pad ][ row ][ row ][ row ][ pad ]
//   mask:       [  0  ][  1  ][  1  ][  1  ][  0  ]
__device__ uint32_t arrow_validity_byte_mask(uint64_t byte_idx,
                                             uint64_t arrow_offset,
                                             uint64_t validity_bits) {
    const uint64_t byte_start = byte_idx * 8;

    uint32_t mask = 0xff;
    if (byte_start < arrow_offset) {
        const uint64_t lead = arrow_offset - byte_start;
        mask = lead >= 8 ? 0 : mask << lead;
    }

    const uint64_t remaining = validity_bits - byte_start;
    if (remaining < 8) {
        mask &= (uint32_t {1} << remaining) - 1;
    }
    return mask;
}

// Combine warp partial counts into one block total. Only thread 0 returns a
// non-zero value so the count kernel does one global atomic per block.
//
//   per-thread counts -> per-warp sums -> block sum -> atomicAdd
__device__ uint64_t block_sum_to_thread_zero(uint64_t value, uint64_t *const warp_counts) {
    const uint32_t thread = threadIdx.x;
    const uint32_t lane = thread & (WARP_SIZE - 1);
    const uint32_t warp = thread / WARP_SIZE;
    const uint32_t block_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    value = warp_sum(value);
    if (lane == 0) {
        warp_counts[warp] = value;
    }
    __syncthreads();

    value = lane < block_warps ? warp_counts[lane] : 0;
    value = warp == 0 ? warp_sum(value) : 0;
    return thread == 0 ? value : 0;
}

} // namespace

// Repack sliced validity when the source bitmap offset does not match the
// Arrow array offset. Each thread writes independent output words.
//
//   thread 0 -> output word 0, word N, ...
//   thread 1 -> output word 1, word N+1, ...
extern "C" __global__ void arrow_validity_repack(const uint8_t *const input,
                                                 uint64_t *const output,
                                                 uint64_t len,
                                                 uint64_t input_offset,
                                                 uint64_t arrow_offset,
                                                 uint64_t input_bytes) {
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t validity_bits = len + arrow_offset;
    const uint64_t output_words = (validity_bits + 63) / 64;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    const int64_t shift = static_cast<int64_t>(input_offset) - static_cast<int64_t>(arrow_offset);

    for (uint64_t word_idx = worker; word_idx < output_words; word_idx += stride) {
        output[word_idx] = repack_word(input, word_idx, shift, arrow_offset, validity_bits, input_bytes);
    }
}

// Count valid rows directly from the device bitmap so Arrow export can provide
// an exact null_count without copying validity to the CPU.
//
//   bytes -> mask padding -> popcount -> block sum -> global count
extern "C" __global__ void arrow_validity_count_valid(const uint8_t *const input,
                                                      uint64_t *const output,
                                                      uint64_t len,
                                                      uint64_t arrow_offset) {
    __shared__ uint64_t warp_counts[WARP_SIZE];

    const uint64_t validity_bits = len + arrow_offset;
    const uint64_t input_bytes = (validity_bits + 7) / 8;
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;

    // Grid-stride over bitmap bytes. Each byte contributes the popcount of only
    // row bits; leading Arrow offset bits and trailing padding bits are masked out.
    uint64_t valid_count = 0;
    for (uint64_t byte_idx = worker; byte_idx < input_bytes; byte_idx += stride) {
        const uint32_t mask = arrow_validity_byte_mask(byte_idx, arrow_offset, validity_bits);
        valid_count += __popc(static_cast<uint32_t>(input[byte_idx]) & mask);
    }

    // Reduce within the block first so global contention is one atomic add per block.
    valid_count = block_sum_to_thread_zero(valid_count, warp_counts);
    if (threadIdx.x == 0) {
        atomicAdd(reinterpret_cast<unsigned long long *>(output),
                  static_cast<unsigned long long>(valid_count));
    }
}
