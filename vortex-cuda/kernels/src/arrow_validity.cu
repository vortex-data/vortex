// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"

#include <stdint.h>

namespace {

// Load the `word_idx`-th little-endian u64 of `input`, treating bytes outside
// `[0, input_bytes)` as zero. `input` must be 8-byte aligned.
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

// Build one 64-bit word of the Arrow validity bitmap.
//
// Output bit `b` for `b` in `[arrow_offset, validity_bits)` equals input bit `b + shift`;
// all other bits are zero. Two adjacent input words are funnel-shifted to align the input
// bits with the output word, then the leading/trailing edges are masked.
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

    // `>> 6` floors also for negative bit positions, unlike `/ 64` which truncates toward zero.
    const int64_t input_bit = static_cast<int64_t>(word_start) + shift;
    const int64_t input_word = input_bit >> 6;
    const uint32_t bit = static_cast<uint32_t>(input_bit & 63);

    const uint64_t lo = load_input_word(input, input_word, input_bytes);
    if (bit == 0) {
        return lo & mask;
    }
    const uint64_t hi = load_input_word(input, input_word + 1, input_bytes);
    return ((lo >> bit) | (hi << (64 - bit))) & mask;
}

// Rebuild a possibly bit-offset Vortex validity bitmap into an Arrow-compatible bitmap.
//
// `input_offset` is the bit offset into `input`; `arrow_offset` is the logical Arrow array offset
// to preserve in the output. Bits outside `[arrow_offset, arrow_offset + len)` are left unset.
// The output allocation must hold `ceil((len + arrow_offset) / 64)` full 64-bit words; every
// word is written, so no zero-initialization of the output is required.
__device__ void arrow_validity_repack_device(const uint8_t *const input,
                                             uint64_t *const output,
                                             uint64_t len,
                                             uint64_t input_offset,
                                             uint64_t arrow_offset,
                                             uint64_t input_bytes) {
    // One worker owns a contiguous range of output words. Each word is rebuilt locally so
    // there are no cross-thread bit writes or atomics.
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t validity_bits = len + arrow_offset;
    const uint64_t output_words = (validity_bits + 63) / 64;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;

    // Translate Arrow-visible output bits back to source bitmap bits. The source bitmap may
    // start at any bit offset, while Arrow's buffer pointer is byte-addressed.
    const int64_t shift = static_cast<int64_t>(input_offset) - static_cast<int64_t>(arrow_offset);

    for (uint64_t word_idx = worker; word_idx < output_words; word_idx += stride) {
        output[word_idx] = repack_word(input, word_idx, shift, arrow_offset, validity_bits, input_bytes);
    }
}

} // namespace

// CUDA entry point for validity bitmap repacking used by Arrow Device export.
extern "C" __global__ void arrow_validity_repack(const uint8_t *const input,
                                                 uint64_t *const output,
                                                 uint64_t len,
                                                 uint64_t input_offset,
                                                 uint64_t arrow_offset,
                                                 uint64_t input_bytes) {
    arrow_validity_repack_device(input, output, len, input_offset, arrow_offset, input_bytes);
}
