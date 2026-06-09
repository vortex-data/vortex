// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"

#include <stdint.h>

namespace {

// Read one validity bit from a little-endian Arrow/Vortex bitmap.
__device__ bool get_bit(const uint8_t *const input, uint64_t bit_idx) {
    return (input[bit_idx / 8] >> (bit_idx % 8)) & 1;
}

// Rebuild a possibly bit-offset Vortex validity bitmap into an Arrow-compatible bitmap.
//
// `input_offset` is the bit offset into `input`; `arrow_offset` is the logical Arrow array offset
// to preserve in the output. Bits outside `[arrow_offset, arrow_offset + len)` are left unset.
__device__ void arrow_validity_repack_device(const uint8_t *const input,
                                             uint8_t *const output,
                                             uint64_t len,
                                             uint64_t input_offset,
                                             uint64_t arrow_offset,
                                             uint64_t validity_bits) {
    // One worker owns a contiguous byte range. Each byte is rebuilt locally so there are no
    // cross-thread bit writes or atomics.
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t output_bytes = (validity_bits + 7) / 8;
    const uint64_t start_byte = start_elem(worker, output_bytes);
    const uint64_t stop_byte = stop_elem(worker, output_bytes);

    for (uint64_t byte_idx = start_byte; byte_idx < stop_byte; byte_idx++) {
        uint8_t byte = 0;
        for (uint64_t bit_idx = 0; bit_idx < 8; bit_idx++) {
            const uint64_t output_bit = byte_idx * 8 + bit_idx;

            // Bits before Arrow's array offset are padding from the consumer's point of view.
            // Tail bits beyond len + offset stay zero so word-at-a-time mask readers are safe.
            if (output_bit >= validity_bits || output_bit < arrow_offset) {
                continue;
            }

            // Translate the Arrow-visible output bit back to the source bitmap bit. The source
            // bitmap may start at any bit offset, while Arrow's buffer pointer is byte-addressed.
            const uint64_t input_bit = input_offset + output_bit - arrow_offset;
            if (input_bit < input_offset + len && get_bit(input, input_bit)) {
                byte |= static_cast<uint8_t>(1u << bit_idx);
            }
        }
        output[byte_idx] = byte;
    }
}

} // namespace

// CUDA entry point for validity bitmap repacking used by Arrow Device export.
extern "C" __global__ void arrow_validity_repack(const uint8_t *const input,
                                                 uint8_t *const output,
                                                 uint64_t len,
                                                 uint64_t input_offset,
                                                 uint64_t arrow_offset,
                                                 uint64_t validity_bits) {
    arrow_validity_repack_device(input, output, len, input_offset, arrow_offset, validity_bits);
}
