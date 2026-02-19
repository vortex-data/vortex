// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "bit_unpack_8.cu"
#include "bit_unpack_16.cu"
#include "bit_unpack_32.cu"
#include "bit_unpack_64.cu"

/// Decodes a single lane of packed data.
///
/// This is a template function that dispatches to the appropriate bitunpack kernel
/// for the element type. Each lane represents a chunk of packed data that must be
/// unpacked in parallel by multiple threads within a block.
///
/// # Parameters
///
/// * `packed_chunk` - Pointer to the start of the packed data chunk
/// * `output_buffer` - Pointer to the output buffer to which the unpacked data is written
/// * `lane` - Lane index within the block (used to determine which packed words to process)
/// * `bit_width` - Number of bits with which each value is encoded
template <typename T>
__device__ inline void bit_unpack_lane(const T *__restrict packed_chunk,
                                       T *__restrict output_buffer,
                                       unsigned int lane,
                                       uint32_t bit_width);

/// Template specializations for `bitunpack_lane_to_smem` for different integer types.
///
/// Generates template specializations for each supported integer size (8, 16, 32, 64 bits).
#define BIT_UNPACK_LANE(bits)                                                                                \
    template <>                                                                                              \
    __device__ inline void bit_unpack_lane<uint##bits##_t>(const uint##bits##_t *in,                         \
                                                           uint##bits##_t *out,                              \
                                                           unsigned int lane,                                \
                                                           uint32_t bw) {                                    \
        bit_unpack_##bits##_lane(in, out, lane, bw);                                                         \
    }

BIT_UNPACK_LANE(8)
BIT_UNPACK_LANE(16)
BIT_UNPACK_LANE(32)
BIT_UNPACK_LANE(64)
