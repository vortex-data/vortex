// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "bit_unpack_8_lanes.cuh"
#include "bit_unpack_16_lanes.cuh"
#include "bit_unpack_32_lanes.cuh"
#include "bit_unpack_64_lanes.cuh"
#include "patches.h"

/// Number of 16-row slices each FastLanes lane is split into.
///
/// A 1024-element chunk has 1024 / bits lanes of `bits` rows each. 8- and
/// 16-bit chunks have 128 and 64 lanes, enough to keep a 64-thread block
/// busy, so their lanes stay whole. 32- and 64-bit chunks have only 32 and 16
/// lanes, so their lanes are split into 2 and 4 slices that separate threads
/// decode.
template <typename T>
constexpr uint32_t LANE_SLICES = sizeof(T) >= 4 ? sizeof(T) * 8 / 16 : 1;

/// Decodes rows `[slice * 16, (slice + 1) * 16)` of one lane of a FastLanes
/// chunk into `output_buffer`, or the whole lane when `LANE_SLICES<T> == 1`.
///
/// # Parameters
///
/// * `packed_chunk` - Pointer to the start of the packed data chunk
/// * `output_buffer` - Pointer to the output buffer to which the unpacked data is written
/// * `lane` - Lane index within the chunk (used to determine which packed words to process)
/// * `bit_width` - Number of bits with which each value is encoded
/// * `slice` - Slice of the lane to decode, less than `LANE_SLICES<T>`
template <typename T>
__device__ inline void bit_unpack_lane_slice(const T *__restrict packed_chunk,
                                             T *__restrict output_buffer,
                                             unsigned int lane,
                                             uint32_t bit_width,
                                             uint32_t slice) {
    if constexpr (sizeof(T) == 1) {
        bit_unpack_8_lane(packed_chunk, output_buffer, 0, lane, bit_width);
    } else if constexpr (sizeof(T) == 2) {
        bit_unpack_16_lane(packed_chunk, output_buffer, 0, lane, bit_width);
    } else if constexpr (sizeof(T) == 4) {
        bit_unpack_32_lane_slice(packed_chunk, output_buffer, 0, lane, bit_width, slice);
    } else {
        static_assert(sizeof(T) == 8);
        bit_unpack_64_lane_slice(packed_chunk, output_buffer, 0, lane, bit_width, slice);
    }
}
