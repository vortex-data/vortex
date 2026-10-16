// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"

// Each block fills a contiguous run of `blockDim.x * ELEMENTS_PER_THREAD` elements with its
// threads striding by `blockDim.x`, so consecutive lanes store consecutive elements and every warp
// store is coalesced. Giving each thread its own contiguous run instead spreads one warp store over
// 32 cache lines.
template <typename ValueT>
__device__ void sequence(ValueT *const output, ValueT base, ValueT multiplier, uint64_t len) {
    const uint64_t elements_per_block = static_cast<uint64_t>(blockDim.x) * ELEMENTS_PER_THREAD;
    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * elements_per_block;
    const uint64_t block_stop = ::min(block_start + elements_per_block, len);

    for (uint64_t idx = block_start + threadIdx.x; idx < block_stop; idx += blockDim.x) {
        output[idx] = static_cast<ValueT>(idx) * multiplier + base;
    }
}

#define GENERATE_KERNEL(ValueT, suffix)                                                                      \
    extern "C" __global__ void sequence_##suffix(ValueT *const output,                                       \
                                                 ValueT base,                                                \
                                                 ValueT multiplier,                                          \
                                                 uint64_t len) {                                             \
        sequence(output, base, multiplier, len);                                                             \
    }

GENERATE_KERNEL(uint8_t, u8);
GENERATE_KERNEL(uint16_t, u16);
GENERATE_KERNEL(uint32_t, u32);
GENERATE_KERNEL(uint64_t, u64);
GENERATE_KERNEL(int8_t, i8);
GENERATE_KERNEL(int16_t, i16);
GENERATE_KERNEL(int32_t, i32);
GENERATE_KERNEL(int64_t, i64);
GENERATE_KERNEL(float, f32);
GENERATE_KERNEL(double, f64);
