// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Kernel wrapper template
template<typename ValueT>
__device__ void for_(
    ValueT *const __restrict values_in_out_array,
    ValueT reference,
    uint64_t array_len
) {
    // Each block handles 2048 elements. Each thread handles 2048 / 64 contiguous elements.
    // The last block and thread are allowed to have less elements.
    const uint8_t elements_per_thread = 32;
    const uint32_t block_start = blockIdx.x * 2048;

    const uint64_t start = block_start + threadIdx.x * elements_per_thread;
    const uint64_t end = (start + elements_per_thread < array_len) ? (start + elements_per_thread) : array_len;

    for (uint64_t idx = start; idx < end; ++idx) {
        values_in_out_array[idx] = values_in_out_array[idx] + reference;
    }
}

// Macro to generate the extern "C" wrapper for each type combination
#define GENERATE_KERNEL(value_suffix, ValueType) \
extern "C" __global__ void for_##value_suffix( \
    ValueType *const __restrict values, \
    ValueType reference, \
    uint64_t array_len \
) { \
    for_(values, reference, array_len); \
}

GENERATE_KERNEL(u8, uint8_t)
GENERATE_KERNEL(u16, uint16_t)
GENERATE_KERNEL(u32, uint32_t)
GENERATE_KERNEL(u64, uint64_t)

GENERATE_KERNEL(i8, int8_t)
GENERATE_KERNEL(i16, int16_t)
GENERATE_KERNEL(i32, int32_t)
GENERATE_KERNEL(i64, int64_t)
