// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

template<typename ValueT>
__device__ void for_(
    ValueT *const __restrict values_in_out_array,
    ValueT reference,
    uint64_t array_len
) {
    // Each block handles 2048 elements with 64 threads. Each thread handles 32 elements.
    const uint32_t elements_per_block = 2048;
    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * elements_per_block;
    const uint64_t block_end = (block_start + elements_per_block < array_len)
                               ? (block_start + elements_per_block)
                               : array_len;

    for (uint64_t idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
        values_in_out_array[idx] = values_in_out_array[idx] + reference;
    }
}

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
