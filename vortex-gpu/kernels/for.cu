// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

template<typename ValueT>
__device__ void for_(
    ValueT *__restrict values_in_out_array,
    ValueT reference
) {
    auto i = threadIdx.x;
    auto block_offset = (blockIdx.x * 1024);

    auto values_in_out = values_in_out_array + block_offset;

    const int thread_ops = 32;

    for (auto j = 0; j < thread_ops; j++) {
        auto idx = i * thread_ops + j;
        values_in_out[idx] = values_in_out[idx] + reference;
    }
}

// Macro to generate the extern "C" wrapper for each type combination
#define GENERATE_KERNEL(value_suffix, ValueType) \
extern "C" __global__ void for_v##value_suffix( \
    ValueType *__restrict values, \
    ValueType reference \
) { \
    for_(values, reference); \
}

GENERATE_KERNEL(u8, uint8_t)
GENERATE_KERNEL(u16, uint16_t)
GENERATE_KERNEL(u32, uint32_t)
GENERATE_KERNEL(u64, uint64_t)

GENERATE_KERNEL(i8, int8_t)
GENERATE_KERNEL(i16, int16_t)
GENERATE_KERNEL(i32, int32_t)
GENERATE_KERNEL(i64, int64_t)
