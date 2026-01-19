// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

template<typename ValueT>
__device__ void for_kernel(
    ValueT *const __restrict values_in_out_array,
    ValueT reference,
    uint64_t array_len
) {
    const uint32_t elements_per_block = 2048;
    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * elements_per_block;
    const uint64_t block_end = (block_start + elements_per_block < array_len)
    ? (block_start + elements_per_block)
    : array_len;


    // Determine the number of loop iterations at compile time. Unroll to process 16 bytes per iteration.
    constexpr auto VALUES_PER_LOOP = 16 / sizeof(ValueT);
    const auto block_start_vec = block_start / VALUES_PER_LOOP;
    const auto block_end_vec = block_end / VALUES_PER_LOOP;

    for (uint64_t idx = block_start_vec + threadIdx.x; idx < block_end_vec; idx += blockDim.x) {
        uint64_t base_idx = idx * VALUES_PER_LOOP;

        // The loop can be unrolled, as `values per loop` is `constexpr`.
        #pragma unroll
        for (uint64_t i = 0; i < VALUES_PER_LOOP; ++i) {
            values_in_out_array[base_idx + i] += reference;
        }
    }

    uint64_t remaining_start = block_end_vec * VALUES_PER_LOOP;
    for (uint64_t idx = remaining_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
        values_in_out_array[idx] = values_in_out_array[idx] + reference;
    }
}

// Macro for all types
#define GENERATE_FOR_KERNEL(value_suffix, ValueType) \
extern "C" __global__ void for_##value_suffix( \
    ValueType *const __restrict values, \
    ValueType reference, \
    uint64_t array_len \
) { \
    for_kernel<ValueType>(values, reference, array_len); \
}

GENERATE_FOR_KERNEL(u8, uint8_t)
GENERATE_FOR_KERNEL(i8, int8_t)

GENERATE_FOR_KERNEL(u16, uint16_t)
GENERATE_FOR_KERNEL(i16, int16_t)

GENERATE_FOR_KERNEL(u32, uint32_t)
GENERATE_FOR_KERNEL(i32, int32_t)

GENERATE_FOR_KERNEL(u64, uint64_t)
GENERATE_FOR_KERNEL(i64, int64_t)
