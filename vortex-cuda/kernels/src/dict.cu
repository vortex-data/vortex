// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "config.cuh"
#include "types.cuh"

template<typename ValueT, typename IndexT>
__device__ void dict_kernel(
    const IndexT *const __restrict codes,
    uint64_t codes_len,
    const ValueT *const __restrict values,
    ValueT *const __restrict output
) {
    const uint32_t elements_per_block = blockDim.x * ELEMENTS_PER_THREAD;
    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * elements_per_block;
    const uint64_t block_end = (block_start + elements_per_block < codes_len)
        ? (block_start + elements_per_block)
        : codes_len;

    for (uint64_t idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
        IndexT code = codes[idx];
        output[idx] = values[code];
    }
}

// Macro to generate dict kernels for all value/index type combinations
#define GENERATE_DICT_KERNEL(value_suffix, ValueType, index_suffix, IndexType) \
extern "C" __global__ void dict_##value_suffix##_##index_suffix( \
    const IndexType *const __restrict codes, \
    uint64_t codes_len, \
    const ValueType *const __restrict values, \
    ValueType *const __restrict output \
) { \
    dict_kernel<ValueType, IndexType>(codes, codes_len, values, output); \
}

// Generate dict kernel for all index types (unsigned integers) for a given value type
#define GENERATE_DICT_FOR_ALL_INDICES(value_suffix, ValueType) \
    GENERATE_DICT_KERNEL(value_suffix, ValueType, u8, uint8_t) \
    GENERATE_DICT_KERNEL(value_suffix, ValueType, u16, uint16_t) \
    GENERATE_DICT_KERNEL(value_suffix, ValueType, u32, uint32_t) \
    GENERATE_DICT_KERNEL(value_suffix, ValueType, u64, uint64_t)

// Generate for all native ptypes & decimal values
FOR_EACH_NUMERIC(GENERATE_DICT_FOR_ALL_INDICES)

