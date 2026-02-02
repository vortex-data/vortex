// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_fp16.h>
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

// Generate for all combinations of value types and index types
// Value types: u8, i8, u16, i16, u32, i32, u64, i64
// Index types: u8, u16, u32, u64 (codes are typically unsigned)

#define GENERATE_DICT_KERNELS_FOR_VALUE(value_suffix, ValueType) \
    GENERATE_DICT_KERNEL(value_suffix, ValueType, u8, uint8_t) \
    GENERATE_DICT_KERNEL(value_suffix, ValueType, u16, uint16_t) \
    GENERATE_DICT_KERNEL(value_suffix, ValueType, u32, uint32_t) \
    GENERATE_DICT_KERNEL(value_suffix, ValueType, u64, uint64_t)

GENERATE_DICT_KERNELS_FOR_VALUE(u8, uint8_t)
GENERATE_DICT_KERNELS_FOR_VALUE(i8, int8_t)
GENERATE_DICT_KERNELS_FOR_VALUE(u16, uint16_t)
GENERATE_DICT_KERNELS_FOR_VALUE(i16, int16_t)
GENERATE_DICT_KERNELS_FOR_VALUE(u32, uint32_t)
GENERATE_DICT_KERNELS_FOR_VALUE(i32, int32_t)
GENERATE_DICT_KERNELS_FOR_VALUE(u64, uint64_t)
GENERATE_DICT_KERNELS_FOR_VALUE(i64, int64_t)

// Float types
GENERATE_DICT_KERNELS_FOR_VALUE(f16, __half)
GENERATE_DICT_KERNELS_FOR_VALUE(f32, float)
GENERATE_DICT_KERNELS_FOR_VALUE(f64, double)

// Decimal types (128-bit and 256-bit)
GENERATE_DICT_KERNELS_FOR_VALUE(i128, int128_t)
GENERATE_DICT_KERNELS_FOR_VALUE(i256, int256_t)
