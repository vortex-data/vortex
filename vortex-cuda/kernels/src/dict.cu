// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "config.cuh"
#include "types.cuh"

template <typename ValueT, typename IndexT>
__device__ void dict_kernel(const IndexT *const __restrict codes,
                            uint64_t codes_len,
                            const ValueT *const __restrict values,
                            ValueT *const __restrict output) {
    const uint32_t elements_per_block = blockDim.x * ELEMENTS_PER_THREAD;
    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * elements_per_block;
    const uint64_t block_end =
        (block_start + elements_per_block < codes_len) ? (block_start + elements_per_block) : codes_len;

    for (uint64_t idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
        const IndexT code = codes[idx];
        output[idx] = values[code];
    }
}

// Nullable dictionary values require gathering validity through the row codes:
//
//     row_validity[i] = values_validity[codes[i]]
//
// Vortex represents this gather lazily as `Dict<bool>`; null codes are carried by the resulting
// boolean array's own validity. Consequently, this kernel may materialize validity for a
// dictionary whose logical values are strings or another non-boolean type.
//
// Vortex booleans are bit-packed, so fixed-width `output[i] = values[codes[i]]` semantics do not
// apply. Assigning each complete output byte to one thread avoids races between individual bit
// writes.
template <typename IndexT>
__device__ void dict_bool_kernel(const IndexT *const __restrict codes,
                                 uint64_t codes_len,
                                 const uint8_t *const __restrict values,
                                 uint64_t values_bit_offset,
                                 uint8_t *const __restrict output) {
    const uint64_t output_len = (codes_len + 7) / 8;
    const uint32_t elements_per_block = blockDim.x * ELEMENTS_PER_THREAD;
    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * elements_per_block;
    const uint64_t block_end =
        (block_start + elements_per_block < output_len) ? (block_start + elements_per_block) : output_len;

    for (uint64_t output_idx = block_start + threadIdx.x; output_idx < block_end; output_idx += blockDim.x) {
        const uint64_t row_start = output_idx * 8;
        uint8_t packed = 0;
#pragma unroll
        for (uint32_t bit = 0; bit < 8; ++bit) {
            const uint64_t row = row_start + bit;
            if (row < codes_len) {
                const uint64_t value_idx = values_bit_offset + static_cast<uint64_t>(codes[row]);
                const uint8_t value = (values[value_idx / 8] >> (value_idx % 8)) & 1;
                packed |= static_cast<uint8_t>(value << bit);
            }
        }
        output[output_idx] = packed;
    }
}

// Macro to generate dict kernels for all fixed-width value/index type combinations.
#define GENERATE_DICT_KERNEL(value_suffix, ValueType, index_suffix, IndexType)                               \
    extern "C" __global__ void dict_##value_suffix##_##index_suffix(                                         \
        const IndexType *const __restrict codes,                                                             \
        uint64_t codes_len,                                                                                  \
        const ValueType *const __restrict values,                                                            \
        ValueType *const __restrict output) {                                                                \
        dict_kernel<ValueType, IndexType>(codes, codes_len, values, output);                                 \
    }

// Generate dict kernel for all index types (unsigned integers) for a given value type
#define GENERATE_DICT_FOR_ALL_INDICES(value_suffix, ValueType)                                               \
    GENERATE_DICT_KERNEL(value_suffix, ValueType, u8, uint8_t)                                               \
    GENERATE_DICT_KERNEL(value_suffix, ValueType, u16, uint16_t)                                             \
    GENERATE_DICT_KERNEL(value_suffix, ValueType, u32, uint32_t)                                             \
    GENERATE_DICT_KERNEL(value_suffix, ValueType, u64, uint64_t)

#define GENERATE_DICT_BOOL_KERNEL(index_suffix, IndexType)                                                   \
    extern "C" __global__ void dict_bool_##index_suffix(const IndexType *const __restrict codes,             \
                                                        uint64_t codes_len,                                  \
                                                        const uint8_t *const __restrict values,              \
                                                        uint64_t values_bit_offset,                          \
                                                        uint8_t *const __restrict output) {                  \
        dict_bool_kernel<IndexType>(codes, codes_len, values, values_bit_offset, output);                    \
    }

// Generate fixed-width kernels for all native ptypes and decimal values.
FOR_EACH_NUMERIC(GENERATE_DICT_FOR_ALL_INDICES)

// Boolean values use a different physical layout and launch unit, but still dispatch over the
// same unsigned dictionary-code types.
GENERATE_DICT_BOOL_KERNEL(u8, uint8_t)
GENERATE_DICT_BOOL_KERNEL(u16, uint16_t)
GENERATE_DICT_BOOL_KERNEL(u32, uint32_t)
GENERATE_DICT_BOOL_KERNEL(u64, uint64_t)
