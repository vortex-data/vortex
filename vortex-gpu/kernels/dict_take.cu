// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

template<typename CodeT, typename ValueT>
__device__ void dict_take(
    const CodeT *__restrict codes_array,
    const ValueT *__restrict values,
    ValueT *__restrict values_out
) {
    auto i = threadIdx.x;
    auto block_offset = (blockIdx.x * 1024);

    auto codes = codes_array + block_offset;
    auto out = values_out + block_offset;

    const int thread_ops = 32;

    for (auto j = 0; j < thread_ops; j++) {
            auto idx = i * thread_ops + j;
            out[idx] = values[codes[idx]];
    }
}

template<typename CodeT, typename ValueT>
__device__ void dict_take_masked(
    const CodeT *__restrict codes_array,
    const ValueT *__restrict values,
    const uint32_t *__restrict mask_array,
    ValueT *__restrict values_out
) {
    auto i = threadIdx.x;
    auto block_offset = (blockIdx.x * 1024);
    auto mask_block_offset = (blockIdx.x * (1024 / 32));

    auto codes = codes_array + block_offset;
    auto mask = mask_array + mask_block_offset;
    auto out = values_out + block_offset;

    const int thread_ops = 32;

    for (auto j = 0; j < thread_ops; j++) {
        if (mask[i] >> j & 1) {
            auto idx = i * thread_ops + j;
            auto code = codes[idx];
            out[idx] = values[code];
        }
    }
}

// Macro to generate the extern "C" wrapper for each type combination
#define GENERATE_KERNEL(code_suffix, value_suffix, CodeType, ValueType) \
extern "C" __global__ void dict_take_c##code_suffix##_v##value_suffix( \
    const CodeType *__restrict codes_array, \
    const ValueType *__restrict values, \
    ValueType *__restrict values_out \
) { \
    dict_take(codes_array, values, values_out); \
} \
extern "C" __global__ void dict_take_masked_c##code_suffix##_v##value_suffix( \
    const CodeType *__restrict codes_array, \
    const ValueType *__restrict values, \
    const uint32_t *__restrict mask, \
    ValueType *__restrict values_out \
) { \
    dict_take_masked(codes_array, values, mask, values_out); \
}

// Generate all combinations
// Unsigned types
GENERATE_KERNEL(u8,  u8,  uint8_t,  uint8_t)
GENERATE_KERNEL(u8,  u16, uint8_t,  uint16_t)
GENERATE_KERNEL(u8,  u32, uint8_t,  uint32_t)
GENERATE_KERNEL(u8,  u64, uint8_t,  uint64_t)
GENERATE_KERNEL(u8,  i8,  uint8_t,  int8_t)
GENERATE_KERNEL(u8,  i16, uint8_t,  int16_t)
GENERATE_KERNEL(u8,  i32, uint8_t,  int32_t)
GENERATE_KERNEL(u8,  i64, uint8_t,  int64_t)

GENERATE_KERNEL(u16, u8,  uint16_t, uint8_t)
GENERATE_KERNEL(u16, u16, uint16_t, uint16_t)
GENERATE_KERNEL(u16, u32, uint16_t, uint32_t)
GENERATE_KERNEL(u16, u64, uint16_t, uint64_t)
GENERATE_KERNEL(u16, i8,  uint16_t, int8_t)
GENERATE_KERNEL(u16, i16, uint16_t, int16_t)
GENERATE_KERNEL(u16, i32, uint16_t, int32_t)
GENERATE_KERNEL(u16, i64, uint16_t, int64_t)

GENERATE_KERNEL(u32, u8,  uint32_t, uint8_t)
GENERATE_KERNEL(u32, u16, uint32_t, uint16_t)
GENERATE_KERNEL(u32, u32, uint32_t, uint32_t)
GENERATE_KERNEL(u32, u64, uint32_t, uint64_t)
GENERATE_KERNEL(u32, i8,  uint32_t, int8_t)
GENERATE_KERNEL(u32, i16, uint32_t, int16_t)
GENERATE_KERNEL(u32, i32, uint32_t, int32_t)
GENERATE_KERNEL(u32, i64, uint32_t, int64_t)

GENERATE_KERNEL(u64, u8,  uint64_t, uint8_t)
GENERATE_KERNEL(u64, u16, uint64_t, uint16_t)
GENERATE_KERNEL(u64, u32, uint64_t, uint32_t)
GENERATE_KERNEL(u64, u64, uint64_t, uint64_t)
GENERATE_KERNEL(u64, i8,  uint64_t, int8_t)
GENERATE_KERNEL(u64, i16, uint64_t, int16_t)
GENERATE_KERNEL(u64, i32, uint64_t, int32_t)
GENERATE_KERNEL(u64, i64, uint64_t, int64_t)

// Signed types
GENERATE_KERNEL(i8,  u8,  int8_t,  uint8_t)
GENERATE_KERNEL(i8,  u16, int8_t,  uint16_t)
GENERATE_KERNEL(i8,  u32, int8_t,  uint32_t)
GENERATE_KERNEL(i8,  u64, int8_t,  uint64_t)
GENERATE_KERNEL(i8,  i8,  int8_t,  int8_t)
GENERATE_KERNEL(i8,  i16, int8_t,  int16_t)
GENERATE_KERNEL(i8,  i32, int8_t,  int32_t)
GENERATE_KERNEL(i8,  i64, int8_t,  int64_t)

GENERATE_KERNEL(i16, u8,  int16_t, uint8_t)
GENERATE_KERNEL(i16, u16, int16_t, uint16_t)
GENERATE_KERNEL(i16, u32, int16_t, uint32_t)
GENERATE_KERNEL(i16, u64, int16_t, uint64_t)
GENERATE_KERNEL(i16, i8,  int16_t, int8_t)
GENERATE_KERNEL(i16, i16, int16_t, int16_t)
GENERATE_KERNEL(i16, i32, int16_t, int32_t)
GENERATE_KERNEL(i16, i64, int16_t, int64_t)

GENERATE_KERNEL(i32, u8,  int32_t, uint8_t)
GENERATE_KERNEL(i32, u16, int32_t, uint16_t)
GENERATE_KERNEL(i32, u32, int32_t, uint32_t)
GENERATE_KERNEL(i32, u64, int32_t, uint64_t)
GENERATE_KERNEL(i32, i8,  int32_t, int8_t)
GENERATE_KERNEL(i32, i16, int32_t, int16_t)
GENERATE_KERNEL(i32, i32, int32_t, int32_t)
GENERATE_KERNEL(i32, i64, int32_t, int64_t)

GENERATE_KERNEL(i64, u8,  int64_t, uint8_t)
GENERATE_KERNEL(i64, u16, int64_t, uint16_t)
GENERATE_KERNEL(i64, u32, int64_t, uint32_t)
GENERATE_KERNEL(i64, u64, int64_t, uint64_t)
GENERATE_KERNEL(i64, i8,  int64_t, int8_t)
GENERATE_KERNEL(i64, i16, int64_t, int16_t)
GENERATE_KERNEL(i64, i32, int64_t, int32_t)
GENERATE_KERNEL(i64, i64, int64_t, int64_t)