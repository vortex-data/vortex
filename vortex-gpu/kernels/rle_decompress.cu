// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

template<typename IndicesT, typename ValueT, typename OffsetsT>
__device__ void rle_decompress(
    const IndicesT *__restrict indices_array,
    const ValueT *__restrict values_array,
    const OffsetsT *__restrict offsets,
    ValueT *__restrict values_out
) {
    auto i = threadIdx.x;
    auto block_offset = blockIdx.x * 1024;

    auto indices = indices_array + block_offset;
    auto out = values_out + block_offset;
    auto values = values_array + offsets[blockIdx.x];

    const int thread_ops = 32;

    for (auto j = 0; j < thread_ops; j++) {
        auto idx = i * thread_ops + j;
        out[idx] = values[indices[idx]];
    }
}

// Macro to generate the extern "C" wrapper for each type combination
#define GENERATE_KERNEL(indices_suffix, value_suffix, offsets_suffix, IndicesType, ValueType, OffsetsType) \
extern "C" __global__ void rle_decompress_i##indices_suffix##_v##value_suffix##_o##offsets_suffix( \
    const IndicesType *__restrict indices_array, \
    const ValueType *__restrict values_array, \
    const OffsetsType *__restrict offsets, \
    ValueType *__restrict values_out \
) { \
    rle_decompress(indices_array, values_array, offsets, values_out); \
}

// Generate all combinations
// Unsigned types
GENERATE_KERNEL(u8, u8,  u8, uint8_t,  uint8_t,  uint8_t)
GENERATE_KERNEL(u8, u8,  u16, uint8_t, uint8_t,  uint16_t)
GENERATE_KERNEL(u8, u8,  u32, uint8_t, uint8_t,  uint32_t)
GENERATE_KERNEL(u8, u8,  u64, uint8_t, uint8_t,  uint64_t)

GENERATE_KERNEL(u8, u16, u8, uint8_t,  uint16_t, uint8_t)
GENERATE_KERNEL(u8, u16, u16, uint8_t, uint16_t, uint16_t)
GENERATE_KERNEL(u8, u16, u32, uint8_t, uint16_t, uint32_t)
GENERATE_KERNEL(u8, u16, u64, uint8_t, uint16_t, uint64_t)

GENERATE_KERNEL(u8, u32, u8, uint8_t,  uint32_t, uint8_t)
GENERATE_KERNEL(u8, u32, u16, uint8_t, uint32_t, uint16_t)
GENERATE_KERNEL(u8, u32, u32, uint8_t, uint32_t, uint32_t)
GENERATE_KERNEL(u8, u32, u64, uint8_t, uint32_t, uint64_t)

GENERATE_KERNEL(u8, u64, u8, uint8_t,  uint64_t, uint8_t)
GENERATE_KERNEL(u8, u64, u16, uint8_t, uint64_t, uint16_t)
GENERATE_KERNEL(u8, u64, u32, uint8_t, uint64_t, uint32_t)
GENERATE_KERNEL(u8, u64, u64, uint8_t, uint64_t, uint64_t)

GENERATE_KERNEL(u16, u8,  u8, uint16_t,  uint8_t,  uint8_t)
GENERATE_KERNEL(u16, u8,  u16, uint16_t, uint8_t,  uint16_t)
GENERATE_KERNEL(u16, u8,  u32, uint16_t, uint8_t,  uint32_t)
GENERATE_KERNEL(u16, u8,  u64, uint16_t, uint8_t,  uint64_t)

GENERATE_KERNEL(u16, u16, u8, uint16_t,  uint16_t, uint8_t)
GENERATE_KERNEL(u16, u16, u16, uint16_t, uint16_t, uint16_t)
GENERATE_KERNEL(u16, u16, u32, uint16_t, uint16_t, uint32_t)
GENERATE_KERNEL(u16, u16, u64, uint16_t, uint16_t, uint64_t)

GENERATE_KERNEL(u16, u32, u8, uint16_t,  uint32_t, uint8_t)
GENERATE_KERNEL(u16, u32, u16, uint16_t, uint32_t, uint16_t)
GENERATE_KERNEL(u16, u32, u32, uint16_t, uint32_t, uint32_t)
GENERATE_KERNEL(u16, u32, u64, uint16_t, uint32_t, uint64_t)

GENERATE_KERNEL(u16, u64, u8, uint16_t,  uint64_t, uint8_t)
GENERATE_KERNEL(u16, u64, u16, uint16_t, uint64_t, uint16_t)
GENERATE_KERNEL(u16, u64, u32, uint16_t, uint64_t, uint32_t)
GENERATE_KERNEL(u16, u64, u64, uint16_t, uint64_t, uint64_t)

// Signed types
GENERATE_KERNEL(u8, i8,  u8, uint8_t,  int8_t,  uint8_t)
GENERATE_KERNEL(u8, i8,  u16, uint8_t, int8_t,  uint16_t)
GENERATE_KERNEL(u8, i8,  u32, uint8_t, int8_t,  uint32_t)
GENERATE_KERNEL(u8, i8,  u64, uint8_t, int8_t,  uint64_t)

GENERATE_KERNEL(u8, i16, u8, uint8_t,  int16_t, uint8_t)
GENERATE_KERNEL(u8, i16, u16, uint8_t, int16_t, uint16_t)
GENERATE_KERNEL(u8, i16, u32, uint8_t, int16_t, uint32_t)
GENERATE_KERNEL(u8, i16, u64, uint8_t, int16_t, uint64_t)

GENERATE_KERNEL(u8, i32, u8, uint8_t,  int32_t, uint8_t)
GENERATE_KERNEL(u8, i32, u16, uint8_t, int32_t, uint16_t)
GENERATE_KERNEL(u8, i32, u32, uint8_t, int32_t, uint32_t)
GENERATE_KERNEL(u8, i32, u64, uint8_t, int32_t, uint64_t)

GENERATE_KERNEL(u8, i64, u8, uint8_t,  int64_t, uint8_t)
GENERATE_KERNEL(u8, i64, u16, uint8_t, int64_t, uint16_t)
GENERATE_KERNEL(u8, i64, u32, uint8_t, int64_t, uint32_t)
GENERATE_KERNEL(u8, i64, u64, uint8_t, int64_t, uint64_t)

GENERATE_KERNEL(u16, i8,  u8, uint16_t,  int8_t,  uint8_t)
GENERATE_KERNEL(u16, i8,  u16, uint16_t, int8_t,  uint16_t)
GENERATE_KERNEL(u16, i8,  u32, uint16_t, int8_t,  uint32_t)
GENERATE_KERNEL(u16, i8,  u64, uint16_t, int8_t,  uint64_t)

GENERATE_KERNEL(u16, i16, u8, uint16_t,  int16_t, uint8_t)
GENERATE_KERNEL(u16, i16, u16, uint16_t, int16_t, uint16_t)
GENERATE_KERNEL(u16, i16, u32, uint16_t, int16_t, uint32_t)
GENERATE_KERNEL(u16, i16, u64, uint16_t, int16_t, uint64_t)

GENERATE_KERNEL(u16, i32, u8, uint16_t,  int32_t, uint8_t)
GENERATE_KERNEL(u16, i32, u16, uint16_t, int32_t, uint16_t)
GENERATE_KERNEL(u16, i32, u32, uint16_t, int32_t, uint32_t)
GENERATE_KERNEL(u16, i32, u64, uint16_t, int32_t, uint64_t)

GENERATE_KERNEL(u16, i64, u8, uint16_t,  int64_t, uint8_t)
GENERATE_KERNEL(u16, i64, u16, uint16_t, int64_t, uint16_t)
GENERATE_KERNEL(u16, i64, u32, uint16_t, int64_t, uint32_t)
GENERATE_KERNEL(u16, i64, u64, uint16_t, int64_t, uint64_t)

// Float types
GENERATE_KERNEL(u8, f32, u8, uint8_t,  float, uint8_t)
GENERATE_KERNEL(u8, f32, u16, uint8_t, float, uint16_t)
GENERATE_KERNEL(u8, f32, u32, uint8_t, float, uint32_t)
GENERATE_KERNEL(u8, f32, u64, uint8_t, float, uint64_t)

GENERATE_KERNEL(u8, f64, u8, uint8_t,  double, uint8_t)
GENERATE_KERNEL(u8, f64, u16, uint8_t, double, uint16_t)
GENERATE_KERNEL(u8, f64, u32, uint8_t, double, uint32_t)
GENERATE_KERNEL(u8, f64, u64, uint8_t, double, uint64_t)

GENERATE_KERNEL(u16, f32, u8, uint16_t,  float, uint8_t)
GENERATE_KERNEL(u16, f32, u16, uint16_t, float, uint16_t)
GENERATE_KERNEL(u16, f32, u32, uint16_t, float, uint32_t)
GENERATE_KERNEL(u16, f32, u64, uint16_t, float, uint64_t)

GENERATE_KERNEL(u16, f64, u8, uint16_t,  double, uint8_t)
GENERATE_KERNEL(u16, f64, u16, uint16_t, double, uint16_t)
GENERATE_KERNEL(u16, f64, u32, uint16_t, double, uint32_t)
GENERATE_KERNEL(u16, f64, u64, uint16_t, double, uint64_t)
