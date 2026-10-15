// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Comparison operators, numbered as `CompareOp` in `exact_filter.rs`.
enum CompareOp : uint32_t {
    COMPARE_EQ = 0,
    COMPARE_NOT_EQ = 1,
    COMPARE_GT = 2,
    COMPARE_GTE = 3,
    COMPARE_LT = 4,
    COMPARE_LTE = 5,
};

template <typename T>
__device__ inline bool compare(T value, T literal, uint32_t op) {
    switch (op) {
    case COMPARE_EQ:
        return value == literal;
    case COMPARE_NOT_EQ:
        return value != literal;
    case COMPARE_GT:
        return value > literal;
    case COMPARE_GTE:
        return value >= literal;
    case COMPARE_LT:
        return value < literal;
    case COMPARE_LTE:
        return value <= literal;
    default:
        __builtin_unreachable();
    }
}

// Writes bit `i` of `words` (LSB first) as `values[i] <op> literal`, or ANDs it into the bit
// already there if `combine` is set, so conjunctions accumulate in one buffer. Floating-point
// comparisons follow IEEE 754: NaN compares false, as in the consumers that filter afterwards.
//
// Launch config: grid_dim = (array_len / 2048, 1, 1), block_dim = (64, 1, 1). Each step of a
// warp covers 32 consecutive elements, which its ballot turns into one word.
template <typename T>
__device__ void compare_bits(const T *__restrict values,
                             uint64_t array_len,
                             uint32_t op,
                             T literal,
                             uint32_t *__restrict words,
                             uint32_t combine) {
    constexpr uint32_t elements_per_block = 2048;
    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * elements_per_block;
    const uint32_t lane = threadIdx.x % warpSize;
    for (uint32_t offset = 0; offset < elements_per_block; offset += blockDim.x) {
        const uint64_t idx = block_start + offset + threadIdx.x;
        const bool keep = idx < array_len && compare(values[idx], literal, op);
        const uint32_t bits = __ballot_sync(0xffffffff, keep);
        const uint64_t first = idx - lane;
        if (lane == 0 && first < array_len) {
            const uint64_t word = first / 32;
            words[word] = combine ? (words[word] & bits) : bits;
        }
    }
}

#define GENERATE_COMPARE_KERNEL(suffix, Type)                                                                \
    extern "C" __global__ void compare_##suffix(const Type *__restrict values,                               \
                                                uint64_t array_len,                                          \
                                                uint32_t op,                                                 \
                                                Type literal,                                                \
                                                uint32_t *__restrict words,                                  \
                                                uint32_t combine) {                                          \
        compare_bits<Type>(values, array_len, op, literal, words, combine);                                  \
    }

GENERATE_COMPARE_KERNEL(u8, uint8_t)
GENERATE_COMPARE_KERNEL(u16, uint16_t)
GENERATE_COMPARE_KERNEL(u32, uint32_t)
GENERATE_COMPARE_KERNEL(u64, uint64_t)
GENERATE_COMPARE_KERNEL(i8, int8_t)
GENERATE_COMPARE_KERNEL(i16, int16_t)
GENERATE_COMPARE_KERNEL(i32, int32_t)
GENERATE_COMPARE_KERNEL(i64, int64_t)
GENERATE_COMPARE_KERNEL(f32, float)
GENERATE_COMPARE_KERNEL(f64, double)
