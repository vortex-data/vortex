// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include "types.cuh"
#include <stdint.h>
#include <type_traits>

// Widen decimal values buffers to a wider physical width. Two callers use these kernels:
//   - The Arrow export casts to the width that the Arrow schema fixes: 4 bytes for Decimal32,
//     8 bytes for Decimal64, 16 bytes for Decimal128, and 32 bytes for Decimal256.
//   - The DecimalByteParts decoder casts a narrow MSP to the canonical width of its decimal
//     dtype, from 2 to 32 bytes.
//
// The Rust side rejects narrowing casts, because detecting overflow on the device would require
// synchronizing an overflow flag back to the host.

// Low 64-bit conversion for 16-, 32- and 64-bit outputs.
template <typename Input>
__device__ __forceinline__ int64_t decimal_to_i64(Input value) {
    if constexpr (std::is_same_v<Input, int128_t>) {
        return value.lo;
    } else if constexpr (std::is_same_v<Input, int256_t>) {
        return value.parts[0];
    } else {
        return static_cast<int64_t>(value);
    }
}

// 128-bit conversion for Decimal128 outputs.
template <typename Input>
__device__ __forceinline__ int128_t decimal_to_i128(Input value) {
    if constexpr (std::is_same_v<Input, int128_t>) {
        return value;
    } else if constexpr (std::is_same_v<Input, int256_t>) {
        return int128_t {value.parts[0], value.parts[1]};
    } else {
        const int64_t lo = static_cast<int64_t>(value);
        const int64_t hi = value < 0 ? -1 : 0;
        return int128_t {lo, hi};
    }
}

// Convert one value to the output physical width.
template <typename Output, typename Input>
__device__ __forceinline__ Output decimal_cast_value(Input value) {
    if constexpr (std::is_same_v<Output, int16_t>) {
        return static_cast<int16_t>(decimal_to_i64(value));
    } else if constexpr (std::is_same_v<Output, int32_t>) {
        return static_cast<int32_t>(decimal_to_i64(value));
    } else if constexpr (std::is_same_v<Output, int64_t>) {
        return decimal_to_i64(value);
    } else if constexpr (std::is_same_v<Output, int128_t>) {
        return decimal_to_i128(value);
    } else {
        static_assert(std::is_same_v<Output, int256_t>);
        if constexpr (std::is_same_v<Input, int256_t>) {
            return value;
        } else {
            const int128_t value128 = decimal_to_i128(value);
            const int64_t sign = value128.hi < 0 ? -1 : 0;
            return int256_t {{value128.lo, value128.hi, sign, sign}};
        }
    }
}

// Cast a contiguous values buffer to the output physical width.
template <typename Input, typename Output>
__device__ void
decimal_cast_device(const Input *__restrict input, Output *__restrict output, uint64_t array_len) {
    const uint64_t elements_per_block = blockDim.x * ELEMENTS_PER_THREAD;
    const uint64_t block_start = blockIdx.x * elements_per_block;
    const uint64_t block_stop = min(block_start + elements_per_block, array_len);

    for (uint64_t idx = block_start + threadIdx.x; idx < block_stop; idx += blockDim.x) {
        output[idx] = decimal_cast_value<Output>(input[idx]);
    }
}

// Generate the i16, i32, i64, i128 and i256 cast kernels for one input storage type.
#define GENERATE_DECIMAL_CAST_KERNELS(input_suffix, InputType)                                               \
    extern "C" __global__ void decimal_cast_##input_suffix##_i16(const InputType *__restrict input,          \
                                                                 int16_t *__restrict output,                 \
                                                                 uint64_t array_len) {                       \
        decimal_cast_device(input, output, array_len);                                                       \
    }                                                                                                        \
    extern "C" __global__ void decimal_cast_##input_suffix##_i32(const InputType *__restrict input,          \
                                                                 int32_t *__restrict output,                 \
                                                                 uint64_t array_len) {                       \
        decimal_cast_device(input, output, array_len);                                                       \
    }                                                                                                        \
    extern "C" __global__ void decimal_cast_##input_suffix##_i64(const InputType *__restrict input,          \
                                                                 int64_t *__restrict output,                 \
                                                                 uint64_t array_len) {                       \
        decimal_cast_device(input, output, array_len);                                                       \
    }                                                                                                        \
    extern "C" __global__ void decimal_cast_##input_suffix##_i128(const InputType *__restrict input,         \
                                                                  int128_t *__restrict output,               \
                                                                  uint64_t array_len) {                      \
        decimal_cast_device(input, output, array_len);                                                       \
    }                                                                                                        \
    extern "C" __global__ void decimal_cast_##input_suffix##_i256(const InputType *__restrict input,         \
                                                                  int256_t *__restrict output,               \
                                                                  uint64_t array_len) {                      \
        decimal_cast_device(input, output, array_len);                                                       \
    }

FOR_EACH_SIGNED_INT(GENERATE_DECIMAL_CAST_KERNELS)
FOR_EACH_LARGE_DECIMAL(GENERATE_DECIMAL_CAST_KERNELS)
