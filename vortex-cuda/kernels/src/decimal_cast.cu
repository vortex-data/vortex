// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include "types.cuh"
#include <stdint.h>
#include <type_traits>

// Arrow decimal schemas fix the physical values buffer width:
//   - Decimal32: 4 bytes per value.
//   - Decimal64: 8 bytes per value.
//   - Decimal128: 16 bytes per value.
//   - Decimal256: 32 bytes per value.
//
// Vortex storage width can differ, so export casts to the schema-implied width.
// Rust-side export rejects narrowing casts because detecting overflow on-device
// would require synchronizing an overflow flag back to the host.

// Low 64-bit conversion for Decimal32/64 outputs.
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

// Convert one value to the Arrow schema's physical width.
template <typename Output, typename Input>
__device__ __forceinline__ Output decimal_cast_value(Input value) {
    if constexpr (std::is_same_v<Output, int32_t>) {
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

// Cast a contiguous values buffer to the Arrow schema's physical width.
template <typename Input, typename Output>
__device__ void
decimal_cast_device(const Input *__restrict input, Output *__restrict output, uint64_t array_len) {
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t startElem = start_elem(worker, array_len);
    const uint64_t stopElem = stop_elem(worker, array_len);

    if (startElem >= array_len) {
        return;
    }

    for (uint64_t idx = startElem; idx < stopElem; idx++) {
        output[idx] = decimal_cast_value<Output>(input[idx]);
    }
}

// Generate Decimal32/64/128/256 cast kernels for one input storage type.
#define GENERATE_DECIMAL_CAST_KERNELS(input_suffix, InputType)                                               \
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
