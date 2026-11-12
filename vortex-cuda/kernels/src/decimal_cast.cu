// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include "types.cuh"
#include <stdint.h>
#include <type_traits>

// Widen decimal values buffers to a wider physical width. Two callers use these kernels:
//   - The Arrow export casts to the width that the Arrow schema fixes: 4 bytes for Decimal32,
//     8 bytes for Decimal64, 16 bytes for Decimal128, and 32 bytes for Decimal256.
//   - The Widen executor casts a narrow decimal child to the canonical width of its decimal
//     dtype, from 2 to 32 bytes.
//
// Only widening pairs are instantiated. A narrowing cast can overflow, and detecting that on the
// device would require synchronizing an overflow flag back to the host, so the Rust side rejects
// narrowing before it loads a kernel.

// Sign-extend one value to the wider output width in a single step, with no intermediate width.
// Native outputs use one `static_cast`. The 128-bit and 256-bit outputs keep the input's words
// and fill the upper words with its sign.
template <typename Output, typename Input>
__device__ __forceinline__ Output decimal_cast_value(Input value) {
    static_assert(sizeof(Input) < sizeof(Output), "decimal casts on the GPU only widen");

    if constexpr (std::is_same_v<Output, int128_t>) {
        static_assert(std::is_integral_v<Input> && std::is_signed_v<Input>);
        const int64_t lo = static_cast<int64_t>(value);
        return int128_t {lo, lo < 0 ? -1 : 0};
    } else if constexpr (std::is_same_v<Output, int256_t> && std::is_same_v<Input, int128_t>) {
        const int64_t sign = value.hi < 0 ? -1 : 0;
        return int256_t {{value.lo, value.hi, sign, sign}};
    } else if constexpr (std::is_same_v<Output, int256_t>) {
        static_assert(std::is_integral_v<Input> && std::is_signed_v<Input>);
        const int64_t lo = static_cast<int64_t>(value);
        const int64_t sign = lo < 0 ? -1 : 0;
        return int256_t {{lo, sign, sign, sign}};
    } else {
        static_assert(std::is_integral_v<Input> && std::is_signed_v<Input>);
        static_assert(std::is_integral_v<Output> && std::is_signed_v<Output>);
        return static_cast<Output>(value);
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

#define GENERATE_DECIMAL_CAST(input_suffix, InputType, output_suffix, OutputType)                            \
    extern "C" __global__ void decimal_cast_##input_suffix##_##output_suffix(                                \
        const InputType *__restrict input,                                                                   \
        OutputType *__restrict output,                                                                       \
        uint64_t array_len) {                                                                                \
        decimal_cast_device(input, output, array_len);                                                       \
    }

GENERATE_DECIMAL_CAST(i8, int8_t, i16, int16_t)
GENERATE_DECIMAL_CAST(i8, int8_t, i32, int32_t)
GENERATE_DECIMAL_CAST(i8, int8_t, i64, int64_t)
GENERATE_DECIMAL_CAST(i8, int8_t, i128, int128_t)
GENERATE_DECIMAL_CAST(i8, int8_t, i256, int256_t)
GENERATE_DECIMAL_CAST(i16, int16_t, i32, int32_t)
GENERATE_DECIMAL_CAST(i16, int16_t, i64, int64_t)
GENERATE_DECIMAL_CAST(i16, int16_t, i128, int128_t)
GENERATE_DECIMAL_CAST(i16, int16_t, i256, int256_t)
GENERATE_DECIMAL_CAST(i32, int32_t, i64, int64_t)
GENERATE_DECIMAL_CAST(i32, int32_t, i128, int128_t)
GENERATE_DECIMAL_CAST(i32, int32_t, i256, int256_t)
GENERATE_DECIMAL_CAST(i64, int64_t, i128, int128_t)
GENERATE_DECIMAL_CAST(i64, int64_t, i256, int256_t)
GENERATE_DECIMAL_CAST(i128, int128_t, i256, int256_t)
