// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "scalar_kernel.cuh"
#include "types.cuh"
#include <type_traits>

template <typename InputT, typename OutputT>
struct WidenOp {
    static_assert(sizeof(InputT) < sizeof(OutputT));
    static_assert(std::is_signed_v<InputT> == std::is_signed_v<OutputT>);

    __device__ OutputT operator()(InputT value) const {
        return static_cast<OutputT>(value);
    }
};

#define GENERATE_WIDEN(input_suffix, InputT, output_suffix, OutputT)                                         \
    extern "C" __global__ void widen_##input_suffix##_##output_suffix(const InputT *__restrict input,        \
                                                                      OutputT *__restrict output,            \
                                                                      uint64_t array_len) {                  \
        scalar_kernel(input, output, array_len, WidenOp<InputT, OutputT> {});                                \
    }

GENERATE_WIDEN(i8, int8_t, i16, int16_t)
GENERATE_WIDEN(i8, int8_t, i32, int32_t)
GENERATE_WIDEN(i8, int8_t, i64, int64_t)
GENERATE_WIDEN(i16, int16_t, i32, int32_t)
GENERATE_WIDEN(i16, int16_t, i64, int64_t)
GENERATE_WIDEN(i32, int32_t, i64, int64_t)
GENERATE_WIDEN(u8, uint8_t, u16, uint16_t)
GENERATE_WIDEN(u8, uint8_t, u32, uint32_t)
GENERATE_WIDEN(u8, uint8_t, u64, uint64_t)
GENERATE_WIDEN(u16, uint16_t, u32, uint32_t)
GENERATE_WIDEN(u16, uint16_t, u64, uint64_t)
GENERATE_WIDEN(u32, uint32_t, u64, uint64_t)
