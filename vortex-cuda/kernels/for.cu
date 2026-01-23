// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "scalar_kernel.cuh"

// Frame-of-Reference operation: adds a reference value to each element.
template<typename T>
struct ForOp {
    T reference;

    __device__ __forceinline__ T operator()(T value) const {
        return value + reference;
    }
};

// Macro to generate in-place FoR kernel for each type.
#define GENERATE_FOR_KERNEL(suffix, Type) \
extern "C" __global__ void for_##suffix( \
    Type *__restrict values, \
    Type reference, \
    uint64_t array_len \
) { \
    scalar_kernel_inplace(values, array_len, ForOp<Type>{reference}); \
}

// Macro to generate FoR kernel with separate input/output buffers.
#define GENERATE_FOR_IN_OUT_KERNEL(suffix, Type) \
extern "C" __global__ void for_in_out_##suffix( \
    const Type *__restrict input, \
    Type *__restrict output, \
    Type reference, \
    uint64_t array_len \
) { \
    scalar_kernel(input, output, array_len, ForOp<Type>{reference}); \
}

// In-place variants (modifies input buffer)
GENERATE_FOR_KERNEL(u8, uint8_t)
GENERATE_FOR_KERNEL(i8, int8_t)
GENERATE_FOR_KERNEL(u16, uint16_t)
GENERATE_FOR_KERNEL(i16, int16_t)
GENERATE_FOR_KERNEL(u32, uint32_t)
GENERATE_FOR_KERNEL(i32, int32_t)
GENERATE_FOR_KERNEL(u64, uint64_t)
GENERATE_FOR_KERNEL(i64, int64_t)

// Separate input/output variants (preserves input buffer)
GENERATE_FOR_IN_OUT_KERNEL(u8, uint8_t)
GENERATE_FOR_IN_OUT_KERNEL(i8, int8_t)
GENERATE_FOR_IN_OUT_KERNEL(u16, uint16_t)
GENERATE_FOR_IN_OUT_KERNEL(i16, int16_t)
GENERATE_FOR_IN_OUT_KERNEL(u32, uint32_t)
GENERATE_FOR_IN_OUT_KERNEL(i32, int32_t)
GENERATE_FOR_IN_OUT_KERNEL(u64, uint64_t)
GENERATE_FOR_IN_OUT_KERNEL(i64, int64_t)
