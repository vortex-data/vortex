// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "scalar_kernel.cuh"

// ZigZag decode operation.
//
// Converts unsigned integers back to signed using the ZigZag encoding scheme.
// Formula: decoded = (encoded >> 1) ^ -(encoded & 1)
// This interleaves positive and negative numbers: 0, -1, 1, -2, 2, -3, ...
template <typename UnsignedT>
struct ZigZagOp {
    __device__ inline UnsignedT operator()(UnsignedT value) const {
        return (value >> 1) ^ (UnsignedT(0) - (value & 1));
    }
};

// Macro to generate the in-place ZigZag kernel for each type.
#define GENERATE_ZIGZAG_KERNEL(suffix, UnsignedType)                                                         \
    extern "C" __global__ void zigzag_##suffix(UnsignedType *__restrict values, uint64_t array_len) {        \
        scalar_kernel_inplace(values, array_len, ZigZagOp<UnsignedType> {});                                 \
    }

GENERATE_ZIGZAG_KERNEL(u8, uint8_t)
GENERATE_ZIGZAG_KERNEL(u16, uint16_t)
GENERATE_ZIGZAG_KERNEL(u32, uint32_t)
GENERATE_ZIGZAG_KERNEL(u64, uint64_t)
