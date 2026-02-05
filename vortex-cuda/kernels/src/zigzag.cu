// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "scalar_kernel.cuh"

// ZigZag decode operation.
// Converts unsigned integers back to signed using the ZigZag encoding scheme.
// Formula: decoded = (encoded >> 1) ^ -(encoded & 1)
// This interleaves positive and negative numbers: 0, -1, 1, -2, 2, -3, ...
template<typename UnsignedT, typename SignedT>
struct ZigZagOp {
    __device__ __forceinline__ SignedT operator()(UnsignedT value) const {
        // ZigZag decode: (n >> 1) ^ -(n & 1)
        // The -(n & 1) is equivalent to: if (n & 1) then -1 else 0
        return static_cast<SignedT>((value >> 1) ^ (~(value & 1) + 1));
    }
};

// Macro to generate ZigZag kernel for each type.
// In-place operation: unsigned input, signed output (same size, reinterpret).
#define GENERATE_ZIGZAG_KERNEL(suffix, UnsignedType, SignedType) \
extern "C" __global__ void zigzag_##suffix( \
    UnsignedType *__restrict values, \
    uint64_t array_len \
) { \
    scalar_kernel(values, reinterpret_cast<SignedType*>(values), array_len, \
                  ZigZagOp<UnsignedType, SignedType>{}); \
}

GENERATE_ZIGZAG_KERNEL(u8, uint8_t, int8_t)
GENERATE_ZIGZAG_KERNEL(u16, uint16_t, int16_t)
GENERATE_ZIGZAG_KERNEL(u32, uint32_t, int32_t)
GENERATE_ZIGZAG_KERNEL(u64, uint64_t, int64_t)
