// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Compare each element of a typed array against a scalar value, producing
/// a packed bitmask where bit `i` is set iff the comparison holds for
/// element `i`.
///
/// Output layout: LSB-first within each byte, little-endian byte order
/// (same as Arrow/Vortex validity bitmaps).  The output buffer must have
/// at least `ceil(array_len / 8)` bytes.
///
/// Comparison operators are encoded as:
///   0 = Eq, 1 = NotEq, 2 = Gt, 3 = Gte, 4 = Lt, 5 = Lte

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "types.cuh"

enum CompareOp : uint8_t {
    CMP_EQ = 0,
    CMP_NOT_EQ = 1,
    CMP_GT = 2,
    CMP_GTE = 3,
    CMP_LT = 4,
    CMP_LTE = 5,
};

template <typename T>
__device__ inline bool compare(T value, T scalar, CompareOp op) {
    switch (op) {
    case CMP_EQ:
        return value == scalar;
    case CMP_NOT_EQ:
        return value != scalar;
    case CMP_GT:
        return value > scalar;
    case CMP_GTE:
        return value >= scalar;
    case CMP_LT:
        return value < scalar;
    case CMP_LTE:
        return value <= scalar;
    default:
        return false;
    }
}

/// Each thread processes 32 elements and writes one uint32_t of packed bits.
/// A block of 64 threads therefore covers 2048 elements (matching the
/// ELEMENTS_PER_BLOCK convention used by other kernels).
template <typename T>
__device__ void compare_kernel(const T *__restrict input,
                               uint8_t *__restrict output,
                               uint64_t array_len,
                               T scalar,
                               CompareOp op) {
    constexpr uint32_t BITS_PER_THREAD = 32;
    const uint32_t ELEMENTS_PER_BLOCK = blockDim.x * BITS_PER_THREAD; // 64 * 32 = 2048

    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * ELEMENTS_PER_BLOCK;
    const uint64_t elem_start = block_start + static_cast<uint64_t>(threadIdx.x) * BITS_PER_THREAD;

    uint32_t bits = 0;

    // Each thread evaluates 32 elements and packs the results into a uint32_t.
    #pragma unroll
    for (uint32_t i = 0; i < BITS_PER_THREAD; ++i) {
        uint64_t idx = elem_start + i;
        if (idx < array_len && compare(input[idx], scalar, op)) {
            bits |= (1u << i);
        }
    }

    // Write the packed 32 bits as 4 bytes in little-endian order (LSB first).
    // Output byte index = elem_start / 8.  Since elem_start is aligned to 32,
    // this is always aligned to a 4-byte boundary.
    uint64_t byte_offset = elem_start / 8;
    if (byte_offset < (array_len + 7) / 8) {
        // Write as a single uint32_t for efficiency.
        reinterpret_cast<uint32_t *>(output + byte_offset)[0] = bits;
    }
}

#define GENERATE_COMPARE_KERNEL(suffix, Type)                                                                \
    extern "C" __global__ void compare_##suffix(const Type *__restrict input,                                \
                                                uint8_t *__restrict output,                                  \
                                                uint64_t array_len,                                          \
                                                Type scalar,                                                 \
                                                uint8_t op) {                                                \
        compare_kernel<Type>(input, output, array_len, scalar, static_cast<CompareOp>(op));                  \
    }

FOR_EACH_INTEGER(GENERATE_COMPARE_KERNEL)
GENERATE_COMPARE_KERNEL(f32, float)
GENERATE_COMPARE_KERNEL(f64, double)