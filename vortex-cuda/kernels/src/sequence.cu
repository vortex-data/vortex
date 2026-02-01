// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include <stdint.h>

#define MIN(a, b) (((a) < (b)) : (a) : (b))

template<typename ValueT>
__device__ void sequence(
    ValueT *const output,
    ValueT base,
    ValueT multiplier,
    uint64_t len
) {
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;

    const uint64_t elemStart = MIN(worker * ELEMENTS_PER_THREAD, len);
    const uint64_t elemEnd = MIN(elemStart + ELEMENTS_PER_THREAD, len);

    for (uint64_t idx = elemStart; idx < elemEnd; idx++) {
        output[idx] = static_cast<ValueT>(idx) * multiplier + base;
    }
}

#define GENERATE_KERNEL(ValueT, suffix) \
extern "C" __global__ void sequence_##suffix( \
    ValueT *const output, \
    ValueT base, \
    ValueT multiplier, \
    uint64_t len \
) { \
    sequence(output, base, multiplier, len); \
}

GENERATE_KERNEL(uint8_t, u8);
GENERATE_KERNEL(uint16_t, u16);
GENERATE_KERNEL(uint32_t, u32);
GENERATE_KERNEL(uint64_t, u64);
GENERATE_KERNEL(int8_t, i8);
GENERATE_KERNEL(int16_t, i16);
GENERATE_KERNEL(int32_t, i32);
GENERATE_KERNEL(int64_t, i64);
GENERATE_KERNEL(float, f32);
GENERATE_KERNEL(double, f64);
