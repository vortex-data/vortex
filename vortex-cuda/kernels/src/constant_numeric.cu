// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include "types.cuh"
#include <cuda_fp16.h>

// Fill an output buffer with a constant value.
template<typename T>
__device__ void constant_fill(
    T *__restrict output,
    T value,
    uint64_t array_len
) {
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t startElem = START_ELEM(worker, array_len);
    const uint64_t stopElem = STOP_ELEM(worker, array_len);

    if (startElem >= array_len) {
        return;
    }

    for (uint64_t idx = startElem; idx < stopElem; idx++) {
        output[idx] = value;
    }
}

#define GENERATE_CONSTANT_NUMERIC_KERNEL(suffix, Type) \
extern "C" __global__ void constant_numeric_##suffix( \
    Type *__restrict output, \
    Type value, \
    uint64_t array_len \
) { \
    constant_fill(output, value, array_len); \
}

FOR_EACH_NUMERIC(GENERATE_CONSTANT_NUMERIC_KERNEL)
