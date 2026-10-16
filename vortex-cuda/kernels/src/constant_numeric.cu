// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include "types.cuh"
#include <cuda_fp16.h>

// Fill an output buffer with a constant value. Threads stride by `blockDim.x` within their block's
// contiguous run so that warp stores are coalesced.
template <typename T>
__device__ void constant_fill(T *__restrict output, T value, uint64_t array_len) {
    const uint64_t elements_per_block = static_cast<uint64_t>(blockDim.x) * ELEMENTS_PER_THREAD;
    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * elements_per_block;
    const uint64_t block_stop = ::min(block_start + elements_per_block, array_len);

    for (uint64_t idx = block_start + threadIdx.x; idx < block_stop; idx += blockDim.x) {
        output[idx] = value;
    }
}

#define GENERATE_CONSTANT_NUMERIC_KERNEL(suffix, Type)                                                       \
    extern "C" __global__ void constant_numeric_##suffix(Type *__restrict output,                            \
                                                         Type value,                                         \
                                                         uint64_t array_len) {                               \
        constant_fill(output, value, array_len);                                                             \
    }

FOR_EACH_NUMERIC(GENERATE_CONSTANT_NUMERIC_KERNEL)
