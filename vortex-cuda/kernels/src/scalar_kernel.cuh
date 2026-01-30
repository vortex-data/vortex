// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Generic scalar kernel that applies an operation element-wise.
//
// Op should be a functor with: OutputT operator()(InputT value) const
//
// Launch config: grid_dim = (array_len / 2048, 1, 1), block_dim = (64, 1, 1)
// Each block handles 2048 elements, 64 threads per block.
// Vectorized to process 16 bytes per iteration for better memory throughput.
template<typename InputT, typename OutputT, typename Op>
__device__ void scalar_kernel(
    const InputT *__restrict in,
    OutputT *__restrict out,
    uint64_t array_len,
    Op op
) {
    const uint32_t elements_per_block = 2048;
    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * elements_per_block;
    const uint64_t block_end = (block_start + elements_per_block < array_len)
        ? (block_start + elements_per_block)
        : array_len;

    // Vectorized loop - process 16 bytes per iteration for better memory throughput.
    constexpr auto VALUES_PER_LOOP = 16 / sizeof(InputT);
    const auto block_start_vec = block_start / VALUES_PER_LOOP;
    const auto block_end_vec = block_end / VALUES_PER_LOOP;

    for (uint64_t idx = block_start_vec + threadIdx.x; idx < block_end_vec; idx += blockDim.x) {
        uint64_t base_idx = idx * VALUES_PER_LOOP;

        // The loop can be unrolled, as `VALUES_PER_LOOP` is `constexpr`.
        #pragma unroll
        for (uint64_t i = 0; i < VALUES_PER_LOOP; ++i) {
            out[base_idx + i] = op(in[base_idx + i]);
        }
    }

    // Remainder loop for elements that don't fit in the vectorized loop.
    uint64_t remaining_start = block_end_vec * VALUES_PER_LOOP;
    for (uint64_t idx = remaining_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
        out[idx] = op(in[idx]);
    }
}

// In-place variant (same input/output buffer, same type).
template<typename T, typename Op>
__device__ void scalar_kernel_inplace(
    T *__restrict values,
    uint64_t array_len,
    Op op
) {
    scalar_kernel<T, T>(values, values, array_len, op);
}
