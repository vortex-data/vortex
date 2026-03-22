// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Test kernel to verify Rust launch config matches CUDA kernel config.
// This kernel writes the kernel-side constants to an output buffer so
// the Rust test can verify they match.

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "config.cuh"

// Kernel that outputs the config values for verification.
// Output buffer layout: [elements_per_thread, block_dim_x, elements_per_block]
extern "C" __global__ void config_check(uint32_t *output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        output[0] = ELEMENTS_PER_THREAD;
        output[1] = blockDim.x;
        output[2] = blockDim.x * ELEMENTS_PER_THREAD;
    }
}
