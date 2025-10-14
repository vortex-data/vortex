// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Fused kernel combining FastLanes bitpacking unpack with Frame-of-Reference addition
// This avoids an intermediate memory write/read by fusing the operations

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "fastlanes_common.cuh"


__device__ void fls_unpack_6bw_32ow_device(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    int i = thread_idx;
    uint32_t src;
    uint32_t tmp;

    src = in[i * 1 + 0];
    tmp = (src >> 0) & MASK(uint32_t, 6);
    out[INDEX(0, (i * 1 + 0))] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 6);
    out[INDEX(1, (i * 1 + 0))] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 6);
    out[INDEX(2, (i * 1 + 0))] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 6);
    out[INDEX(3, (i * 1 + 0))] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 6);
    out[INDEX(4, (i * 1 + 0))] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[i * 1 + 0 + 32 * 1];
    tmp |= (src & MASK(uint32_t, 4)) << 2;
    out[INDEX(5, (i * 1 + 0))] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 6);
    out[INDEX(6, (i * 1 + 0))] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 6);
    out[INDEX(7, (i * 1 + 0))] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 6);
    out[INDEX(8, (i * 1 + 0))] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 6);
    out[INDEX(9, (i * 1 + 0))] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[i * 1 + 0 + 32 * 2];
    tmp |= (src & MASK(uint32_t, 2)) << 4;
    out[INDEX(10, (i * 1 + 0))] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 6);
    out[INDEX(11, (i * 1 + 0))] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 6);
    out[INDEX(12, (i * 1 + 0))] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 6);
    out[INDEX(13, (i * 1 + 0))] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 6);
    out[INDEX(14, (i * 1 + 0))] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[i * 1 + 0 + 32 * 3];
    tmp |= (src & MASK(uint32_t, 0)) << 6;
    out[INDEX(15, (i * 1 + 0))] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 6);
    out[INDEX(16, (i * 1 + 0))] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 6);
    out[INDEX(17, (i * 1 + 0))] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 6);
    out[INDEX(18, (i * 1 + 0))] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 6);
    out[INDEX(19, (i * 1 + 0))] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 6);
    out[INDEX(20, (i * 1 + 0))] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[i * 1 + 0 + 32 * 4];
    tmp |= (src & MASK(uint32_t, 4)) << 2;
    out[INDEX(21, (i * 1 + 0))] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 6);
    out[INDEX(22, (i * 1 + 0))] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 6);
    out[INDEX(23, (i * 1 + 0))] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 6);
    out[INDEX(24, (i * 1 + 0))] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 6);
    out[INDEX(25, (i * 1 + 0))] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[i * 1 + 0 + 32 * 5];
    tmp |= (src & MASK(uint32_t, 2)) << 4;
    out[INDEX(26, (i * 1 + 0))] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 6);
    out[INDEX(27, (i * 1 + 0))] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 6);
    out[INDEX(28, (i * 1 + 0))] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 6);
    out[INDEX(29, (i * 1 + 0))] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 6);
    out[INDEX(30, (i * 1 + 0))] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    out[INDEX(31, (i * 1 + 0))] = tmp;
}

// Device function template (callable from device code)
template<typename ValueT>
__device__ __forceinline__ void for_device(
    ValueT *__restrict values_in_out,
    ValueT reference,
    int thread_idx
) {
    auto i = thread_idx;
    const int thread_ops = blockDim.x;

    for (auto j = 0; j < thread_ops; j++) {
        auto idx = INDEX(j, i);
        values_in_out[idx] = values_in_out[idx] + reference;
    }
}


// Fused kernel: bitpack unpack (3bw) + FoR addition in one pass
// This eliminates the intermediate write-to-memory and read-from-memory
// by keeping unpacked values in registers/L1 cache and immediately adding the reference
extern "C" __global__ void fused_bitpack6_for_u32(
    const uint32_t *__restrict packed_in,
    uint32_t *__restrict unpacked_out,
    uint32_t reference
) {
    int i = threadIdx.x;
    auto in = packed_in + (blockIdx.x * (128 * 6 / sizeof(uint32_t)));
    const uint32_t fl_lane_count = 32;
    auto blockSize = blockDim.x * fl_lane_count;
    auto out = unpacked_out + (blockIdx.x * 1024);

    __shared__ uint32_t shared_data[1024];

    fls_unpack_6bw_32ow_device(in, shared_data, i);

    for_device(shared_data, reference, i);

    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + threadIdx.x;
        out[idx] = shared_data[idx];
    }
}
