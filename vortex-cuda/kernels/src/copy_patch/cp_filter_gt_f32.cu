// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <stdint.h>

extern "C" __device__ void cp_filter_op(const float *__restrict dec_buf,
                                        uint8_t *__restrict mask,
                                        float c) {
    int tid = threadIdx.x;
#pragma unroll
    for (int j = 0; j < 32; j++) {
        int idx = tid * 32 + j;
        mask[idx] = (dec_buf[idx] > c) ? 1u : 0u;
    }
}
