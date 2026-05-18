// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Arith stencil: element + constant. Defines the same symbol `cp_arith_op`
// as the other arith stencils; the executor picks which PTX to feed into
// the cuLink stage so the trampoline resolves to this one.

#include <stdint.h>

extern "C" __device__ void cp_arith_op(const float *__restrict dec_buf,
                                       float *__restrict out,
                                       float c) {
    int tid = threadIdx.x;
#pragma unroll
    for (int j = 0; j < 32; j++) {
        int idx = tid * 32 + j;
        out[idx] = dec_buf[idx] + c;
    }
}
