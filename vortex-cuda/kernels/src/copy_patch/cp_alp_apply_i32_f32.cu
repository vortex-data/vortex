// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// ALP transform stencil for (i32 -> f32). Reinterprets the bit-unpacked u32
// payload as i32 (BitPacked stores ALP's i32 codes as u32), then applies
// `decoded = (float)code * f * e`. Operates on a 1024-element chunk in
// shared memory; one warp (32 threads) processes 32 elements per thread.

#include <stdint.h>

extern "C" __device__ void cp_alp_apply(const uint32_t *__restrict enc_buf,
                                        float *__restrict dec_buf,
                                        float f,
                                        float e) {
    int tid = threadIdx.x;
#pragma unroll
    for (int j = 0; j < 32; j++) {
        int idx = tid * 32 + j;
        int32_t code = static_cast<int32_t>(enc_buf[idx]);
        dec_buf[idx] = static_cast<float>(code) * f * e;
    }
}
