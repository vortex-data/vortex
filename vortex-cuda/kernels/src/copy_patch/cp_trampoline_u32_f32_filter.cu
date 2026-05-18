// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Filter variant of the Copy-and-Patch trampoline: u32 bitpacked -> ALP f32 ->
// boolean predicate vs constant. Output is a uint8 mask (one byte per element)
// for prototype simplicity; a real implementation would emit a Vortex bitmap.

#include <stdint.h>

extern "C" __device__ void cp_unpack(const uint32_t *__restrict in,
                                     uint32_t *__restrict enc_buf);
extern "C" __device__ void cp_alp_apply(const uint32_t *__restrict enc_buf,
                                        float *__restrict dec_buf,
                                        float f,
                                        float e);
extern "C" __device__ void cp_filter_op(const float *__restrict dec_buf,
                                        uint8_t *__restrict mask,
                                        float c);

extern "C" __global__ void cp_trampoline_filter_u32_f32(const uint32_t *__restrict full_in,
                                                        uint8_t *__restrict full_mask,
                                                        uint64_t /*array_len*/,
                                                        uint32_t enc_stride_words,
                                                        float f,
                                                        float e,
                                                        float c) {
    __shared__ uint32_t enc_buf[1024];
    __shared__ float dec_buf[1024];

    const uint32_t *chunk_in = full_in + static_cast<uint64_t>(blockIdx.x) * enc_stride_words;
    uint8_t *chunk_mask = full_mask + static_cast<uint64_t>(blockIdx.x) * 1024;

    cp_unpack(chunk_in, enc_buf);
    __syncwarp();
    cp_alp_apply(enc_buf, dec_buf, f, e);
    __syncwarp();
    cp_filter_op(dec_buf, chunk_mask, c);
}
