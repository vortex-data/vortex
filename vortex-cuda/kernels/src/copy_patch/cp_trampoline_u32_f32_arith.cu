// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Copy-and-Patch trampoline: u32 bitpacked -> i32 reinterpret -> ALP -> f32 -> arith op.
//
// The trampoline is the only kernel entry point; all other work happens in
// `extern "C" __device__` stencils that are linked in at runtime via
// `cuLinkCreate` / `cuLinkAddData(PTX)` / `cuLinkComplete`. The bit width,
// ALP encoding, and arithmetic op are all *structural* variants — selected
// by which stencil PTX module the executor feeds the linker. The same name
// `cp_unpack` is exported by every `cp_unpack_u32_bw<N>` module, so the
// trampoline's extern reference resolves to the one specialization that
// was linked in.
//
// Layout per block:
//   - one warp (32 threads) per 1024-element chunk
//   - `enc_buf` holds bit-unpacked u32 codes (interpreted as i32 by ALP)
//   - `dec_buf` holds ALP-decoded f32 values
//   - chunks aligned to FastLanes 1024-element granularity

#include <stdint.h>

extern "C" __device__ void cp_unpack(const uint32_t *__restrict in,
                                     uint32_t *__restrict enc_buf);
extern "C" __device__ void cp_alp_apply(const uint32_t *__restrict enc_buf,
                                        float *__restrict dec_buf,
                                        float f,
                                        float e);
extern "C" __device__ void cp_arith_op(const float *__restrict dec_buf,
                                       float *__restrict out,
                                       float c);

extern "C" __global__ void cp_trampoline_arith_u32_f32(const uint32_t *__restrict full_in,
                                                       float *__restrict full_out,
                                                       uint64_t /*array_len*/,
                                                       uint32_t enc_stride_words,
                                                       float f,
                                                       float e,
                                                       float c) {
    __shared__ uint32_t enc_buf[1024];
    __shared__ float dec_buf[1024];

    const uint32_t *chunk_in = full_in + static_cast<uint64_t>(blockIdx.x) * enc_stride_words;
    float *chunk_out = full_out + static_cast<uint64_t>(blockIdx.x) * 1024;

    cp_unpack(chunk_in, enc_buf);
    __syncwarp();
    cp_alp_apply(enc_buf, dec_buf, f, e);
    __syncwarp();
    cp_arith_op(dec_buf, chunk_out, c);
}
