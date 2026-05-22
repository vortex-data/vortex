// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// OnPair decompress — 4 tokens/thread, QUANTIZED variable-width dictionary.
//
// Like `vwidth`, but each dict entry is padded up to a multiple of 4 bytes
// ({4,8,12,16}) and placed at a 4-byte-aligned offset in `dict_q4`. That keeps
// the per-entry footprint near the true length (mean ~7 B => 8 B slot, ~512 KB
// total for 65536 entries vs 1 MB padded) WHILE allowing legal aligned word
// loads — the exact-packing `vwidth` must use unaligned `memcpy`. The directory
// `dict_q4_dir[c]` packs (byte offset << 5 | true_len). The kernel loads
// ceil(len/4) aligned `uint32`s (1-4 words, by size class) into the token.
// Decode bytes are identical to `onpair_shmem_4tpt`.

#ifndef WARPS_PER_BLOCK_MAX
#define WARPS_PER_BLOCK_MAX 16u
#endif
#ifndef ONPAIR_LAUNCH_BOUNDS
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(512, 2)
#endif
#define WARP_BUF_BYTES 2080u

__device__ inline uint32_t warp_inclusive_scan_u32_vw4(uint32_t x, int lane) {
    constexpr unsigned mask = 0xffffffffu;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        uint32_t y = __shfl_up_sync(mask, x, offset);
        if (lane >= offset) {
            x += y;
        }
    }
    return x;
}

__device__ inline void emit_token_vw4(uint8_t *s_buf, uint32_t base,
                                      const uint4 &tok, uint32_t len) {
    const uint8_t *tb = reinterpret_cast<const uint8_t *>(&tok);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        if (j < (int)len) {
            s_buf[base + j] = tb[j];
        }
    }
}

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void onpair_shmem_4tpt_vwidth4(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint32_t *__restrict dict_q4_dir, const uint8_t *__restrict dict_q4,
    uint8_t *__restrict output_bytes, uint64_t total_tokens) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint64_t chunk =
        (uint64_t)blockIdx.x * (uint64_t)(blockDim.x >> 5) + (uint64_t)warp_id;
    if (chunk * 128u >= total_tokens) {
        return;
    }

    __shared__ __align__(16) uint8_t s_buf_all[WARPS_PER_BLOCK_MAX * WARP_BUF_BYTES];
    uint8_t *s_buf_base = &s_buf_all[warp_id * WARP_BUF_BYTES];

    const uint64_t base_i = chunk * 128u + (uint64_t)lane;
    uint4 t[4];
    uint32_t l[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint64_t i = base_i + (uint64_t)(k * 32);
        if (i < total_tokens) {
            const uint32_t c = (uint32_t)codes[i];
            const uint32_t e = dict_q4_dir[c];
            const uint32_t off = e >> 5;        // 4-byte-aligned byte offset
            const uint32_t len = e & 0x1fu;     // true length (<=16)
            l[k] = len;
            // Aligned word loads from the 4-aligned slot; read ceil(len/4) words.
            const uint32_t *wp = reinterpret_cast<const uint32_t *>(dict_q4 + (size_t)off);
            uint4 tok;
            tok.x = wp[0];
            tok.y = len > 4u ? wp[1] : 0u;
            tok.z = len > 8u ? wp[2] : 0u;
            tok.w = len > 12u ? wp[3] : 0u;
            t[k] = tok;
        } else {
            t[k] = make_uint4(0u, 0u, 0u, 0u);
            l[k] = 0u;
        }
    }

    uint32_t excl[4];
    uint32_t acc_base = 0u;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t incl = warp_inclusive_scan_u32_vw4(l[k], lane);
        excl[k] = acc_base + (incl - l[k]);
        acc_base += __shfl_sync(mask, incl, 31);
    }
    const uint32_t warp_total = acc_base;

    const uint64_t out_start = chunk_offsets[chunk];
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (l[k] > 0u) {
            emit_token_vw4(s_buf, excl[k], t[k], l[k]);
        }
    }
    __syncwarp();

    const uint32_t head = head_pre < warp_total ? head_pre : warp_total;
    if ((uint32_t)lane < head) {
        output_bytes[out_start + (uint64_t)lane] = s_buf[lane];
    }
    if (head >= warp_total) {
        return;
    }

    const uint32_t body_chunks = (warp_total - head) >> 4;
    for (uint32_t k = lane; k < body_chunks; k += 32u) {
        const uint32_t off = head + k * 16u;
        const uint4 v = *reinterpret_cast<const uint4 *>(s_buf + off);
        __stcs(reinterpret_cast<uint4 *>(output_bytes + out_start + off), v);
    }

    const uint32_t tail_start = head + (body_chunks << 4);
    if ((uint32_t)lane < warp_total - tail_start) {
        output_bytes[out_start + (uint64_t)tail_start + (uint64_t)lane] =
            s_buf[tail_start + lane];
    }
}
