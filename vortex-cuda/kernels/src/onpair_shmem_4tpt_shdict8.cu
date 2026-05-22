// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompress — 4 tokens/thread, dict_s8 staged in SHARED (persistent grid).
//
// The GH200 NCU found decode L1/TEX-cache-REQUEST bound (93%), with the dict only
// 31% L1-resident — every random gather burns an L1/TEX request and the tag/sector
// pipeline. This kernel moves the common-case dict bytes off that path: it
// cooperatively loads the 8-byte-per-entry `dict_s8` (32 KB at bits12, 128 KB at
// bits14) into shared once per block behind one `__syncthreads`, then reads the
// 8 B common case from shared (no cache tag/sector lookup, no L1 miss). Tokens
// longer than 8 B (rare for high-`frac_le8` text) read their high 8 B from global
// `dict_padded`. Persistent grid (~2 blocks/SM) amortises the load over many
// chunks. Decode bytes are identical to `onpair_shmem_4tpt`.

#ifndef WARPS_PER_BLOCK_MAX
#define WARPS_PER_BLOCK_MAX 8u
#endif
#ifndef ONPAIR_LAUNCH_BOUNDS
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(256, 2)
#endif
#define WARP_BUF_BYTES 2080u

__device__ inline uint32_t warp_inclusive_scan_u32_shd8(uint32_t x, int lane) {
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

__device__ inline void emit_token_shd8(uint8_t *s_buf, uint32_t base,
                                       const uint4 &tok, uint32_t len) {
    const uint8_t *tb = reinterpret_cast<const uint8_t *>(&tok);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        if (j < (int)len) {
            s_buf[base + j] = tb[j];
        }
    }
}

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void onpair_shmem_4tpt_shdict8(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_s8, const uint8_t *__restrict dict_padded,
    const uint8_t *__restrict lens, uint8_t *__restrict output_bytes,
    uint64_t total_tokens, uint32_t dict_entries) {
    constexpr unsigned mask = 0xffffffffu;
    extern __shared__ __align__(16) uint8_t s_dyn[];

    // Shared layout: [dict_s8 (8 B/entry) | lens | per-warp staging].
    uint8_t *s_d8 = s_dyn;
    uint8_t *s_lens = s_dyn + (size_t)dict_entries * 8u;
    const uint32_t scratch_off = (dict_entries * 8u + dict_entries + 15u) & ~15u;
    uint8_t *s_buf_all = s_dyn + scratch_off;

    // Cooperative one-time load: dict_s8 as aligned uint2 (8 B), then lens.
    const uint2 *g_d8 = reinterpret_cast<const uint2 *>(dict_s8);
    uint2 *sd8 = reinterpret_cast<uint2 *>(s_d8);
    for (uint32_t k = threadIdx.x; k < dict_entries; k += blockDim.x) {
        sd8[k] = g_d8[k];
    }
    for (uint32_t k = threadIdx.x; k < dict_entries; k += blockDim.x) {
        s_lens[k] = lens[k];
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint32_t warps_per_block = blockDim.x >> 5;
    const uint64_t warp_stride = (uint64_t)gridDim.x * (uint64_t)warps_per_block;
    const uint64_t first_chunk =
        (uint64_t)blockIdx.x * (uint64_t)warps_per_block + (uint64_t)warp_id;

    uint8_t *s_buf_base = &s_buf_all[warp_id * WARP_BUF_BYTES];

    for (uint64_t chunk = first_chunk; chunk * 128u < total_tokens;
         chunk += warp_stride) {
        const uint64_t base_i = chunk * 128u + (uint64_t)lane;
        uint4 t[4];
        uint32_t l[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t i = base_i + (uint64_t)(k * 32);
            if (i < total_tokens) {
                const uint32_t c = (uint32_t)codes[i];
                const uint32_t len = (uint32_t)s_lens[c];
                const uint2 lo = sd8[c];  // 8 B common case from shared
                uint4 tok;
                tok.x = lo.x;
                tok.y = lo.y;
                if (len > 8u) {  // rare high 8 B from global padded dict
                    const uint2 hi = *reinterpret_cast<const uint2 *>(
                        dict_padded + (size_t)c * 16u + 8u);
                    tok.z = hi.x;
                    tok.w = hi.y;
                } else {
                    tok.z = 0u;
                    tok.w = 0u;
                }
                t[k] = tok;
                l[k] = len;
            } else {
                t[k] = make_uint4(0u, 0u, 0u, 0u);
                l[k] = 0u;
            }
        }

        uint32_t excl[4];
        uint32_t acc_base = 0u;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t incl = warp_inclusive_scan_u32_shd8(l[k], lane);
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
                emit_token_shd8(s_buf, excl[k], t[k], l[k]);
            }
        }
        __syncwarp();

        const uint32_t head = head_pre < warp_total ? head_pre : warp_total;
        if ((uint32_t)lane < head) {
            output_bytes[out_start + (uint64_t)lane] = s_buf[lane];
        }
        if (head < warp_total) {
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
        __syncwarp();
    }
}
