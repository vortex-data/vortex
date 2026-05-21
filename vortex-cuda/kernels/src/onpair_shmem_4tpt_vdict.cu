// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompress — 4 tokens/thread, persistent block, *variable-length*
// (un-padded) dictionary resident in shared memory.
//
// Companion to `onpair_shmem_4tpt_pdict` (padded 16 B/entry dict in shared),
// which regressed: the 64 KB padded dict halved occupancy (1 block/SM) and
// random 16 B `uint4` shared reads produced ~82M bank conflicts.
//
// This variant stores the packed `dict_bytes` (no per-entry padding) in shared
// — ~17-20 KB for bits12 dicts vs 64 KB padded — so two blocks/SM stay
// feasible. Each token reads only its `len` bytes (mean ~6) instead of a fixed
// 16, so it moves ~2.5x less shared data. The (off,len) descriptor is read
// from the global `dict_table` (8 B, L2-resident), replacing the baseline's
// 16 B `dict_padded` gather + 1 B `lens` gather.
//
// Token bytes are copied shared->shared one byte at a time because `off` is
// byte-granular (unaligned) so an aligned `uint4` shared read is impossible.
// Whether the smaller footprint + smaller reads beat the byte-granular bank
// conflicts is an open question answered by NCU.
//
// Persistent grid-stride loop amortises the one-time cooperative dict load
// across thousands of chunks per block. Targets bits12 dicts.

#ifndef WARPS_PER_BLOCK_MAX
#define WARPS_PER_BLOCK_MAX 8u
#endif
#ifndef ONPAIR_LAUNCH_BOUNDS
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(256, 2)
#endif
#define WARP_BUF_BYTES 2080u

__device__ inline uint32_t warp_inclusive_scan_u32_vdict(uint32_t x, int lane) {
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

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void onpair_shmem_4tpt_vdict(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint64_t *__restrict dict_table, const uint8_t *__restrict dict_bytes,
    uint8_t *__restrict output_bytes, uint64_t total_tokens,
    uint32_t dict_bytes_len) {
    constexpr unsigned mask = 0xffffffffu;
    extern __shared__ __align__(16) uint8_t s_dyn[];

    // Shared layout: [packed dict bytes | per-warp staging].
    uint8_t *s_dict = s_dyn;
    const uint32_t scratch_off = (dict_bytes_len + 15u) & ~15u;
    uint8_t *s_buf_all = s_dyn + scratch_off;

    // Cooperative one-time load of the packed dict (aligned uint4; dict_bytes
    // is padded by 16 trailing bytes so the over-copy stays in-bounds).
    const uint4 *g_dict4 = reinterpret_cast<const uint4 *>(dict_bytes);
    uint4 *s_dict4 = reinterpret_cast<uint4 *>(s_dict);
    const uint32_t dict_u4 = (dict_bytes_len + 15u) >> 4;
    for (uint32_t k = threadIdx.x; k < dict_u4; k += blockDim.x) {
        s_dict4[k] = g_dict4[k];
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
        uint32_t off[4];
        uint32_t l[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t i = base_i + (uint64_t)(k * 32);
            if (i < total_tokens) {
                const uint32_t c = (uint32_t)codes[i];
                const uint64_t entry = dict_table[c];
                off[k] = (uint32_t)(entry >> 16);
                l[k] = (uint32_t)(entry & 0xffffu);
            } else {
                off[k] = 0u;
                l[k] = 0u;
            }
        }

        uint32_t excl[4];
        uint32_t acc_base = 0u;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t incl = warp_inclusive_scan_u32_vdict(l[k], lane);
            excl[k] = acc_base + (incl - l[k]);
            acc_base += __shfl_sync(mask, incl, 31);
        }
        const uint32_t warp_total = acc_base;

        const uint64_t out_start = chunk_offsets[chunk];
        const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
        uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t base = excl[k];
            const uint32_t o = off[k];
            for (uint32_t j = 0; j < l[k]; ++j) {
                s_buf[base + j] = s_dict[o + j];
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
                const uint32_t boff = head + k * 16u;
                const uint4 v = *reinterpret_cast<const uint4 *>(s_buf + boff);
                __stcs(reinterpret_cast<uint4 *>(output_bytes + out_start + boff), v);
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
