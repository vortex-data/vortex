// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompress — 4 tokens/thread, *persistent* block with the padded
// 16-byte dictionary resident in shared memory.
//
// Background. The plain `onpair_shmem_4tpt` kernel is L1/TEX-cache-request
// bound (NCU: L1/TEX throughput ~93%, DRAM ~17%): the dominant cost is the
// uncoalesced 16-byte `uint4` gather into `dict_padded` for every token. The
// dict is L2-resident (94% L2 hit) but each gather still burns L1/TEX request
// throughput.
//
// Earlier "dict in shared" attempts (see `onpair_shmem_tma.cu` history)
// regressed 22-33% because they kept the 1-block-per-1024-tokens launch
// shape: a block loads the whole 64 KB padded dict but only produces ~5 KB of
// output before exiting, so the load is never amortised and the
// `__syncthreads` recurs per block.
//
// This kernel fixes the amortisation: it launches a *fixed, persistent* grid
// (~2 blocks/SM) and each warp walks the chunk space in a grid-stride loop.
// The dict is cooperatively loaded into shared once per block, behind a
// single `__syncthreads`, then reused across thousands of chunks. Token
// expansion reads the dict via aligned `uint4` loads from shared, bypassing
// the L1/TEX tag/sector pipeline entirely.
//
// Targets `bits12` dictionaries (<= 4096 entries -> 64 KB padded) which fit
// in Hopper shared memory with the opt-in larger carveout.

#ifndef WARPS_PER_BLOCK_MAX
#define WARPS_PER_BLOCK_MAX 8u
#endif
#ifndef ONPAIR_LAUNCH_BOUNDS
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(256, 2)
#endif
// Each warp holds up to 128 x 16 = 2048 B token bytes plus head-shift slack.
#define WARP_BUF_BYTES 2080u

__device__ inline uint32_t warp_inclusive_scan_u32_pdict(uint32_t x, int lane) {
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

__device__ inline void emit_token_pdict(uint8_t *s_buf, uint32_t base,
                                        const uint4 &tok, uint32_t len) {
    const uint8_t *tb = reinterpret_cast<const uint8_t *>(&tok);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        if (j < (int)len) {
            s_buf[base + j] = tb[j];
        }
    }
}

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void onpair_shmem_4tpt_pdict(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_padded, const uint8_t *__restrict lens,
    uint8_t *__restrict output_bytes, uint64_t total_tokens,
    uint32_t dict_entries) {
    constexpr unsigned mask = 0xffffffffu;
    extern __shared__ __align__(16) uint8_t s_dyn[];

    // Shared layout: [padded dict | lens | per-warp staging].
    uint8_t *s_dict = s_dyn;
    uint8_t *s_lens = s_dyn + (size_t)dict_entries * 16u;
    const uint32_t scratch_off =
        (dict_entries * 16u + dict_entries + 15u) & ~15u;
    uint8_t *s_buf_all = s_dyn + scratch_off;

    // Cooperative one-time load of the dict (aligned uint4) and lens.
    const uint4 *g_dict4 = reinterpret_cast<const uint4 *>(dict_padded);
    uint4 *s_dict4 = reinterpret_cast<uint4 *>(s_dict);
    for (uint32_t k = threadIdx.x; k < dict_entries; k += blockDim.x) {
        s_dict4[k] = g_dict4[k];
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
                t[k] = *reinterpret_cast<const uint4 *>(s_dict + (size_t)c * 16u);
                l[k] = (uint32_t)s_lens[c];
            } else {
                t[k] = make_uint4(0u, 0u, 0u, 0u);
                l[k] = 0u;
            }
        }

        uint32_t excl[4];
        uint32_t acc_base = 0u;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t incl = warp_inclusive_scan_u32_pdict(l[k], lane);
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
                emit_token_pdict(s_buf, excl[k], t[k], l[k]);
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
        // Ensure all lanes finished reading s_buf before the next chunk
        // overwrites it.
        __syncwarp();
    }
}
