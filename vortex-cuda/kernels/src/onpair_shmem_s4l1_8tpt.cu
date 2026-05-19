// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompress — stride-4 × 8 tokens per thread, 256 tokens per warp.
//
// For the worst short-mean columns (e.g. TPC-H l_returnflag at mean = 1.00),
// `onpair_shmem_s4l1_4tpt` still produces only ~128 bytes of body drain
// per warp — at 8 uint4 stores there's room to push further. 8 tokens
// per thread brings 256 B body output for mean-1 workloads (16 uint4
// stores, all 32 lanes spread across two rounds). Stride-4 keeps the
// per-token register cost to one u32, so 8 slots = 8 regs.

#define WARPS_PER_BLOCK_MAX 16u
// 256 tokens × 4 ladder + slack.
#define WARP_BUF_BYTES 1056u
#define MAX_LEN_PAD 4u
#define TOKENS_PER_THREAD 8u

__device__ inline uint32_t warp_inclusive_scan_u32_s4l1_8tpt(uint32_t x, int lane) {
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

__device__ inline void emit_tok_s4_8tpt(uint8_t *s_buf, uint32_t base,
                                        uint32_t tok, uint32_t len) {
    const uint8_t *tb = reinterpret_cast<const uint8_t *>(&tok);
#pragma unroll
    for (int j = 0; j < (int)MAX_LEN_PAD; ++j) {
        if (j < (int)len) {
            s_buf[base + j] = tb[j];
        }
    }
}

extern "C" __global__ __launch_bounds__(384, 4) void onpair_shmem_s4l1_8tpt(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_padded_s4, const uint8_t *__restrict lens,
    uint8_t *__restrict output_bytes, uint64_t total_tokens) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint64_t chunk =
        (uint64_t)blockIdx.x * (uint64_t)(blockDim.x >> 5) + (uint64_t)warp_id;
    if (chunk * 256u >= total_tokens) {
        return;
    }

    __shared__ __align__(16) uint8_t s_buf_all[WARPS_PER_BLOCK_MAX * WARP_BUF_BYTES];
    uint8_t *s_buf_base = &s_buf_all[warp_id * WARP_BUF_BYTES];

    const uint64_t base_i = chunk * 256u + (uint64_t)lane;
    uint32_t t[TOKENS_PER_THREAD];
    uint32_t l[TOKENS_PER_THREAD];
#pragma unroll
    for (int k = 0; k < (int)TOKENS_PER_THREAD; ++k) {
        const uint64_t i = base_i + (uint64_t)(k * 32);
        if (i < total_tokens) {
            const uint32_t c = (uint32_t)codes[i];
            t[k] = *reinterpret_cast<const uint32_t *>(
                dict_padded_s4 + (size_t)c * MAX_LEN_PAD);
            l[k] = (uint32_t)lens[c];
        } else {
            t[k] = 0u;
            l[k] = 0u;
        }
    }

    uint32_t excl[TOKENS_PER_THREAD];
    uint32_t acc = 0u;
#pragma unroll
    for (int k = 0; k < (int)TOKENS_PER_THREAD; ++k) {
        const uint32_t incl = warp_inclusive_scan_u32_s4l1_8tpt(l[k], lane);
        excl[k] = acc + (incl - l[k]);
        acc += __shfl_sync(mask, incl, 31);
    }
    const uint32_t warp_total = acc;

    const uint64_t out_start = chunk_offsets[chunk];
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

#pragma unroll
    for (int k = 0; k < (int)TOKENS_PER_THREAD; ++k) {
        if (l[k] > 0u) {
            emit_tok_s4_8tpt(s_buf, excl[k], t[k], l[k]);
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
