// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompress — stride-4 × 4 tokens per thread, 128 tokens per warp.
//
// For columns with dict `max_len ≤ 4` and very short mean (e.g. TPC-H
// l_returnflag/l_linestatus at mean = 1.00), even `onpair_shmem_s4l1_2tpt`
// produces only ~64 bytes of body drain per warp — most lanes idle.
// 4 tokens per thread doubles output per warp again to ~128 bytes for
// these workloads while keeping each token's byte ladder at just 4
// predicate evaluations. Register pressure is moderate: a u32 token
// per slot, four slots per thread = 4 registers for tokens.

#define WARPS_PER_BLOCK_MAX 16u
// 128 tokens × 4 B ladder + head/tail slack, rounded to 16.
#define WARP_BUF_BYTES 544u
#define MAX_LEN_PAD 4u

__device__ inline uint32_t warp_inclusive_scan_u32_s4l1_4tpt(uint32_t x, int lane) {
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

__device__ inline void emit_tok_s4(uint8_t *s_buf, uint32_t base,
                                   uint32_t tok, uint32_t len) {
    const uint8_t *tb = reinterpret_cast<const uint8_t *>(&tok);
#pragma unroll
    for (int j = 0; j < (int)MAX_LEN_PAD; ++j) {
        if (j < (int)len) {
            s_buf[base + j] = tb[j];
        }
    }
}

extern "C" __global__ __launch_bounds__(512, 4) void onpair_shmem_s4l1_4tpt(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_padded_s4, const uint8_t *__restrict lens,
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
    uint32_t t[4];
    uint32_t l[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
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

    uint32_t excl[4];
    uint32_t acc = 0u;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t incl = warp_inclusive_scan_u32_s4l1_4tpt(l[k], lane);
        excl[k] = acc + (incl - l[k]);
        acc += __shfl_sync(mask, incl, 31);
    }
    const uint32_t warp_total = acc;

    const uint64_t out_start = chunk_offsets[chunk];
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (l[k] > 0u) {
            emit_tok_s4(s_buf, excl[k], t[k], l[k]);
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
