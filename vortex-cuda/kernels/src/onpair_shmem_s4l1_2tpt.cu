// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompress — stride-4 × 2 tokens per thread.
//
// Combines stride-4 specialization (dict `max_len ≤ 4`, 4-deep byte
// ladder, u32 token load) with the 2-tokens-per-thread amortization.
// Each lane writes at most 8 conditional bytes (2 × 4 ladder depth)
// — half the work of `onpair_shmem_s8_2tpt`. The dict at stride-4 is
// tiny (≤ 1 KB for 256-entry dicts, e.g. TPC-H l_returnflag/linestatus,
// l_shipmode), so L1 hit rate on dict reads is essentially 100 % even
// at full Hopper occupancy.

#define WARPS_PER_BLOCK_MAX 16u
#define WARP_BUF_BYTES 1056u
#define MAX_LEN_PAD 4u

__device__ inline uint32_t warp_inclusive_scan_u32_s4l1_2tpt(uint32_t x, int lane) {
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

extern "C" __global__ __launch_bounds__(512, 4) void onpair_shmem_s4l1_2tpt(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_padded_s4, const uint8_t *__restrict lens,
    uint8_t *__restrict output_bytes, uint64_t total_tokens) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint64_t chunk =
        (uint64_t)blockIdx.x * (uint64_t)(blockDim.x >> 5) + (uint64_t)warp_id;
    if (chunk * 64u >= total_tokens) {
        return;
    }

    __shared__ __align__(16) uint8_t s_buf_all[WARPS_PER_BLOCK_MAX * WARP_BUF_BYTES];
    uint8_t *s_buf_base = &s_buf_all[warp_id * WARP_BUF_BYTES];

    const uint64_t i0 = chunk * 64u + (uint64_t)lane;
    const uint64_t i1 = i0 + 32u;
    const bool a0 = (i0 < total_tokens);
    const bool a1 = (i1 < total_tokens);

    uint32_t t0 = 0u, t1 = 0u;
    uint32_t l0 = 0u, l1 = 0u;
    if (a0) {
        const uint32_t c0 = (uint32_t)codes[i0];
        t0 = *reinterpret_cast<const uint32_t *>(dict_padded_s4 + (size_t)c0 * MAX_LEN_PAD);
        l0 = (uint32_t)lens[c0];
    }
    if (a1) {
        const uint32_t c1 = (uint32_t)codes[i1];
        t1 = *reinterpret_cast<const uint32_t *>(dict_padded_s4 + (size_t)c1 * MAX_LEN_PAD);
        l1 = (uint32_t)lens[c1];
    }

    const uint32_t incl0 = warp_inclusive_scan_u32_s4l1_2tpt(l0, lane);
    const uint32_t excl0 = incl0 - l0;
    const uint32_t warp_total0 = __shfl_sync(mask, incl0, 31);
    const uint32_t incl1 = warp_inclusive_scan_u32_s4l1_2tpt(l1, lane);
    const uint32_t excl1 = incl1 - l1;
    const uint32_t warp_total1 = __shfl_sync(mask, incl1, 31);
    const uint32_t warp_total = warp_total0 + warp_total1;

    const uint64_t out_start = chunk_offsets[chunk];
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

    if (a0) {
        const uint8_t *tb0 = reinterpret_cast<const uint8_t *>(&t0);
#pragma unroll
        for (int j = 0; j < (int)MAX_LEN_PAD; ++j) {
            if (j < (int)l0) {
                s_buf[excl0 + j] = tb0[j];
            }
        }
    }
    if (a1) {
        const uint8_t *tb1 = reinterpret_cast<const uint8_t *>(&t1);
        const uint32_t base1 = warp_total0 + excl1;
#pragma unroll
        for (int j = 0; j < (int)MAX_LEN_PAD; ++j) {
            if (j < (int)l1) {
                s_buf[base1 + j] = tb1[j];
            }
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
