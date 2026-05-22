// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decode ABLATION harness (NCU proxy — NOT byte-exact; timing only).
//
// NCU is blocked in-container, so we infer the limiter by removing one component
// at a time and measuring the speedup. Same `Stride16` arg signature and 4tpt /
// 128-thread launch as `b128o12`. Compile-time flags select what to ablate:
//   (none)              full decode (baseline, byte-exact)
//   ABLATE_GATHER       skip the dict gather (token bytes = cheap fn of code)
//   ABLATE_EMIT         skip writing token bytes into the shared staging buffer
//   ABLATE_DRAIN        skip the coalesced global output drain (write 1 sentinel)
//   ABLATE_SCAN         skip the 4x warp prefix-scan (use fixed offsets)
// The component whose removal speeds the kernel up most is the bottleneck.

#ifndef WARPS_PER_BLOCK_MAX
#define WARPS_PER_BLOCK_MAX 4u
#endif
#ifndef ONPAIR_LAUNCH_BOUNDS
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(128, 12)
#endif
#ifndef ONPAIR_ABLATE_NAME
#define ONPAIR_ABLATE_NAME onpair_shmem_4tpt_ablate
#endif
#define WARP_BUF_BYTES 2080u

__device__ inline uint32_t warp_inclusive_scan_u32_abl(uint32_t x, int lane) {
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

__device__ inline void emit_token_abl(uint8_t *s_buf, uint32_t base,
                                      const uint4 &tok, uint32_t len) {
    const uint8_t *tb = reinterpret_cast<const uint8_t *>(&tok);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        if (j < (int)len) {
            s_buf[base + j] = tb[j];
        }
    }
}

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void ONPAIR_ABLATE_NAME(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_padded, const uint8_t *__restrict lens,
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
#ifdef ABLATE_GATHER
            // No dict read: synthesize bytes from the code, keep lens load.
            t[k] = make_uint4(c * 2654435761u, c, c, c);
            l[k] = (uint32_t)lens[c];
#else
            t[k] = *reinterpret_cast<const uint4 *>(dict_padded + (size_t)c * 16u);
            l[k] = (uint32_t)lens[c];
#endif
        } else {
            t[k] = make_uint4(0u, 0u, 0u, 0u);
            l[k] = 0u;
        }
    }

    uint32_t excl[4];
    uint32_t warp_total;
#ifdef ABLATE_SCAN
    // Fixed offsets instead of the prefix scan: each lane's tokens packed at a
    // constant stride. Writes ~512 B/warp (comparable to short-text reality).
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        excl[k] = ((uint32_t)lane + k * 32u) * 4u;
        if (l[k] > 8u) {
            l[k] = 8u;  // keep within the fixed stride
        }
    }
    warp_total = 128u * 4u;
#else
    uint32_t acc_base = 0u;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t incl = warp_inclusive_scan_u32_abl(l[k], lane);
        excl[k] = acc_base + (incl - l[k]);
        acc_base += __shfl_sync(mask, incl, 31);
    }
    warp_total = acc_base;
#endif

    const uint64_t out_start = chunk_offsets[chunk];
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

#if defined(ABLATE_EMIT_CFREE)
    // Same store COUNT as the real emit (len bytes/token) but conflict-free
    // addressing: at each j, the 32 lanes hit 32 distinct banks (stride 128 B,
    // lane*4). Wrong output — isolates bank-conflict cost from store-count cost.
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint8_t *tb = reinterpret_cast<const uint8_t *>(&t[k]);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (j < (int)l[k]) {
                s_buf_base[(uint32_t)lane * 4u + (uint32_t)j * 128u] = tb[j];
            }
        }
    }
#elif !defined(ABLATE_EMIT)
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (l[k] > 0u) {
            emit_token_abl(s_buf, excl[k], t[k], l[k]);
        }
    }
#endif
    __syncwarp();

#ifdef ABLATE_DRAIN
    // Skip the coalesced drain; write one sentinel so emit/scan aren't elided.
    if (lane == 0) {
        output_bytes[out_start] = s_buf[0] + (uint8_t)warp_total;
    }
#else
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
#endif
}
