// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompress — 2 tokens per thread, 64 tokens per warp-chunk.
//
// For short-mean columns (e.g. text with mean ≈ 3-4 B/token) the
// production `onpair_shmem` kernel is bottlenecked by per-warp-iter
// fixed cost: a 32-token chunk produces only ~100-130 B of body
// output, so the aligned `uint4` body drain runs over ~6-8 lanes
// while 24+ lanes sit idle; the head/tail epilogue then dominates.
//
// This variant: each lane handles 2 consecutive tokens, so a warp
// processes 64 tokens / chunk = ~200-260 B / chunk for mean=3-4.
// That puts ~16 lanes into the body drain (vs ~7 before) and halves
// the per-byte cost of the head/tail epilogue. Warp scan stays the
// same width (32 lanes); chunk_offsets is per 64-token group.
//
// Same ABI shape as `onpair_shmem` except `chunk_offsets` is sized
// for 64-token chunks rather than 32-token chunks.

#define WARPS_PER_BLOCK_MAX 16u
// Each warp's scratch holds up to 64 tokens × 16 B = 1024 B, plus
// head-shift slack. Round up to 16-B multiple.
#define WARP_BUF_BYTES 1056u

__device__ inline uint32_t warp_inclusive_scan_u32_2tpt(uint32_t x, int lane) {
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

extern "C" __global__ __launch_bounds__(512, 4) void onpair_shmem_2tpt(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_padded, const uint8_t *__restrict lens,
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

    // Each lane covers tokens at i0 = chunk*64 + lane and i1 = chunk*64 + lane + 32.
    // The "stride-32" assignment (rather than consecutive pairs) keeps the
    // per-lane warp-scan output sub-range monotonic w.r.t. lane id, which
    // matches the linear memcpy layout below.
    const uint64_t i0 = chunk * 64u + (uint64_t)lane;
    const uint64_t i1 = i0 + 32u;
    const bool a0 = (i0 < total_tokens);
    const bool a1 = (i1 < total_tokens);

    uint4 t0 = make_uint4(0u, 0u, 0u, 0u);
    uint4 t1 = make_uint4(0u, 0u, 0u, 0u);
    uint32_t l0 = 0u, l1 = 0u;
    if (a0) {
        const uint32_t c0 = (uint32_t)codes[i0];
        t0 = *reinterpret_cast<const uint4 *>(dict_padded + (size_t)c0 * 16u);
        l0 = (uint32_t)lens[c0];
    }
    if (a1) {
        const uint32_t c1 = (uint32_t)codes[i1];
        t1 = *reinterpret_cast<const uint4 *>(dict_padded + (size_t)c1 * 16u);
        l1 = (uint32_t)lens[c1];
    }

    // Inclusive scan over l0 first, then over l1 with a base.
    const uint32_t incl0 = warp_inclusive_scan_u32_2tpt(l0, lane);
    const uint32_t excl0 = incl0 - l0;
    const uint32_t warp_total0 = __shfl_sync(mask, incl0, 31);
    const uint32_t incl1 = warp_inclusive_scan_u32_2tpt(l1, lane);
    const uint32_t excl1 = incl1 - l1;
    const uint32_t warp_total1 = __shfl_sync(mask, incl1, 31);
    const uint32_t warp_total = warp_total0 + warp_total1;

    const uint64_t out_start = chunk_offsets[chunk];
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

    // Phase 3: byte-write both tokens to shared scratch. The two halves
    // are written contiguously (token i0 in [0, warp_total0), token i1
    // in [warp_total0, warp_total)). Explicit unrolled byte ladder to
    // keep tokens in register — NVCC's `memcpy(_, _, runtime_len)`
    // lowering spills to local memory and reads back per byte.
    if (a0) {
        const uint8_t *tb0 = reinterpret_cast<const uint8_t *>(&t0);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (j < (int)l0) {
                s_buf[excl0 + j] = tb0[j];
            }
        }
    }
    if (a1) {
        const uint8_t *tb1 = reinterpret_cast<const uint8_t *>(&t1);
        const uint32_t base1 = warp_total0 + excl1;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (j < (int)l1) {
                s_buf[base1 + j] = tb1[j];
            }
        }
    }
    __syncwarp();

    // Phase 4: aligned drain (same head/body/tail pattern as `onpair_shmem`).
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
