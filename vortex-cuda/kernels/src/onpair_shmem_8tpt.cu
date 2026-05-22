// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompress — 8 tokens per thread, 256 tokens per warp-chunk.
//
// Extends the amortisation sweep beyond `onpair_shmem_4tpt`: each warp now
// covers 256 tokens, so the per-warp head/tail drain epilogue is amortised over
// ~twice the output, and 8 dict loads are issued per thread before the scan
// (more memory-level parallelism per thread to tolerate latency under low
// occupancy). Cost: 8× warp scans and 8 `uint4`/`uint32` live values double the
// register pressure vs 4tpt, which can force spills or cut occupancy — the open
// question this kernel answers on B200.
//
// chunk_offsets must be the 256-token cumulative layout (registry chunk_size=256).

// Each warp's 4112 B staging buffer caps the static-shared block size: 8 warps
// (256 threads) use 32 KB, under the 48 KB static cap; 16 warps would overflow.
#ifndef WARPS_PER_BLOCK_MAX
#define WARPS_PER_BLOCK_MAX 8u
#endif
#ifndef ONPAIR_LAUNCH_BOUNDS
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(256, 4)
#endif
// Each warp holds up to 256 × 16 = 4096 B token bytes plus head-shift slack.
#define WARP_BUF_BYTES 4112u

__device__ inline uint32_t warp_inclusive_scan_u32_8tpt(uint32_t x, int lane) {
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

__device__ inline void emit_token_8tpt(uint8_t *s_buf, uint32_t base,
                                       const uint4 &tok, uint32_t len) {
    const uint8_t *tb = reinterpret_cast<const uint8_t *>(&tok);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        if (j < (int)len) {
            s_buf[base + j] = tb[j];
        }
    }
}

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void onpair_shmem_8tpt(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_padded, const uint8_t *__restrict lens,
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
    uint4 t[8];
    uint32_t l[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint64_t i = base_i + (uint64_t)(k * 32);
        if (i < total_tokens) {
            const uint32_t c = (uint32_t)codes[i];
            t[k] = *reinterpret_cast<const uint4 *>(dict_padded + (size_t)c * 16u);
            l[k] = (uint32_t)lens[c];
        } else {
            t[k] = make_uint4(0u, 0u, 0u, 0u);
            l[k] = 0u;
        }
    }

    uint32_t excl[8];
    uint32_t acc_base = 0u;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint32_t incl = warp_inclusive_scan_u32_8tpt(l[k], lane);
        excl[k] = acc_base + (incl - l[k]);
        acc_base += __shfl_sync(mask, incl, 31);
    }
    const uint32_t warp_total = acc_base;

    const uint64_t out_start = chunk_offsets[chunk];
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

#pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (l[k] > 0u) {
            emit_token_8tpt(s_buf, excl[k], t[k], l[k]);
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
