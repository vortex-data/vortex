// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompress — 4 tokens per thread, 128 tokens per warp-chunk.
//
// Doubles the amortisation knob beyond `onpair_shmem_2tpt`. With each
// warp covering 128 tokens (up to ~512 B output for short-mean text)
// the body drain saturates all 32 lanes and the head/tail epilogue
// shrinks to a single occurrence per ~500 B of output.
//
// Cost: 4× warp scans (20 `__shfl_up_sync` per warp), 4× byte-write
// ladders. Register pressure is higher — `__launch_bounds__(512, 2)`
// gives each thread up to 64 registers at 32 warps/SM occupancy.

#define WARPS_PER_BLOCK_MAX 16u
// Each warp holds up to 128 × 16 = 2048 B token bytes plus head-shift
// slack. Round up to 16-B multiple.
#define WARP_BUF_BYTES 2080u

__device__ inline uint32_t warp_inclusive_scan_u32_4tpt(uint32_t x, int lane) {
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

__device__ inline void emit_token(uint8_t *s_buf, uint32_t base,
                                  const uint4 &tok, uint32_t len) {
    const uint8_t *tb = reinterpret_cast<const uint8_t *>(&tok);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        if (j < (int)len) {
            s_buf[base + j] = tb[j];
        }
    }
}

extern "C" __global__ __launch_bounds__(512, 2) void onpair_shmem_4tpt(
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

    // Lane handles tokens i_k = chunk*128 + lane + k*32 for k in 0..4.
    // Layout: all 32 lanes' i0 first (32 tokens), then all i1 (32 tokens),
    // then i2, i3 — keeps in-warp scan/excl arithmetic linear in lane id.
    uint64_t base_i = chunk * 128u + (uint64_t)lane;
    uint4 t[4];
    uint32_t l[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint64_t i = base_i + (uint64_t)(k * 32);
        const bool active = (i < total_tokens);
        if (active) {
            const uint32_t c = (uint32_t)codes[i];
            t[k] = *reinterpret_cast<const uint4 *>(dict_padded + (size_t)c * 16u);
            l[k] = (uint32_t)lens[c];
        } else {
            t[k] = make_uint4(0u, 0u, 0u, 0u);
            l[k] = 0u;
        }
    }

    uint32_t excl[4];
    uint32_t sub_total[4];
    uint32_t acc_base = 0u;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t incl = warp_inclusive_scan_u32_4tpt(l[k], lane);
        excl[k] = acc_base + (incl - l[k]);
        sub_total[k] = __shfl_sync(mask, incl, 31);
        acc_base += sub_total[k];
    }
    const uint32_t warp_total = acc_base;

    const uint64_t out_start = chunk_offsets[chunk];
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (l[k] > 0u) {
            emit_token(s_buf, excl[k], t[k], l[k]);
        }
    }
    __syncwarp();

    // Phase 4: aligned drain (head / body / tail).
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
