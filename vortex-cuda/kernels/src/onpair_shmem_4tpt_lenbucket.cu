// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompress — 4 tokens/thread, variable-stride "length-bucket" dict.
//
// The flat dict pads every entry to 16 B (dict_padded), so a 4096-entry bits12
// dict is 64 KB and a 65536-entry bits16 dict is 1 MB. Most tokens are short
// (mean ~6 B), so most of that is wasted padding that bloats the L1 working set
// — the cause of the L1/TEX-request bottleneck (and bits16's 35% L1 hit).
//
// This layout (built decode-side from the on-disk bytes — NO compressor/on-disk
// change) sorts dict entries into 4 width buckets and packs each at its bucket
// stride: len 1-4 → stride 4, 5-8 → stride 8, 9-12 → stride 12, 13-16 → stride
// 16. Codes are relabeled so the code value alone selects the bucket via three
// thresholds (t1,t2,t3). Total dict shrinks to n0·4+n1·8+n2·12+n3·16 (often
// ~2-3x smaller) → better L1 residency, and each read is exactly the bucket
// stride (aligned), so no over-read and no split8read-style double read.
//
// Cost: a per-token bucket branch (some warp divergence). Whether the smaller
// working set beats the divergence is an NCU question.

#ifndef WARPS_PER_BLOCK_MAX
#define WARPS_PER_BLOCK_MAX 16u
#endif
#ifndef ONPAIR_LAUNCH_BOUNDS
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(512, 2)
#endif
#define WARP_BUF_BYTES 2080u

__device__ inline uint32_t warp_inclusive_scan_u32_lb(uint32_t x, int lane) {
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

// Load `stride` bytes (4/8/12/16) at a bucket-aligned address into a uint4.
__device__ inline uint4 lb_load(const uint8_t *p, uint32_t stride) {
    uint4 t = make_uint4(0u, 0u, 0u, 0u);
    if (stride == 4u) {
        t.x = *reinterpret_cast<const uint32_t *>(p);
    } else if (stride == 8u) {
        const uint2 v = *reinterpret_cast<const uint2 *>(p);
        t.x = v.x;
        t.y = v.y;
    } else if (stride == 12u) {
        // stride-12 regions are only 4-byte aligned, so read three 4-byte words.
        const uint32_t *w = reinterpret_cast<const uint32_t *>(p);
        t.x = w[0];
        t.y = w[1];
        t.z = w[2];
    } else {
        t = *reinterpret_cast<const uint4 *>(p);
    }
    return t;
}

__device__ inline void emit_token_lb(uint8_t *s_buf, uint32_t base,
                                     const uint4 &tok, uint32_t len) {
    const uint8_t *tb = reinterpret_cast<const uint8_t *>(&tok);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        if (j < (int)len) {
            s_buf[base + j] = tb[j];
        }
    }
}

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void onpair_shmem_4tpt_lenbucket(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_lb, const uint8_t *__restrict lens,
    uint8_t *__restrict output_bytes, uint64_t total_tokens, uint32_t t1,
    uint32_t t2, uint32_t t3, uint32_t base1, uint32_t base2, uint32_t base3) {
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
            l[k] = (uint32_t)lens[c];
            uint32_t addr, stride;
            if (c < t1) {
                stride = 4u;
                addr = (c) * 4u;
            } else if (c < t2) {
                stride = 8u;
                addr = base1 + (c - t1) * 8u;
            } else if (c < t3) {
                stride = 12u;
                addr = base2 + (c - t2) * 12u;
            } else {
                stride = 16u;
                addr = base3 + (c - t3) * 16u;
            }
            t[k] = lb_load(dict_lb + addr, stride);
        } else {
            t[k] = make_uint4(0u, 0u, 0u, 0u);
            l[k] = 0u;
        }
    }

    uint32_t excl[4];
    uint32_t acc_base = 0u;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t incl = warp_inclusive_scan_u32_lb(l[k], lane);
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
            emit_token_lb(s_buf, excl[k], t[k], l[k]);
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
