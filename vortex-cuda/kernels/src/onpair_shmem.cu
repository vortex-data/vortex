// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair flat-chunked decompress — GSST shared-mem staging recipe.
// See `vortex-cuda/PERF_RESEARCH.md` (Vonk 2025).
//
// One warp per 32-token chunk. Each lane:
//   1. Loads its code → uint4 token (16-B stride padded dict) + len.
//   2. Warp inclusive-scan over `len` → per-lane byte offset.
//   3. Variable-length byte-write to per-warp shared scratch.
//   4. After `__syncwarp`, the warp drains shared → global with one
//      aligned uint4 store per 16-byte body chunk plus up to 15 head
//      and 15 tail byte stores around the unaligned global cursor.
//
// Inactive lanes (last partial chunk) have `len = 0`, so their `incl`
// propagates the prior sum unchanged. That means `incl` at lane 31 is
// always the warp total — no ballot/clz branch needed.
//
// Per-warp scratch: 32 × 16 useful + 16 head shift + 16 tail pad,
// rounded to 34 × 16 = 544 B. With WARPS_PER_BLOCK_MAX = 16 that's at
// most 8.5 KB shared per block, far below A100's 192 KB unified budget.

#define WARPS_PER_BLOCK_MAX 16u
#define WARP_BUF_BYTES 544u

__device__ inline uint32_t warp_inclusive_scan_u32(uint32_t x, int lane) {
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

extern "C" __global__ __launch_bounds__(512, 4) void onpair_shmem(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_padded, const uint8_t *__restrict lens,
    uint8_t *__restrict output_bytes, uint64_t total_tokens) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint64_t chunk =
        (uint64_t)blockIdx.x * (uint64_t)(blockDim.x >> 5) + (uint64_t)warp_id;
    if (chunk * 32u >= total_tokens) {
        return;
    }

    __shared__ __align__(16) uint8_t s_buf_all[WARPS_PER_BLOCK_MAX * WARP_BUF_BYTES];
    uint8_t *s_buf_base = &s_buf_all[warp_id * WARP_BUF_BYTES];

    // Phase 1: load token + len.
    const uint64_t i = chunk * 32u + (uint64_t)lane;
    const bool active = (i < total_tokens);
    uint4 token = make_uint4(0u, 0u, 0u, 0u);
    uint32_t len = 0u;
    if (active) {
        const uint32_t code = (uint32_t)codes[i];
        token = *reinterpret_cast<const uint4 *>(dict_padded + (size_t)code * 16u);
        len = (uint32_t)lens[code];
    }

    // Phase 2: warp scan for per-lane byte offset + warp total.
    const uint32_t incl = warp_inclusive_scan_u32(len, lane);
    const uint32_t excl = incl - len;
    const uint32_t warp_total = __shfl_sync(mask, incl, 31);

    // Phase 3: byte-write to shared, shifted so `s_buf + head` is
    // 16-aligned (matching the head-aligned global cursor below).
    // The byte ladder is `#pragma unroll`'d to 16 explicit conditional
    // stores from register — otherwise NVCC lowers `memcpy(_, _, len)`
    // with runtime `len` to a runtime loop that first spills `token`
    // to local memory (HBM!) and then byte-reads it back, which costs
    // ~one HBM round trip per byte.
    const uint64_t out_start = chunk_offsets[chunk];
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);
    if (active) {
        const uint8_t *token_bytes = reinterpret_cast<const uint8_t *>(&token);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (j < (int)len) {
                s_buf[excl + j] = token_bytes[j];
            }
        }
    }
    __syncwarp();

    // Phase 4: aligned drain. `head` may shrink if a very short chunk
    // emits fewer than head_pre bytes total.
    const uint32_t head = head_pre < warp_total ? head_pre : warp_total;
    if ((uint32_t)lane < head) {
        output_bytes[out_start + (uint64_t)lane] = s_buf[lane];
    }
    if (head >= warp_total) {
        return;
    }

    // Body: aligned uint4 stores. Source and dest are both 16-aligned
    // at `+ head`, so each iter is a natural u128 transaction.
    // __stcs: streaming hint — output bytes aren't re-read here, so
    // bypassing L1 write-back keeps the cache free for the random
    // `dict_padded` reads.
    const uint32_t body_chunks = (warp_total - head) >> 4;
    for (uint32_t k = lane; k < body_chunks; k += 32u) {
        const uint32_t off = head + k * 16u;
        const uint4 v = *reinterpret_cast<const uint4 *>(s_buf + off);
        __stcs(reinterpret_cast<uint4 *>(output_bytes + out_start + off), v);
    }

    // Tail: up to 15 trailing bytes.
    const uint32_t tail_start = head + (body_chunks << 4);
    if ((uint32_t)lane < warp_total - tail_start) {
        output_bytes[out_start + (uint64_t)tail_start + (uint64_t)lane] =
            s_buf[tail_start + lane];
    }
}
