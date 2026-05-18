// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompress — stride-8 specialization.
//
// When the OnPair dict's max entry length is ≤ 8 B (known AOT from the
// dict's `lens` table), the host packs the dict at 8 B / entry instead of
// 16, halving dict size and L1 sector traffic. This kernel:
//   * loads tokens as `uint64_t` (8 B aligned).
//   * unrolls the byte memcpy ladder to 8 iterations (NVCC emits ~8
//     conditional byte stores instead of 16), halving Phase 3 LSU work.
//
// ABI matches `onpair_shmem` except `dict_padded` is stride-8 (size =
// `dict_size * 8`) and `MAX_LEN_PAD = 8`.

#define WARPS_PER_BLOCK_MAX 16u
#define WARP_BUF_BYTES 544u
#define MAX_LEN_PAD 8u

__device__ inline uint32_t warp_inclusive_scan_u32_s8(uint32_t x, int lane) {
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

extern "C" __global__ __launch_bounds__(256, 8) void onpair_shmem_s8(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_padded_s8, const uint8_t *__restrict lens,
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

    const uint64_t i = chunk * 32u + (uint64_t)lane;
    const bool active = (i < total_tokens);
    uint64_t token = 0u;
    uint32_t len = 0u;
    if (active) {
        const uint32_t code = (uint32_t)codes[i];
        // Stride-8 aligned u64 load (vs uint4 in the stride-16 variant).
        token = *reinterpret_cast<const uint64_t *>(dict_padded_s8 + (size_t)code * MAX_LEN_PAD);
        len = (uint32_t)lens[code];
    }

    const uint32_t incl = warp_inclusive_scan_u32_s8(len, lane);
    const uint32_t excl = incl - len;
    const uint32_t warp_total = __shfl_sync(mask, incl, 31);

    const uint64_t out_start = chunk_offsets[chunk];
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

    // Phase 3: byte-write to shared, max 8 stores per lane (vs 16 in
    // the stride-16 variant). Each `if` predicates one byte store; NVCC
    // unrolls the loop fully because MAX_LEN_PAD is a compile-time const.
    if (active) {
        const uint8_t *token_bytes = reinterpret_cast<const uint8_t *>(&token);
#pragma unroll
        for (int j = 0; j < (int)MAX_LEN_PAD; ++j) {
            if (j < (int)len) {
                s_buf[excl + j] = token_bytes[j];
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
