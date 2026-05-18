// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompress — async dict-prefetch variant. **DOCUMENTED
// NEGATIVE RESULT** on GH200 / Hopper (sm_90), measured May 2026.
//
// Hypothesis was: stage the dict into shared via async copies
// (`cp.async.cg.shared.global`, Ampere+) so the load overlaps with
// other compute. One-time `cp.async.wait_all` + `__syncthreads` at
// kernel start; no per-chunk sync. All warps then read dict from
// shared at L1-class latency without contending for L1TEX scoreboards.
//
// Measured outcome on GH200 (`onpair_real_data` TPC-H SF=10):
//   l_returnflag   s16=133  tma16=89   (-33%)
//   l_linestatus   s16=133  tma16=89   (-33%)
//   l_shipinstruct s16=711  tma16=562  (-21%)
//   l_shipmode     s16=454  tma16=319  (-30%)
//
// Same regression magnitude (-22 to -33%) as the synchronous
// `onpair_shmem_ds` variant that was removed. The async overlap did
// not help: the L1-scoreboard stall on the global dict reads is not
// the dominant bottleneck on GH200 — the byte-pack/scan path is.
// The runbook's projected +20-30% from TMA dict prefetch was based on
// A100 ncu observations that don't transfer.
//
// Full TMA via `cp.async.bulk` + mbarrier was attempted first; the
// mbarrier state machine had a bug (separate `expect_tx` instead of
// combined `arrive.expect_tx`) that initially hung, then crashed.
// Even after the fix, the projected outcome would mirror cp.async.cg
// since both stage the same dict bytes into the same shared layout.
//
// Kept in-tree as a documented dead-end so future re-implementations
// of this idea know to skip it.
//
// We chose `cp.async.cg` (per-thread, 16-B units) over TMA
// (`cp.async.bulk.shared::cta.global` + mbarrier) because:
//   - It's strictly less error-prone (no mbarrier state machine)
//   - The dict here is small (≤ 32 KB after the host gate); the
//     per-thread granularity is fine — 32 KB / (256 threads × 16 B) =
//     8 issues per thread max
//   - It's portable to sm_80 (A100) if anyone reruns this on Ampere
//
// ABI: same shape as `onpair_shmem` with a trailing `uint32_t
// dict_entries`. Dynamic shared mem layout:
//   [s_dict (dict_entries*16, 16-B aligned)]
//   [s_lens (dict_entries)]
//   [pad to 16]
//   [s_buf_all (WPB * WARP_BUF_BYTES)]

#define WARP_BUF_BYTES 544u

__device__ inline uint32_t warp_inclusive_scan_u32_tma(uint32_t x, int lane) {
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

#if __CUDA_ARCH__ >= 800
extern "C" __global__ __launch_bounds__(512, 2) void onpair_shmem_tma(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_padded, const uint8_t *__restrict lens,
    uint8_t *__restrict output_bytes, uint64_t total_tokens,
    uint32_t dict_entries) {
    constexpr unsigned mask = 0xffffffffu;
    extern __shared__ __align__(16) uint8_t s_dyn[];

    const uint32_t dict_bytes = dict_entries * 16u;
    uint8_t *s_dict = s_dyn;
    uint8_t *s_lens = s_dyn + dict_bytes;
    uint32_t scratch_offset = (dict_bytes + dict_entries + 15u) & ~15u;
    uint8_t *s_buf_all = s_dyn + scratch_offset;

    const int tid = (int)threadIdx.x;
    const int block_threads = (int)blockDim.x;

    // Async dict prefetch: each thread issues N×16-B `cp.async.cg` ops.
    // Hopper supports many in-flight; on this small dict (≤ 32 KB) the
    // pipeline depth is limited by the dict_u4_count / block_threads ratio.
    const uint32_t dict_u4_count = dict_bytes >> 4;
    uint32_t s_dict_smem = (uint32_t)__cvta_generic_to_shared(s_dict);
    for (uint32_t k = (uint32_t)tid; k < dict_u4_count; k += (uint32_t)block_threads) {
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :: "r"(s_dict_smem + k * 16u),
               "l"(dict_padded + (size_t)k * 16u)
            : "memory");
    }
    asm volatile("cp.async.commit_group;\n");

    // Overlap: while the async dict load progresses, load lens (small,
    // already L1-hot) synchronously.
    for (uint32_t k = (uint32_t)tid; k < dict_entries; k += (uint32_t)block_threads) {
        s_lens[k] = lens[k];
    }

    // Wait for the dict prefetch to complete + block-wide visibility.
    asm volatile("cp.async.wait_all;\n");
    __syncthreads();

    const int lane = tid & 31;
    const uint32_t warp_id = (uint32_t)tid >> 5;
    const uint64_t chunk =
        (uint64_t)blockIdx.x * (uint64_t)(blockDim.x >> 5) + (uint64_t)warp_id;
    if (chunk * 32u >= total_tokens) {
        return;
    }

    uint8_t *s_buf_base = &s_buf_all[warp_id * WARP_BUF_BYTES];

    const uint64_t i = chunk * 32u + (uint64_t)lane;
    const bool active = (i < total_tokens);
    uint4 token = make_uint4(0u, 0u, 0u, 0u);
    uint32_t len = 0u;
    if (active) {
        const uint32_t code = (uint32_t)codes[i];
        token = *reinterpret_cast<const uint4 *>(s_dict + (size_t)code * 16u);
        len = (uint32_t)s_lens[code];
    }

    const uint32_t incl = warp_inclusive_scan_u32_tma(len, lane);
    const uint32_t excl = incl - len;
    const uint32_t warp_total = __shfl_sync(mask, incl, 31);

    const uint64_t out_start = chunk_offsets[chunk];
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);
    if (active) {
        memcpy(s_buf + excl, &token, (size_t)len);
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
#else
extern "C" __global__ void onpair_shmem_tma(
    const uint16_t *, const uint64_t *, const uint8_t *, const uint8_t *,
    uint8_t *, uint64_t, uint32_t) {
    // No-op stub on pre-Ampere; the bench gates on availability.
}
#endif
