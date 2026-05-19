// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompress — Hopper TMA bulk dict-prefetch variant.
//
// Round 3 of the dict-into-shared experiment, attempting the path the
// runbook projects as the biggest H100/H200 win:
//   - One thread issues `cp.async.bulk.shared::cta.global` for the
//     whole dict in a single hardware-managed transaction (TMA),
//     bypassing the per-thread LSU pipeline that `cp.async.cg` uses.
//   - All threads wait via `mbarrier.try_wait.parity` (no
//     __syncthreads in the critical path; the one-time init fence is
//     per-launch).
//
// Two earlier attempts in this slot failed:
//   v1: separate `mbarrier.expect_tx` (without arrive) → spin
//       deadlocked because arrival count never reached 0.
//   v2: per-thread `cp.async.cg.shared.global` → ran but regressed
//       22-33% (same anti-pattern as `__syncthreads`-based dict
//       load), then crashed on max_len=16 columns with
//       CUDA_ERROR_ILLEGAL_ADDRESS for reasons that didn't manifest
//       on the max_len=1 columns.
//
// This v3 uses TMA bulk + combined `arrive.expect_tx` + a single
// asm block for init/arrive/bulk (compiler can't reorder), plus a
// `__syncthreads()` after init so non-zero threads see the barrier
// before they wait.

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

#if __CUDA_ARCH__ >= 900
extern "C" __global__ __launch_bounds__(512, 2) void onpair_shmem_tma(
    const uint16_t *__restrict codes, const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_padded, const uint8_t *__restrict lens,
    uint8_t *__restrict output_bytes, uint64_t total_tokens,
    uint32_t dict_entries) {
    constexpr unsigned mask = 0xffffffffu;
    extern __shared__ __align__(16) uint8_t s_dyn[];

    const uint32_t dict_bytes = dict_entries * 16u;
    uint8_t *s_dict = s_dyn;
    uint64_t *bar = reinterpret_cast<uint64_t *>(s_dyn + dict_bytes);
    uint8_t *s_lens = s_dyn + dict_bytes + 16u;
    uint32_t scratch_offset = (dict_bytes + 16u + dict_entries + 15u) & ~15u;
    uint8_t *s_buf_all = s_dyn + scratch_offset;

    const int tid = (int)threadIdx.x;

    if (tid == 0) {
        const uint32_t bar_smem = (uint32_t)__cvta_generic_to_shared(bar);
        const uint32_t dict_smem = (uint32_t)__cvta_generic_to_shared(s_dict);
        // All three ops in a single asm block so the compiler cannot
        // reorder them. `arrive.expect_tx` is the *combined* form —
        // decrements arrival count AND sets the expected tx-bytes,
        // both in one PTX instruction.
        asm volatile(
            "mbarrier.init.shared.b64 [%0], 1;\n"
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %2;\n"
            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
            " [%1], [%3], %2, [%0];\n"
            :: "r"(bar_smem),
               "r"(dict_smem),
               "r"(dict_bytes),
               "l"(dict_padded)
            : "memory");
    }
    // Init must be visible to all threads before they wait.
    __syncthreads();

    // Overlap window: load `lens` (small, L1-hot) while the TMA is in
    // flight. Re-reads `lens` from global; on the next launch the L1
    // is warm so this is cheap.
    for (uint32_t k = (uint32_t)tid; k < dict_entries; k += (uint32_t)blockDim.x) {
        s_lens[k] = lens[k];
    }

    // Per-thread spin on the mbarrier. Hopper's mbarrier wait is
    // hardware-backed; idle warps don't block other warps from making
    // progress.
    {
        const uint32_t bar_smem = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile(
            "{\n"
            " .reg .pred P1;\n"
            "WAIT_LOOP: mbarrier.try_wait.parity.shared.b64 P1, [%0], 0;\n"
            " @P1 bra WAIT_DONE;\n"
            " bra WAIT_LOOP;\n"
            "WAIT_DONE:\n"
            "}\n"
            :: "r"(bar_smem));
    }

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
}
#endif
