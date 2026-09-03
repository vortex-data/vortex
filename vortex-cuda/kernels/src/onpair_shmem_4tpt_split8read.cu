// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompress — 4 tokens/thread, split-read dictionary.
//
// One warp decodes one 128-token batch (4 tokens per lane) in four phases:
//
// 1. Load   — each lane fetches its 4 codes, their first-8 dictionary bytes
//             (`dict_s8`), and their lengths.
// 2. Scan   — a warp prefix-scan of the lengths positions every token within
//             the batch and yields the batch's total decoded size.
// 3. Stage  — token bytes are gathered into a per-warp shared staging buffer;
//             only the rare `len > 8` tokens touch the full 16-byte-row
//             `dict_padded`.
// 4. Drain  — the staged bytes stream to global output as an aligned `uint4`
//             body between a byte head and tail.
//
// Baseline `onpair_shmem_4tpt` is L1/TEX-cache-request bound on the per-token
// 16-byte `uint4` gather into the 64 KB padded dict, where the dict L1 hit rate
// is only ~31% (the 64 KB dict thrashes against the streaming codes/output).
//
// Most tokens are short (mean dict len ~6). This variant reads the common case
// from the **32 KB** `dict_s8` array (first 8 bytes/entry, `uint2`) and only
// touches the 64 KB `dict_padded` for the rare `len > 8` tokens. Halving the
// hot dict working set aims to raise the dict L1 hit rate, cutting L2 sectors
// and L1/TEX-request pressure. As a bonus, holding `uint2 lo[4]` (32 B) instead
// of `uint4 t[4]` (64 B) lowers register pressure.

#ifndef WARPS_PER_BLOCK_MAX
#define WARPS_PER_BLOCK_MAX 16u
#endif
// The launch bounds are an exception, not a pattern: this kernel was tuned on
// GH200 where NCU showed it register-capped at 64 (50% occupancy); bounding to
// 2 blocks/SM was benchmarked as the best trade against spills. Re-evaluate on
// new architectures rather than copying this to other kernels.
#ifndef ONPAIR_LAUNCH_BOUNDS
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(512, 2)
#endif
#define WARP_BUF_BYTES 2080u

__device__ inline uint32_t onpair_warp_inclusive_scan_u32(uint32_t x, uint32_t lane) {
    constexpr unsigned mask = 0xffffffffu;
#pragma unroll
    for (uint8_t offset = 1; offset < 32; offset <<= 1) {
        uint32_t y = __shfl_up_sync(mask, x, offset);
        if (lane >= offset) {
            x += y;
        }
    }
    return x;
}

// One lane's slice of a 128-token batch: 4 tokens, strided a warp apart.
struct OnPairTokens {
    // First 8 dictionary bytes of each token (the common-case read).
    uint2 lo[4];
    // Dictionary codes, kept for the rare `len > 8` high-byte gather.
    uint32_t code[4];
    // Decoded token lengths.
    uint32_t len[4];
};

// Phase 1 — load this lane's 4 (code, dict_s8 bytes, length) triples. Tokens
// outside the visible window load as empty. `CodeT` is the code stream's
// element type (u16 natively, u8 when the compressor narrowed the codes).
template <typename CodeT>
__device__ inline OnPairTokens onpair_load_tokens(const CodeT *__restrict codes,
                                                  const uint8_t *__restrict dict_s8,
                                                  const uint8_t *__restrict lens,
                                                  uint64_t base_i,
                                                  uint64_t token_start,
                                                  uint64_t token_end) {
    OnPairTokens t;
#pragma unroll
    for (uint8_t k = 0; k < 4; ++k) {
        const uint64_t i = base_i + (uint64_t)(k * 32);
        if (i >= token_start && i < token_end) {
            const uint32_t code = (uint32_t)codes[i];
            t.code[k] = code;
            t.lo[k] = *reinterpret_cast<const uint2 *>(dict_s8 + (size_t)code * 8u);
            t.len[k] = (uint32_t)lens[code];
        } else {
            t.code[k] = 0u;
            t.lo[k] = make_uint2(0u, 0u);
            t.len[k] = 0u;
        }
    }
    return t;
}

// Phase 2 — position every token within the batch: `excl[k]` is the exclusive
// prefix (the token's staging offset) via 4 chained warp scans of the lengths.
// Returns the batch's total decoded byte count.
__device__ inline uint32_t onpair_scan_offsets(const uint32_t (&len)[4], uint32_t lane, uint32_t (&excl)[4]) {
    constexpr unsigned mask = 0xffffffffu;
    uint32_t acc_base = 0u;
#pragma unroll
    for (uint8_t k = 0; k < 4; ++k) {
        const uint32_t incl = onpair_warp_inclusive_scan_u32(len[k], lane);
        excl[k] = acc_base + (incl - len[k]);
        acc_base += __shfl_sync(mask, incl, 31);
    }
    return acc_base;
}

// Phase 3 — gather each token's bytes into the warp's shared staging buffer at
// its scanned offset. The common case writes only the 8 `dict_s8` bytes
// already in registers; tokens longer than 8 bytes take the rare path through
// the full padded dictionary.
__device__ inline void onpair_stage_tokens(const OnPairTokens &t,
                                           const uint32_t (&excl)[4],
                                           const uint8_t *__restrict dict_padded,
                                           uint8_t *__restrict s_buf) {
#pragma unroll
    for (uint8_t k = 0; k < 4; ++k) {
        const uint32_t len = t.len[k];
        if (len == 0u) {
            continue;
        }
        const uint32_t base = excl[k];
        const uint8_t *lob = reinterpret_cast<const uint8_t *>(&t.lo[k]);
        const uint32_t nlo = len < 8u ? len : 8u;
#pragma unroll
        for (uint8_t j = 0; j < 8; ++j) {
            if (j < nlo) {
                s_buf[base + j] = lob[j];
            }
        }
        if (len > 8u) {
            // Rare path: high bytes from the full padded dict.
            const uint2 hi = *reinterpret_cast<const uint2 *>(dict_padded + (size_t)t.code[k] * 16u + 8u);
            const uint8_t *hib = reinterpret_cast<const uint8_t *>(&hi);
#pragma unroll
            for (uint8_t j = 0; j < 8; ++j) {
                if (8u + j < len) {
                    s_buf[base + 8 + j] = hib[j];
                }
            }
        }
    }
}

// Phase 4 — copy the staged batch to global output: a byte head up to the
// first 16-aligned output address, an aligned `uint4` body with streaming
// stores, and a byte tail. The caller offset `s_buf` so shared and global
// 16-byte alignment phases match.
__device__ inline void onpair_drain(const uint8_t *__restrict s_buf,
                                    uint8_t *__restrict output_bytes,
                                    uint64_t out_start,
                                    uint32_t head_pre,
                                    uint32_t warp_total,
                                    uint32_t lane) {
    const uint32_t head = head_pre < warp_total ? head_pre : warp_total;
    if (lane < head) {
        output_bytes[out_start + (uint64_t)lane] = s_buf[lane];
    }
    if (head >= warp_total) {
        return;
    }

    const uint32_t body_chunks = (warp_total - head) >> 4;
    for (uint8_t k = (uint8_t)lane; k < body_chunks; k += 32u) {
        const uint32_t off = head + k * 16u;
        const uint4 v = *reinterpret_cast<const uint4 *>(s_buf + off);
        __stcs(reinterpret_cast<uint4 *>(output_bytes + out_start + off), v);
    }

    const uint32_t tail_start = head + (body_chunks << 4);
    if (lane < warp_total - tail_start) {
        output_bytes[out_start + (uint64_t)tail_start + (uint64_t)lane] = s_buf[tail_start + lane];
    }
}

template <typename CodeT>
__device__ inline void onpair_decode_body(const CodeT *__restrict codes,
                                          const uint64_t *__restrict chunk_offsets,
                                          const uint8_t *__restrict dict_s8,
                                          const uint8_t *__restrict dict_padded,
                                          const uint8_t *__restrict lens,
                                          uint8_t *__restrict output_bytes,
                                          uint64_t token_start,
                                          uint64_t token_end,
                                          uint64_t first_batch,
                                          uint64_t byte_start) {
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint64_t window_chunk = (uint64_t)blockIdx.x * (uint64_t)(blockDim.x >> 5) + (uint64_t)warp_id;
    const uint64_t chunk = first_batch + window_chunk;
    if (chunk * 128u >= token_end) {
        return;
    }

    __shared__ __align__(16) uint8_t s_buf_all[WARPS_PER_BLOCK_MAX * WARP_BUF_BYTES];
    uint8_t *s_buf_base = &s_buf_all[warp_id * WARP_BUF_BYTES];

    const OnPairTokens t =
        onpair_load_tokens(codes, dict_s8, lens, chunk * 128u + (uint64_t)lane, token_start, token_end);

    uint32_t excl[4];
    const uint32_t warp_total = onpair_scan_offsets(t.len, lane, excl);

    // Offset the staging buffer by (out_start % 16) so the drain's global
    // 16-byte stores land aligned when copied from 16-aligned shared reads.
    const uint64_t out_start = chunk == first_batch ? 0 : chunk_offsets[chunk] - byte_start;
    const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
    uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

    onpair_stage_tokens(t, excl, dict_padded, s_buf);
    __syncwarp();

    onpair_drain(s_buf, output_bytes, out_start, head_pre, warp_total, lane);
}

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void
onpair_shmem_4tpt_split8read_u8(const uint8_t *__restrict codes,
                                const uint64_t *__restrict chunk_offsets,
                                const uint8_t *__restrict dict_s8,
                                const uint8_t *__restrict dict_padded,
                                const uint8_t *__restrict lens,
                                uint8_t *__restrict output_bytes,
                                uint64_t token_start,
                                uint64_t token_end,
                                uint64_t first_batch,
                                uint64_t byte_start) {
    onpair_decode_body<uint8_t>(codes,
                                chunk_offsets,
                                dict_s8,
                                dict_padded,
                                lens,
                                output_bytes,
                                token_start,
                                token_end,
                                first_batch,
                                byte_start);
}

extern "C" __global__ ONPAIR_LAUNCH_BOUNDS void
onpair_shmem_4tpt_split8read_u16(const uint16_t *__restrict codes,
                                 const uint64_t *__restrict chunk_offsets,
                                 const uint8_t *__restrict dict_s8,
                                 const uint8_t *__restrict dict_padded,
                                 const uint8_t *__restrict lens,
                                 uint8_t *__restrict output_bytes,
                                 uint64_t token_start,
                                 uint64_t token_end,
                                 uint64_t first_batch,
                                 uint64_t byte_start) {
    onpair_decode_body<uint16_t>(codes,
                                 chunk_offsets,
                                 dict_s8,
                                 dict_padded,
                                 lens,
                                 output_bytes,
                                 token_start,
                                 token_end,
                                 first_batch,
                                 byte_start);
}
