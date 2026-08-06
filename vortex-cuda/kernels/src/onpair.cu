// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"

#include <stdint.h>

// Support kernels for OnPair GPU decompression. The per-batch output-offsets
// regeneration lives in the CUB shim (`cub/kernels/onpair.cu`), where the
// reduction and exclusive scan run as one fused sweep; this module holds the
// window-bounds and view-construction kernels launched around the decode.

// Tokens per decode batch. Must match `ONPAIR_TOKENS_PER_BATCH` in
// `cub/kernels/onpair.cu` and `TOKENS_PER_BATCH` in the Rust launch code.
constexpr uint32_t ONPAIR_TOKENS_PER_BATCH = 128;

// Token and byte positions of the visible code window's bounds, resolved
// entirely on device. A sliced array keeps its whole `codes` child, so the
// decode runs over the full stream; the window's token bounds are read from
// the (possibly slice-narrowed, possibly device-resident) `codes_offsets`
// child — the offsets are nondecreasing, so the window's min and max are its
// first and last elements — and each boundary's byte position is the
// whole-batch prefix from `chunk_offsets` plus a warp reduction over the
// boundary batch's head `[batch_start, boundary)`. Launched after the offsets
// sweep with one 32-thread block per boundary: block 0 resolves
// `codes_offsets[0]`, block 1 resolves `codes_offsets[last]`. The raw token
// boundary is written to `scratch[blockIdx.x]` (a signed negative offset
// sign-extends huge and fails the host's post-readback range validation; the
// reduction clamps it to `total_tokens` so a corrupt offset cannot read out
// of bounds) and the byte position to `scratch[2 + blockIdx.x]`. Block 0
// also packs the sweep's outputs into the scratch — the full decoded heap
// size (`chunk_offsets`'s last entry) into `scratch[4]` and the corruption
// status flag into `scratch[5]` — so the host gates the decode kernel on a
// single readback. A code outside the dictionary contributes zero bytes
// (the sweep already raised the status flag the host checks before trusting
// the scratch). `CodeT` is the code stream's element type (u16 natively, u8
// when the compressor narrowed the codes); `OffsetT` is the `codes_offsets`
// element type.
template <typename CodeT, typename OffsetT>
__device__ inline void onpair_window_offsets_body(const CodeT *__restrict codes,
                                                  const uint8_t *__restrict lens,
                                                  uint32_t dict_size,
                                                  const uint64_t *__restrict chunk_offsets,
                                                  uint64_t total_tokens,
                                                  const OffsetT *__restrict codes_offsets,
                                                  uint64_t last,
                                                  const uint32_t *__restrict status,
                                                  uint64_t *__restrict scratch) {
    const uint64_t requested = (uint64_t)codes_offsets[blockIdx.x == 0 ? 0 : last];
    const uint64_t boundary = requested < total_tokens ? requested : total_tokens;
    const uint64_t batch = boundary / ONPAIR_TOKENS_PER_BATCH;
    const uint64_t batch_base = batch * ONPAIR_TOKENS_PER_BATCH;
    const uint32_t lane = threadIdx.x & 31u;

    uint32_t partial = 0;
#pragma unroll
    for (uint8_t token = 0; token < 4; ++token) {
        const uint64_t i = batch_base + lane + (uint64_t)(token * 32u);
        if (i < boundary) {
            const uint32_t code = (uint32_t)codes[i];
            if (code < dict_size) {
                partial += (uint32_t)lens[code];
            }
        }
    }
#pragma unroll
    for (uint8_t offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
    }
    if (lane == 0) {
        scratch[blockIdx.x] = requested;
        scratch[2 + blockIdx.x] = chunk_offsets[batch] + (uint64_t)partial;
        if (blockIdx.x == 0) {
            const uint64_t num_batches =
                (total_tokens + ONPAIR_TOKENS_PER_BATCH - 1) / ONPAIR_TOKENS_PER_BATCH;
            scratch[4] = chunk_offsets[num_batches];
            scratch[5] = (uint64_t)*status;
        }
    }
}

// One entry point per (code width, codes_offsets ptype) pair:
// `onpair_window_offsets_{code}_{offset}`.
#define GENERATE_WINDOW_OFFSETS_KERNEL(code_suffix, CodeT, offset_suffix, OffsetT)                           \
    extern "C" __global__ void onpair_window_offsets_##code_suffix##_##offset_suffix(                        \
        const CodeT *__restrict codes,                                                                       \
        const uint8_t *__restrict lens,                                                                      \
        uint32_t dict_size,                                                                                  \
        const uint64_t *__restrict chunk_offsets,                                                            \
        uint64_t total_tokens,                                                                               \
        const OffsetT *__restrict codes_offsets,                                                             \
        uint64_t last,                                                                                       \
        const uint32_t *__restrict status,                                                                   \
        uint64_t *__restrict scratch) {                                                                      \
        onpair_window_offsets_body<CodeT, OffsetT>(                                                          \
            codes, lens, dict_size, chunk_offsets, total_tokens, codes_offsets, last, status, scratch);      \
    }

#define GENERATE_WINDOW_OFFSETS_KERNELS(code_suffix, CodeT)                                                  \
    GENERATE_WINDOW_OFFSETS_KERNEL(code_suffix, CodeT, i8, int8_t)                                           \
    GENERATE_WINDOW_OFFSETS_KERNEL(code_suffix, CodeT, i16, int16_t)                                         \
    GENERATE_WINDOW_OFFSETS_KERNEL(code_suffix, CodeT, i32, int32_t)                                         \
    GENERATE_WINDOW_OFFSETS_KERNEL(code_suffix, CodeT, i64, int64_t)                                         \
    GENERATE_WINDOW_OFFSETS_KERNEL(code_suffix, CodeT, u8, uint8_t)                                          \
    GENERATE_WINDOW_OFFSETS_KERNEL(code_suffix, CodeT, u16, uint16_t)                                        \
    GENERATE_WINDOW_OFFSETS_KERNEL(code_suffix, CodeT, u32, uint32_t)                                        \
    GENERATE_WINDOW_OFFSETS_KERNEL(code_suffix, CodeT, u64, uint64_t)

GENERATE_WINDOW_OFFSETS_KERNELS(u8, uint8_t)
GENERATE_WINDOW_OFFSETS_KERNELS(u16, uint16_t)

// Arrow/Vortex variable-length view records are 16 bytes. Values up to 12 bytes
// are stored inline after the u32 length. Longer values store their first four
// bytes, backing-buffer index, and byte offset.
constexpr uint8_t MAX_INLINED_SIZE = 12;

// Build one BinaryView over the flat decoded byte stream. Row `rid`'s bytes
// are `output_bytes[row_offsets[rid]..row_offsets[rid + 1])`; the offsets are
// the Arrow i32 offsets built on device from the lengths child. The Rust
// caller only launches this when the window fits a single backing buffer
// (`MAX_BUFFER_LEN`, i32::MAX), so every offset is non-negative and fits the
// view's u32 fields.
__device__ inline void onpair_write_view(const int32_t *__restrict row_offsets,
                                         const uint8_t *__restrict output_bytes,
                                         uint4 *__restrict views,
                                         uint64_t rid) {
    const uint32_t start = (uint32_t)row_offsets[rid];
    const uint32_t len = (uint32_t)row_offsets[rid + 1] - start;
    if (len <= MAX_INLINED_SIZE) {
        uint32_t words[3] = {0, 0, 0};
#pragma unroll
        for (uint8_t i = 0; i < MAX_INLINED_SIZE; i++) {
            if (i < len) {
                words[i >> 2] |= (uint32_t)output_bytes[start + i] << (8u * (i & 3u));
            }
        }
        views[rid] = make_uint4(len, words[0], words[1], words[2]);
        return;
    }

    const uint32_t prefix = (uint32_t)output_bytes[start] | ((uint32_t)output_bytes[start + 1] << 8u) |
                            ((uint32_t)output_bytes[start + 2] << 16u) |
                            ((uint32_t)output_bytes[start + 3] << 24u);
    views[rid] = make_uint4(len, prefix, 0, start);
}

extern "C" __global__ void onpair_build_views(const int32_t *__restrict row_offsets,
                                              const uint8_t *__restrict output_bytes,
                                              uint4 *__restrict views,
                                              uint64_t num_rows) {
    const uint64_t elements_per_block = (uint64_t)blockDim.x * ELEMENTS_PER_THREAD;
    const uint64_t block_start = (uint64_t)blockIdx.x * elements_per_block;
    const uint64_t block_end =
        (block_start + elements_per_block < num_rows) ? (block_start + elements_per_block) : num_rows;
    for (uint64_t rid = block_start + threadIdx.x; rid < block_end; rid += blockDim.x) {
        onpair_write_view(row_offsets, output_bytes, views, rid);
    }
}
