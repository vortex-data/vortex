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

// The token window of this array's rows, resolved entirely on device from the
// (possibly slice-narrowed, possibly device-resident) `codes_offsets` child:
// the offsets are nondecreasing, so the window's min and max are its first
// and last elements. Writes `bounds[0] = codes_offsets[0]` and
// `bounds[1] = codes_offsets[last]`; a signed negative offset sign-extends
// huge and fails the host's post-readback range validation.
#define GENERATE_TOKEN_BOUNDS_KERNEL(suffix, OffsetT)                                                        \
    extern "C" __global__ void onpair_token_bounds_##suffix(const OffsetT *__restrict codes_offsets,         \
                                                            uint64_t last,                                   \
                                                            uint64_t *__restrict bounds) {                   \
        if (threadIdx.x == 0 && blockIdx.x == 0) {                                                           \
            bounds[0] = (uint64_t)codes_offsets[0];                                                          \
            bounds[1] = (uint64_t)codes_offsets[last];                                                       \
        }                                                                                                    \
    }

GENERATE_TOKEN_BOUNDS_KERNEL(i8, int8_t)
GENERATE_TOKEN_BOUNDS_KERNEL(i16, int16_t)
GENERATE_TOKEN_BOUNDS_KERNEL(i32, int32_t)
GENERATE_TOKEN_BOUNDS_KERNEL(i64, int64_t)
GENERATE_TOKEN_BOUNDS_KERNEL(u8, uint8_t)
GENERATE_TOKEN_BOUNDS_KERNEL(u16, uint16_t)
GENERATE_TOKEN_BOUNDS_KERNEL(u32, uint32_t)
GENERATE_TOKEN_BOUNDS_KERNEL(u64, uint64_t)

// Byte positions of the visible code window's bounds in the full decoded
// stream. A sliced array keeps its whole `codes` child, so the decode runs
// over the full stream; each boundary's byte position is the whole-batch
// prefix from `chunk_offsets` plus a warp reduction over the boundary batch's
// head `[batch_start, boundary)`. Launched after the offsets sweep and the
// token-bounds resolution with one 32-thread block per boundary; the token
// boundaries are read from `bounds[0..2)` (clamped to `total_tokens` so a
// corrupt offset cannot read out of bounds — the host rejects it after
// readback) and the byte positions are written to `bounds[2 + blockIdx.x]`.
// A code outside the dictionary contributes zero bytes (the sweep already
// raised the status flag the host checks before trusting `bounds`). `CodeT`
// is the code stream's element type (u16 natively, u8 when the compressor
// narrowed the codes).
template <typename CodeT>
__device__ inline void onpair_window_offsets_body(const CodeT *__restrict codes,
                                                  const uint8_t *__restrict lens,
                                                  uint32_t dict_size,
                                                  const uint64_t *__restrict chunk_offsets,
                                                  uint64_t total_tokens,
                                                  uint64_t *__restrict bounds) {
    const uint64_t requested = bounds[blockIdx.x];
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
        bounds[2 + blockIdx.x] = chunk_offsets[batch] + (uint64_t)partial;
    }
}

extern "C" __global__ void onpair_window_offsets_u8(const uint8_t *__restrict codes,
                                                    const uint8_t *__restrict lens,
                                                    uint32_t dict_size,
                                                    const uint64_t *__restrict chunk_offsets,
                                                    uint64_t total_tokens,
                                                    uint64_t *__restrict bounds) {
    onpair_window_offsets_body<uint8_t>(codes, lens, dict_size, chunk_offsets, total_tokens, bounds);
}

extern "C" __global__ void onpair_window_offsets_u16(const uint16_t *__restrict codes,
                                                     const uint8_t *__restrict lens,
                                                     uint32_t dict_size,
                                                     const uint64_t *__restrict chunk_offsets,
                                                     uint64_t total_tokens,
                                                     uint64_t *__restrict bounds) {
    onpair_window_offsets_body<uint16_t>(codes, lens, dict_size, chunk_offsets, total_tokens, bounds);
}

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
