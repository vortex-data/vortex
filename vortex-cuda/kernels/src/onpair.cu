// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompression. One thread decodes one row at a time.
//
// On-disk format (mirrors `onpair_lib::Parts`):
//   * `packed`     — LSB-first bit-packed token stream, one u64 of zero
//                    sentinel appended past the last real token (so the
//                    unaligned 32-bit reads here can safely over-read up to
//                    3 bytes).
//   * `boundaries` — per-row token-stream offsets, length `num_rows + 1`.
//                    `boundaries[row]` is the first token index of `row`;
//                    `boundaries[num_rows]` is the total token count.
//   * `dict_table` — per-token `(byte_offset << 16) | byte_length`. Length
//                    `dict_size`. Lengths are bounded by `MAX_TOKEN_SIZE = 16`
//                    so 16 bits suffice. Built host-side from `dict_offsets`.
//   * `dict_bytes` — flat byte buffer for dictionary entries, padded with at
//                    least 16 trailing zeros so the unconditional 16-byte
//                    over-copy per token never reads out of bounds.
//
// Two kernels are emitted per bit width:
//
//   * `onpair_lengths_b<B>`: writes per-row decoded byte counts to
//     `row_lengths[row]`. The host runs an exclusive scan over this on
//     return to derive per-row `output_offsets`. Moving this pass off the
//     CPU is a 5–10× speedup on multi-million-row columns vs. walking the
//     bitstream on one host thread.
//
//   * `onpair_decode_b<B>`: writes decoded bytes to
//     `output_bytes[output_offsets[row]..]`. Reads dict entries through the
//     read-only data cache (`__ldg`) and emits the 16-byte over-copy as a
//     single `memcpy` so nvcc picks the widest aligned store the runtime
//     pointer alignment allows (typically `st.global.v4.u32` for 16-byte-
//     aligned destinations, falling back to narrower stores otherwise).
//
// `BITS` is the column's token bit width, fixed in `9..=16` at compress
// time. Both kernels are monomorphised over all 8 values so every shift /
// mask folds to a literal — same effect as the CPU `dispatch_bits!` macro.

// Read a BITS-wide token from a byte-addressed view of the packed buffer at
// LSB-first bit position `bit_pos`. The 4-byte unaligned load may extend up
// to 3 bytes past the last real token byte; the trailing zero-sentinel u64
// `BitWriter` always emits keeps that read in-bounds.
template <uint32_t BITS>
__device__ __forceinline__ uint32_t onpair_read_token(
    const uint8_t *__restrict__ packed_bytes, uint64_t bit_pos) {
    const uint64_t byte_off = bit_pos >> 3;
    const uint32_t bit_off = static_cast<uint32_t>(bit_pos & 7u);
    uint32_t raw;
    memcpy(&raw, packed_bytes + byte_off, sizeof(uint32_t));
    const uint32_t mask = (1u << BITS) - 1u;
    return (raw >> bit_off) & mask;
}

// Sum decoded byte lengths for a row's token range. Hot loop reads
// `dict_table` through the read-only cache.
template <uint32_t BITS>
__device__ __forceinline__ uint32_t onpair_row_decoded_len(
    const uint64_t *__restrict__ dict_table,
    const uint8_t *__restrict__ packed_bytes,
    uint32_t tok_begin,
    uint32_t tok_end) {
    uint32_t total = 0;
    for (uint32_t t = tok_begin; t < tok_end; ++t) {
        const uint64_t bit_pos = static_cast<uint64_t>(t) * BITS;
        const uint32_t code = onpair_read_token<BITS>(packed_bytes, bit_pos);
        const uint64_t entry = __ldg(dict_table + code);
        total += static_cast<uint32_t>(entry & 0xffffu);
    }
    return total;
}

// Decode one row's token range `[tok_begin, tok_end)` into `out`. Each
// token triggers a fixed 16-byte over-copy from the dictionary followed by
// `cur += len` to discard the over-copy tail. Mirrors the CPU
// `decode_span_unchecked` hot loop. Both `dict_table` and `dict_bytes` are
// fetched through `__ldg` (read-only data cache) so the per-block working
// set hits the texture-style cache instead of the load/store unit's normal
// L1 path; on A100/H100 this measures consistently better when the dict is
// reused across many threads.
template <uint32_t BITS>
__device__ __forceinline__ void onpair_decode_row_inner(
    const uint64_t *__restrict__ dict_table,
    const uint8_t *__restrict__ dict_bytes,
    const uint8_t *__restrict__ packed_bytes,
    uint32_t tok_begin,
    uint32_t tok_end,
    uint8_t *__restrict__ out) {
    uint8_t *cur = out;
    for (uint32_t t = tok_begin; t < tok_end; ++t) {
        const uint64_t bit_pos = static_cast<uint64_t>(t) * BITS;
        const uint32_t code = onpair_read_token<BITS>(packed_bytes, bit_pos);
        const uint64_t entry = __ldg(dict_table + code);
        const uint32_t off = static_cast<uint32_t>(entry >> 16);
        const uint32_t len = static_cast<uint32_t>(entry & 0xffffu);
        // 16-byte over-copy from `dict_bytes + off` to `cur`. `memcpy` of a
        // compile-time-fixed 16 bytes is recognised by nvcc as a `ld.v4.u32`
        // / `st.v4.u32` pair when alignment is provable, and degrades to
        // narrower stores otherwise. Marking source `const __restrict__`
        // lets nvcc emit `LDG` (read-only cache) on Ampere / Hopper.
        memcpy(cur, dict_bytes + off, 16);
        cur += len;
    }
}

#define ONPAIR_BLOCK_SPAN()                                                                          \
    const uint64_t elems_per_block = (uint64_t)blockDim.x * ELEMENTS_PER_THREAD;                     \
    const uint64_t block_start = (uint64_t)blockIdx.x * elems_per_block;                             \
    const uint64_t block_end = (block_start + elems_per_block < num_rows)                            \
                                   ? (block_start + elems_per_block)                                 \
                                   : num_rows

#define GEN_ONPAIR_KERNELS(B)                                                                        \
    extern "C" __global__ void onpair_lengths_b##B(                                                  \
        const uint64_t *__restrict__ dict_table,                                                     \
        const uint64_t *__restrict__ packed,                                                         \
        const uint32_t *__restrict__ boundaries,                                                     \
        uint32_t *__restrict__ row_lengths,                                                          \
        uint64_t num_rows) {                                                                         \
        const uint8_t *packed_bytes = reinterpret_cast<const uint8_t *>(packed);                     \
        ONPAIR_BLOCK_SPAN();                                                                         \
        for (uint64_t row = block_start + threadIdx.x; row < block_end; row += blockDim.x) {         \
            row_lengths[row] = onpair_row_decoded_len<B>(                                            \
                dict_table, packed_bytes, boundaries[row], boundaries[row + 1]);                     \
        }                                                                                            \
    }                                                                                                \
    extern "C" __global__ void onpair_decode_b##B(                                                   \
        const uint64_t *__restrict__ dict_table,                                                     \
        const uint8_t *__restrict__ dict_bytes,                                                      \
        const uint64_t *__restrict__ packed,                                                         \
        const uint32_t *__restrict__ boundaries,                                                     \
        const uint64_t *__restrict__ output_offsets,                                                 \
        uint8_t *__restrict__ output_bytes,                                                          \
        uint64_t num_rows) {                                                                         \
        const uint8_t *packed_bytes = reinterpret_cast<const uint8_t *>(packed);                     \
        ONPAIR_BLOCK_SPAN();                                                                         \
        for (uint64_t row = block_start + threadIdx.x; row < block_end; row += blockDim.x) {         \
            onpair_decode_row_inner<B>(                                                              \
                dict_table,                                                                          \
                dict_bytes,                                                                          \
                packed_bytes,                                                                        \
                boundaries[row],                                                                     \
                boundaries[row + 1],                                                                 \
                output_bytes + output_offsets[row]);                                                 \
        }                                                                                            \
    }

GEN_ONPAIR_KERNELS(9)
GEN_ONPAIR_KERNELS(10)
GEN_ONPAIR_KERNELS(11)
GEN_ONPAIR_KERNELS(12)
GEN_ONPAIR_KERNELS(13)
GEN_ONPAIR_KERNELS(14)
GEN_ONPAIR_KERNELS(15)
GEN_ONPAIR_KERNELS(16)
