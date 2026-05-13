// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// FSST decompression with split-based parallelism (GSST-style).
//
// Each thread decodes one "split": a code-boundary-aligned chunk of the
// compressed stream bounded by [split_in_offsets[i], split_in_offsets[i+1]).
// Output goes to [split_out_offsets[i], split_out_offsets[i+1]). Splits are
// emitted by the CPU pre-pass every ~target_split_bytes compressed bytes at
// the next code boundary, so splits never start mid-escape.
//
// Compared to a per-string kernel, splits give near-uniform work per lane
// (bounded by the split byte cap), so warp retirement isn't gated on the
// longest string in the warp.
//
// Per-thread `Scratch` holds 24 bytes across three u64 lanes (`low`, `mid`,
// `high`) plus a `cursor` byte counter. `Scratch::drain` picks the largest
// aligned store the gates allow (alignment of out_pos, cursor, remaining
// out_end room).
//
//   width   gate                                         ptx
//   ------  ------------------------------------------   ----------------
//   16 B    out_pos % 16 == 0, cursor ≥ 16, room ≥ 16    st.global.v2.u64
//    8 B    out_pos %  8 == 0, cursor ≥  8, room ≥  8    st.global.u64
//    4 B    out_pos %  4 == 0, cursor ≥  4, room ≥  4    st.global.u32
//    2 B    out_pos %  2 == 0, cursor ≥  2, room ≥  2    st.global.u16
//    1 B    (always)                                     st.global.u8
//
// The compressed code stream is staged in a per-thread register `chunk`
// (up to 8 bytes), carried over from the per-string kernel: align-up with
// the largest aligned load at each step, then steady-state `ld.global.u64`
// refills at `chunk_bytes == 0`. The `chunk_bytes == 1 && code == 255`
// boundary case reads the literal directly from global.
//
//   load    gate                                       ptx
//   ------  -----------------------------------------  ----------------
//    u64    chunk_bytes == 0, fill_pos % 8 == 0        ld.global.u64
//    u32    chunk_bytes + 4 ≤ 8, fill_pos % 4 == 0     ld.global.u32
//    u16    chunk_bytes + 2 ≤ 8, fill_pos % 2 == 0     ld.global.u16
//    u8     (always)                                   ld.global.u8
//
// The 256-entry symbol table is cooperatively loaded into shared memory
// once per block. Per-string kernels were fine with global reads (L1
// retention), but with splits the per-thread workload is smaller and the
// table-lookup latency dominates more — shared-memory staging wins back
// what split-based parallelism would otherwise lose.

// 24-byte scratch buffer split across three u64 lanes. `cursor` is the
// number of bytes currently buffered and the next-push offset.
struct Scratch {
    uint64_t low = 0;
    uint64_t mid = 0;
    uint64_t high = 0;
    uint32_t cursor = 0;

    // Insert a length-`len` masked symbol at byte offset `cursor`. The
    // symbol spans at most two of the three lanes. Caller must ensure
    // cursor + len ≤ 24.
    __device__ inline void push(uint64_t sym, uint32_t len) {
        if (cursor < 8) {
            low |= sym << (8u * cursor);
            if (cursor + len > 8) {
                mid |= sym >> (8u * (8u - cursor));
            }
        } else {
            mid |= sym << (8u * (cursor - 8u));
            if (cursor + len > 16) {
                high |= sym >> (8u * (16u - cursor));
            }
        }
        cursor += len;
    }

    // Emit one variable-width aligned store from the low end and slide the
    // kept bytes toward the low end across all three lanes.
    __device__ inline void drain(uint8_t *__restrict out, uint64_t &out_pos, uint64_t out_end) {
        if (cursor >= 16 && (out_pos & 15u) == 0 && out_pos + 16 <= out_end) {
            *reinterpret_cast<ulonglong2 *>(out + out_pos) = make_ulonglong2(low, mid);
            low = high;
            mid = 0;
            high = 0;
            out_pos += 16;
            cursor -= 16;
        } else if (cursor >= 8 && (out_pos & 7u) == 0 && out_pos + 8 <= out_end) {
            *reinterpret_cast<uint64_t *>(out + out_pos) = low;
            low = mid;
            mid = high;
            high = 0;
            out_pos += 8;
            cursor -= 8;
        } else if (cursor >= 4 && (out_pos & 3u) == 0 && out_pos + 4 <= out_end) {
            *reinterpret_cast<uint32_t *>(out + out_pos) = (uint32_t)low;
            low = (low >> 32) | (mid << 32);
            mid = (mid >> 32) | (high << 32);
            high >>= 32;
            out_pos += 4;
            cursor -= 4;
        } else if (cursor >= 2 && (out_pos & 1u) == 0 && out_pos + 2 <= out_end) {
            *reinterpret_cast<uint16_t *>(out + out_pos) = (uint16_t)low;
            low = (low >> 16) | (mid << 48);
            mid = (mid >> 16) | (high << 48);
            high >>= 16;
            out_pos += 2;
            cursor -= 2;
        } else {
            out[out_pos] = (uint8_t)low;
            low = (low >> 8) | (mid << 56);
            mid = (mid >> 8) | (high << 56);
            high >>= 8;
            out_pos += 1;
            cursor -= 1;
        }
    }
};

// Refill `chunk` from `codes_bytes + (in_pos + chunk_bytes)`. Exits early
// once `fill_pos` reaches u64 alignment with `chunk_bytes >= 2`, so the
// next refill (at chunk_bytes == 0) lands on the aligned u64 fast path.
__device__ inline void fsst_chunk_refill(const uint8_t *__restrict codes_bytes,
                                         int32_t in_pos,
                                         int32_t in_end,
                                         uint64_t &chunk,
                                         uint32_t &chunk_bytes) {
#pragma unroll 1
    while (chunk_bytes < 8) {
        const int32_t fill_pos = in_pos + (int32_t)chunk_bytes;
        if (fill_pos >= in_end) {
            break;
        }
        const int32_t remaining = in_end - fill_pos;
        const uint32_t aln = (uint32_t)fill_pos & 7u;
        if (chunk_bytes == 0 && aln == 0 && remaining >= 8) {
            chunk = *reinterpret_cast<const uint64_t *>(codes_bytes + fill_pos);
            chunk_bytes = 8;
            return;
        }
        if (chunk_bytes >= 2 && aln == 0) {
            return;
        }
        if (chunk_bytes + 4u <= 8u && (aln & 3u) == 0 && remaining >= 4) {
            const uint64_t v = *reinterpret_cast<const uint32_t *>(codes_bytes + fill_pos);
            chunk |= v << (8u * chunk_bytes);
            chunk_bytes += 4;
        } else if (chunk_bytes + 2u <= 8u && (aln & 1u) == 0 && remaining >= 2) {
            const uint64_t v = *reinterpret_cast<const uint16_t *>(codes_bytes + fill_pos);
            chunk |= v << (8u * chunk_bytes);
            chunk_bytes += 2;
        } else {
            const uint64_t v = codes_bytes[fill_pos];
            chunk |= v << (8u * chunk_bytes);
            chunk_bytes += 1;
        }
    }
}

extern "C" __global__ void fsst(const uint8_t *__restrict codes_bytes,
                                const int32_t *__restrict split_in_offsets,
                                const int32_t *__restrict split_out_offsets,
                                const uint64_t *__restrict symbols,
                                const uint8_t *__restrict symbol_lengths,
                                uint8_t *__restrict output_bytes,
                                uint64_t num_splits) {
    __shared__ uint64_t sm_symbols[256];
    __shared__ uint8_t sm_symbol_lengths[256];
    for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) {
        sm_symbols[i] = symbols[i];
        sm_symbol_lengths[i] = symbol_lengths[i];
    }
    __syncthreads();

    const uint64_t elements_per_block = (uint64_t)blockDim.x * ELEMENTS_PER_THREAD;
    const uint64_t block_start = (uint64_t)blockIdx.x * elements_per_block;
    const uint64_t block_end = (block_start + elements_per_block < num_splits)
                                   ? (block_start + elements_per_block)
                                   : num_splits;

    for (uint64_t sid = block_start + threadIdx.x; sid < block_end; sid += blockDim.x) {
        int32_t in_pos = split_in_offsets[sid];
        const int32_t in_end = split_in_offsets[sid + 1];
        uint64_t out_pos = (uint64_t)split_out_offsets[sid];
        const uint64_t out_end = (uint64_t)split_out_offsets[sid + 1];

        Scratch scratch;
        uint64_t chunk = 0;
        uint32_t chunk_bytes = 0;

        while (in_pos < in_end) {
            // Drain to scratch.cursor ≤ 16 so the next ≤8-byte symbol fits in 24.
            while (scratch.cursor > 16) {
                scratch.drain(output_bytes, out_pos, out_end);
            }

            // Refill only when chunk is fully drained so the load lands at
            // the aligned `fill_pos` left by the previous refill's early
            // exit. `chunk_bytes == 1 && code == 255` is handled below.
            if (chunk_bytes == 0) {
                fsst_chunk_refill(codes_bytes, in_pos, in_end, chunk, chunk_bytes);
                if (chunk_bytes == 0) {
                    break;
                }
            }

            // Decode next code. 255 is the escape for raw literal bytes.
            const uint8_t code = (uint8_t)(chunk & 0xFFu);
            uint64_t sym;
            uint32_t len, consumed;
            bool boundary_escape = false;
            if (code == 255) {
                if (chunk_bytes >= 2) {
                    sym = (chunk >> 8u) & 0xFFu;
                } else {
                    sym = (uint64_t)codes_bytes[in_pos + 1];
                    boundary_escape = true;
                }
                len = 1;
                consumed = 2;
            } else {
                sym = sm_symbols[code];
                len = sm_symbol_lengths[code];
                consumed = 1;
            }

            const uint64_t mask = (len == 8) ? ~0ULL : ((1ULL << (8u * len)) - 1ULL);
            sym &= mask;

            scratch.push(sym, len);
            in_pos += (int32_t)consumed;
            if (boundary_escape) {
                chunk = 0;
                chunk_bytes = 0;
            } else {
                chunk >>= 8u * consumed;
                chunk_bytes -= consumed;
            }
        }

        // Epilogue: drain everything that's left.
        while (scratch.cursor > 0) {
            scratch.drain(output_bytes, out_pos, out_end);
        }
    }
}
