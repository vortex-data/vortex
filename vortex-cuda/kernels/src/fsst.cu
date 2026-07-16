// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// FSST decompression. A thread decodes one string at a time.
//
// Per-thread `Scratch` holds 24 bytes across three u64 lanes (`low`, `mid`,
// `high`) plus a `cursor` byte counter. Byte i lives at bit (8 * (i mod 8))
// of:
//   low   for i in 0..8
//   mid   for i in 8..16
//   high  for i in 16..24
//
//          lsb                                 msb
//   low:  [ b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7 ]
//   mid:  [ b8 | b9 |b10 |b11 |b12 |b13 |b14 |b15 ]
//   high: [b16 |b17 |b18 |b19 |b20 |b21 |b22 |b23 ]
//
// `Scratch::drain` picks the largest aligned store the gates allow
// (alignment of out_pos, cursor, remaining out_end room). Bytes leave from
// the low end (`low` byte 0); the kept bytes slide N positions toward that
// low end across all three lanes i.e. each u64 right-shifts by N*8 and
// pulls the next lane's low bits up to fill the vacated high bits.
// `Scratch::push` inserts a length-`len` masked symbol at byte offset
// `cursor`, spanning at most two of the three lanes.
//
//   width   gate                                         ptx
//   ------  ------------------------------------------   ----------------
//   16 B    out_pos % 16 == 0, cursor ≥ 16, room ≥ 16    st.global.v2.u64
//    8 B    out_pos %  8 == 0, cursor ≥  8, room ≥  8    st.global.u64
//    4 B    out_pos %  4 == 0, cursor ≥  4, room ≥  4    st.global.u32
//    2 B    out_pos %  2 == 0, cursor ≥  2, room ≥  2    st.global.u16
//    1 B    (always)                                     st.global.u8
//
// The narrow widths cover the prologue alignment-up (out_pos not yet
// 16-aligned) and the epilogue tail (< 16 bytes left, no room for u128).
// In steady state out_pos stays 16-aligned and u128 fires repeatedly.
//
// The 256-entry symbol table (≤ 2 KB) is read directly from global memory.
// Staging it into shared memory measured ~3% slower at 10M rows and ~15%
// slower at 1M rows (benchmarked on clickbench URLs). The hypothesis is that L1
// already holds the table after a few iterations and the explicit shared copy
// adds bank-conflict latency on the warp-divergent `symbols[code]` reads; the
// gap is wider at 1M because the kernel is less bandwidth-bound there, so
// per-load latency shows up more.
//
// Decoded symbols are masked to their valid byte length so the table's high
// bits never leak. The main loop drains to `scratch.cursor ≤ 16`, keeping
// the next add (≤ 8 bytes) within the 24-byte capacity.
//
// `codes_offsets` is templated over the four unsigned integer widths
// (u8/u16/u32/u64). `output_offsets` is uint64_t.

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

// Arrow/Vortex variable-length view records are 16 bytes. Values up to 12
// bytes are stored inline after the u32 length. Longer values store their
// first four bytes, backing-buffer index, and byte offset.
constexpr uint32_t MAX_INLINED_SIZE = 12;

template <typename CodeOffsetT, typename OutputOffsetT>
struct FSSTArgs {
    // Compressed FSST code stream, contiguous across all strings. String
    // `sid`'s codes live in `[codes_offsets[sid], codes_offsets[sid + 1])`.
    const uint8_t *__restrict codes_bytes;
    // Per-string offsets into `codes_bytes`, length `num_strings + 1`.
    const CodeOffsetT *__restrict codes_offsets;
    // FSST symbol table.
    const uint64_t *__restrict symbols;
    // Length in bytes (1..=8) of each entry in `symbols`. The remaining bits
    // are unspecified.
    const uint8_t *__restrict symbol_lengths;
    // Buffer to write decoded data into.
    uint8_t *__restrict output_bytes;
    // Per-string offsets into `output_bytes`, length `num_strings + 1`.
    const OutputOffsetT *__restrict output_offsets;
    // Validity of each string.
    const uint8_t *__restrict validity_bits;
    // Optional output views, one 16-byte uint4 per string. A null pointer
    // requests bytes-only decoding for heaps that need the host rollover path.
    uint4 *__restrict output_views;
};

// Build one BinaryView from the bytes this thread just decoded. The Rust
// caller only provides output_views when every offset fits in the view's u32
// fields and the decoded heap is exposed as backing buffer zero.
template <typename CodeOffsetT, typename OutputOffsetT>
__device__ inline void fsst_write_view(const FSSTArgs<CodeOffsetT, OutputOffsetT> &args, uint64_t sid) {
    if (args.output_views == nullptr) {
        return;
    }

    const uint64_t start = args.output_offsets[sid];
    const uint32_t len = (uint32_t)(args.output_offsets[sid + 1] - start);
    if (len <= MAX_INLINED_SIZE) {
        uint32_t words[3] = {0, 0, 0};
#pragma unroll
        for (uint32_t i = 0; i < MAX_INLINED_SIZE; i++) {
            if (i < len) {
                words[i >> 2] |= (uint32_t)args.output_bytes[start + i] << (8u * (i & 3u));
            }
        }
        args.output_views[sid] = make_uint4(len, words[0], words[1], words[2]);
        return;
    }

    const uint32_t prefix = (uint32_t)args.output_bytes[start] |
                            ((uint32_t)args.output_bytes[start + 1] << 8u) |
                            ((uint32_t)args.output_bytes[start + 2] << 16u) |
                            ((uint32_t)args.output_bytes[start + 3] << 24u);
    args.output_views[sid] = make_uint4(len, prefix, 0, (uint32_t)start);
}

template <typename CodeOffsetT, typename OutputOffsetT>
__device__ inline void fsst_decode_string(const FSSTArgs<CodeOffsetT, OutputOffsetT> &args, uint64_t sid) {
    if (((args.validity_bits[sid >> 3] >> (sid & 7u)) & 1u) == 0u) {
        if (args.output_views != nullptr) {
            args.output_views[sid] = make_uint4(0, 0, 0, 0);
        }
        return;
    }

    CodeOffsetT in_pos = args.codes_offsets[sid];
    const CodeOffsetT in_end = args.codes_offsets[sid + 1];
    uint64_t out_pos = args.output_offsets[sid];
    const uint64_t out_end = args.output_offsets[sid + 1];

    Scratch scratch;

    while (in_pos < in_end) {
        // Drain to scratch.cursor ≤ 16 so the next ≤8-byte symbol fits in 24.
        while (scratch.cursor > 16) {
            scratch.drain(args.output_bytes, out_pos, out_end);
        }

        // Decode next code. 255 is the escape for raw literal bytes.
        const uint8_t code = args.codes_bytes[in_pos];
        uint64_t sym;
        uint32_t len, consumed;
        if (code == 255) {
            sym = (uint64_t)args.codes_bytes[in_pos + 1];
            len = 1;
            consumed = 2;
        } else {
            sym = args.symbols[code];
            len = args.symbol_lengths[code];
            consumed = 1;
        }

        // Zero out the symbol's high bytes beyond its valid length.
        const uint64_t mask = (len == 8) ? ~0ULL : ((1ULL << (8u * len)) - 1ULL);
        sym &= mask;

        scratch.push(sym, len);
        in_pos += (CodeOffsetT)consumed;
    }

    // Epilogue: drain everything that's left.
    while (scratch.cursor > 0) {
        scratch.drain(args.output_bytes, out_pos, out_end);
    }

    fsst_write_view(args, sid);
}

#define FSST_GRID_STRIDE_LOOP(CodeOffsetT, OutputOffsetT, args)                                             \
    const uint64_t elements_per_block = (uint64_t)blockDim.x * ELEMENTS_PER_THREAD;                          \
    const uint64_t block_start = (uint64_t)blockIdx.x * elements_per_block;                                  \
    const uint64_t block_end = (block_start + elements_per_block < num_strings)                              \
                                   ? (block_start + elements_per_block)                                      \
                                   : num_strings;                                                            \
    for (uint64_t sid = block_start + threadIdx.x; sid < block_end; sid += blockDim.x) {                     \
        fsst_decode_string<CodeOffsetT, OutputOffsetT>(args, sid);                                          \
    }

#define GENERATE_FSST_VIEW_KERNEL(suffix, CodeOffsetT)                                                       \
    extern "C" __global__ void fsst_##suffix(const uint8_t *__restrict codes_bytes,                          \
                                             const CodeOffsetT *__restrict codes_offsets,                    \
                                             const uint64_t *__restrict symbols,                             \
                                             const uint8_t *__restrict symbol_lengths,                       \
                                             const uint64_t *__restrict output_offsets,                      \
                                             const uint8_t *__restrict validity_bits,                        \
                                             uint8_t *__restrict output_bytes,                               \
                                             uint4 *__restrict output_views,                                 \
                                             uint64_t num_strings) {                                         \
        const FSSTArgs<CodeOffsetT, uint64_t> args = {                                                       \
            codes_bytes, codes_offsets, symbols, symbol_lengths, output_bytes, output_offsets,              \
            validity_bits, output_views,                                                                     \
        };                                                                                                   \
        FSST_GRID_STRIDE_LOOP(CodeOffsetT, uint64_t, args)                                                   \
    }

#define GENERATE_FSST_VARBIN_KERNEL(suffix, CodeOffsetT)                                                     \
    extern "C" __global__ void fsst_varbin_##suffix(const uint8_t *__restrict codes_bytes,                  \
                                                    const CodeOffsetT *__restrict codes_offsets,             \
                                                    const uint64_t *__restrict symbols,                      \
                                                    const uint8_t *__restrict symbol_lengths,                \
                                                    const int32_t *__restrict output_offsets,                \
                                                    const uint8_t *__restrict validity_bits,                 \
                                                    uint8_t *__restrict output_bytes,                        \
                                                    uint64_t num_strings) {                                  \
        const FSSTArgs<CodeOffsetT, int32_t> args = {                                                        \
            codes_bytes, codes_offsets, symbols, symbol_lengths, output_bytes, output_offsets,              \
            validity_bits, nullptr,                                                                          \
        };                                                                                                   \
        FSST_GRID_STRIDE_LOOP(CodeOffsetT, int32_t, args)                                                    \
    }

GENERATE_FSST_VIEW_KERNEL(u8, uint8_t)
GENERATE_FSST_VIEW_KERNEL(u16, uint16_t)
GENERATE_FSST_VIEW_KERNEL(u32, uint32_t)
GENERATE_FSST_VIEW_KERNEL(u64, uint64_t)

GENERATE_FSST_VARBIN_KERNEL(u8, uint8_t)
GENERATE_FSST_VARBIN_KERNEL(u16, uint16_t)
GENERATE_FSST_VARBIN_KERNEL(u32, uint32_t)
GENERATE_FSST_VARBIN_KERNEL(u64, uint64_t)
