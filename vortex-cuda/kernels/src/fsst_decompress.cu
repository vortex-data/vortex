// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// FSST decompression with split-based parallelism (GSST-style) and
// aligned-output staging via a per-thread register scratch buffer.
//
// Each thread decodes one split: [split_in_offsets[i], split_in_offsets[i+1])
// of compressed bytes into [split_out_offsets[i], split_out_offsets[i+1]).
// Splits are emitted by the CPU pre-pass at code boundaries so they never
// start mid-escape.
//
// Output alignment (GSST paper, Section 4). Writing decoded bytes directly
// to global with byte stores triggers partial-sector transactions on some
// hardware. To emit aligned 8-byte stores, we stage decoded output in a
// 16-byte per-thread scratch held in two u64 registers (scratch_lo holds
// the next 8 bytes to emit; scratch_hi holds the following 8). When scratch
// holds ≥8 bytes and out_pos is 8-aligned, we emit scratch_lo with one
// aligned st.global.u64 and shift scratch_hi down. When out_pos is not yet
// aligned (prologue) or the output tail can't fit 8 more bytes (epilogue),
// we byte-drain scratch one at a time.
//
// Symbol u64 is masked to `sym_len` valid bytes before insertion so the
// high garbage bits from the symbol table can't leak into scratch.
extern "C" __global__ void fsst_decompress(const uint8_t *__restrict codes_bytes,
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
        const int32_t in_end = split_in_offsets[sid + 1];
        const int32_t out_end = split_out_offsets[sid + 1];
        int32_t in_pos = split_in_offsets[sid];
        int32_t out_pos = split_out_offsets[sid];

        // Per-thread output scratch in registers. Holds 0..=16 bytes.
        // Layout: byte i of scratch lives in bit (8*i) of scratch_lo when
        // i<8, or bit (8*(i-8)) of scratch_hi when 8<=i<16.
        uint64_t scratch_lo = 0;
        uint64_t scratch_hi = 0;
        uint32_t scratch_bytes = 0;

        while (in_pos < in_end) {
            // Ensure scratch_bytes < 8 before appending a new (up-to-8-byte) symbol.
            while (scratch_bytes >= 8) {
                if ((out_pos & 7) == 0 && out_pos + 8 <= out_end) {
                    // Aligned 8-byte flush.
                    *reinterpret_cast<uint64_t *>(output_bytes + out_pos) = scratch_lo;
                    scratch_lo = scratch_hi;
                    scratch_hi = 0;
                    out_pos += 8;
                    scratch_bytes -= 8;
                } else {
                    // Byte-drain to advance toward alignment (or toward tail).
                    output_bytes[out_pos] = (uint8_t)(scratch_lo & 0xFFu);
                    out_pos += 1;
                    scratch_lo = (scratch_lo >> 8) | (scratch_hi << 56);
                    scratch_hi >>= 8;
                    scratch_bytes -= 1;
                }
            }

            // Decode next code into (sym, len, consumed).
            const uint8_t code = codes_bytes[in_pos];
            uint64_t sym;
            uint32_t len;
            uint32_t consumed;
            if (code == 255) {
                sym = (uint64_t)codes_bytes[in_pos + 1];
                len = 1;
                consumed = 2;
            } else {
                sym = sm_symbols[code];
                len = sm_symbol_lengths[code];
                consumed = 1;
            }

            // Mask sym to `len` valid low bytes so we don't leak garbage
            // from the high bits into scratch.
            const uint64_t mask = (len == 8) ? ~0ULL : ((1ULL << (8u * len)) - 1ULL);
            sym &= mask;

            // Append `sym` at byte offset `scratch_bytes` within scratch.
            if (scratch_bytes < 8) {
                scratch_lo |= sym << (8u * scratch_bytes);
                if (scratch_bytes + len > 8) {
                    scratch_hi |= sym >> (8u * (8u - scratch_bytes));
                }
            } else {
                scratch_hi |= sym << (8u * (scratch_bytes - 8u));
            }
            scratch_bytes += len;
            in_pos += consumed;
        }

        // Epilogue: flush scratch. Prefer one aligned 8-byte store if we
        // still have ≥8 bytes and room; otherwise byte-drain.
        while (scratch_bytes >= 8 && (out_pos & 7) == 0 && out_pos + 8 <= out_end) {
            *reinterpret_cast<uint64_t *>(output_bytes + out_pos) = scratch_lo;
            scratch_lo = scratch_hi;
            scratch_hi = 0;
            out_pos += 8;
            scratch_bytes -= 8;
        }
        while (scratch_bytes > 0) {
            output_bytes[out_pos] = (uint8_t)(scratch_lo & 0xFFu);
            out_pos += 1;
            scratch_lo = (scratch_lo >> 8) | (scratch_hi << 56);
            scratch_hi >>= 8;
            scratch_bytes -= 1;
        }
    }
}
