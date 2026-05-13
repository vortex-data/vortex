// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// FSST decompression. A thread decodes one string at a time.
//
// Byte-by-byte global writes; no per-thread output scratch and no
// alignment-aware stores yet. The 256-entry symbol table is cooperatively
// loaded into shared memory before decoding begins so every per-code
// lookup in the inner loop hits SRAM.
//
// The compressed code stream is read one byte at a time from global
// memory. The input-side `chunk` staging from the previous commit is
// dropped here so this commit's number isolates the shared-memory
// symbol-table change; chunk-loading is re-added later on top of the
// split-based kernel.
//
// symbols[i] is the 8-byte symbol for code i, stored little-endian in a
// u64: byte 0 lives in bits 0-7, byte 1 in bits 8-15, etc.
// symbol_lengths[i] is the symbol's valid byte count (1-8). Code 255 is
// the escape marker: the next input byte is emitted as a literal.
//
// codes_offsets is templated over the four unsigned integer widths
// (u8/u16/u32/u64). output_offsets is uint64_t.

template <typename OffT>
struct FSSTArgs {
    // Compressed FSST code stream, contiguous across all strings. String
    // `sid`'s codes live in `[codes_offsets[sid], codes_offsets[sid + 1])`.
    const uint8_t *__restrict codes_bytes;
    // Per-string offsets into `codes_bytes`, length `num_strings + 1`.
    const OffT *__restrict codes_offsets;
    // FSST symbol table.
    const uint64_t *__restrict symbols;
    // Length in bytes (1..=8) of each entry in `symbols`. The remaining bits
    // are unspecified.
    const uint8_t *__restrict symbol_lengths;
    // Buffer to write decoded data into.
    uint8_t *__restrict output_bytes;
    // Per-string offsets into `output_bytes`, length `num_strings + 1`.
    const uint64_t *__restrict output_offsets;
    // Validity of each string.
    const uint8_t *__restrict validity_bits;
};

template <typename OffT>
__device__ inline void fsst_decode_string(const FSSTArgs<OffT> &args,
                                          const uint64_t *sm_symbols,
                                          const uint8_t *sm_symbol_lengths,
                                          uint64_t sid) {
    if (((args.validity_bits[sid >> 3] >> (sid & 7u)) & 1u) == 0u) {
        return;
    }

    OffT in_pos = args.codes_offsets[sid];
    const OffT in_end = args.codes_offsets[sid + 1];
    uint64_t out_pos = args.output_offsets[sid];

    while (in_pos < in_end) {
        const uint8_t code = args.codes_bytes[in_pos];
        if (code == 255) {
            // Escape: next byte is a literal.
            args.output_bytes[out_pos] = args.codes_bytes[in_pos + 1];
            in_pos += (OffT)2;
            out_pos += 1;
        } else {
            const uint64_t sym = sm_symbols[code];
            const uint8_t len = sm_symbol_lengths[code];
#pragma unroll 1
            for (uint8_t i = 0; i < len; ++i) {
                args.output_bytes[out_pos + i] = (uint8_t)(sym >> (8u * i));
            }
            in_pos += (OffT)1;
            out_pos += len;
        }
    }
}

#define GENERATE_FSST_KERNEL(suffix, OffT)                                                                   \
    extern "C" __global__ void fsst_##suffix(const uint8_t *__restrict codes_bytes,                          \
                                             const OffT *__restrict codes_offsets,                           \
                                             const uint64_t *__restrict symbols,                             \
                                             const uint8_t *__restrict symbol_lengths,                       \
                                             const uint64_t *__restrict output_offsets,                      \
                                             const uint8_t *__restrict validity_bits,                        \
                                             uint8_t *__restrict output_bytes,                               \
                                             uint64_t num_strings) {                                         \
        const FSSTArgs<OffT> args = {                                                                        \
            codes_bytes,                                                                                     \
            codes_offsets,                                                                                   \
            symbols,                                                                                         \
            symbol_lengths,                                                                                  \
            output_bytes,                                                                                    \
            output_offsets,                                                                                  \
            validity_bits,                                                                                   \
        };                                                                                                   \
                                                                                                             \
        __shared__ uint64_t sm_symbols[256];                                                                 \
        __shared__ uint8_t sm_symbol_lengths[256];                                                           \
        for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) {                                           \
            sm_symbols[i] = symbols[i];                                                                      \
            sm_symbol_lengths[i] = symbol_lengths[i];                                                        \
        }                                                                                                    \
        __syncthreads();                                                                                     \
                                                                                                             \
        const uint64_t elements_per_block = (uint64_t)blockDim.x * ELEMENTS_PER_THREAD;                      \
        const uint64_t block_start = (uint64_t)blockIdx.x * elements_per_block;                              \
        const uint64_t block_end = (block_start + elements_per_block < num_strings)                          \
                                       ? (block_start + elements_per_block)                                  \
                                       : num_strings;                                                        \
                                                                                                             \
        for (uint64_t sid = block_start + threadIdx.x; sid < block_end; sid += blockDim.x) {                 \
            fsst_decode_string<OffT>(args, sm_symbols, sm_symbol_lengths, sid);                              \
        }                                                                                                    \
    }

GENERATE_FSST_KERNEL(u8, uint8_t)
GENERATE_FSST_KERNEL(u16, uint16_t)
GENERATE_FSST_KERNEL(u32, uint32_t)
GENERATE_FSST_KERNEL(u64, uint64_t)
