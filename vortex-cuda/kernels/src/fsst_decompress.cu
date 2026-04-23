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
// symbols[i] is the 8-byte symbol for code i, stored little-endian in a u64.
// symbol_lengths[i] is the symbol's valid byte count (1-8). Code 255 is the
// escape marker: the next input byte is emitted as a literal.
//
// Emission uses an 8-byte fat store whenever (a) `out_pos` is 8-byte aligned
// (CUDA hardware rejects misaligned u64 stores) and (b) at least 8 bytes of
// this split's output region remain. Misaligned positions and the last
// up-to-7 bytes of each split fall back to a scalar loop.
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

        while (in_pos < in_end) {
            const uint8_t code = codes_bytes[in_pos];
            if (code == 255) {
                output_bytes[out_pos] = codes_bytes[in_pos + 1];
                in_pos += 2;
                out_pos += 1;
            } else {
                const uint64_t symbol = sm_symbols[code];
                const uint8_t sym_len = sm_symbol_lengths[code];
                if ((out_pos & 7) == 0 && out_pos + 8 <= out_end) {
                    *reinterpret_cast<uint64_t *>(output_bytes + out_pos) = symbol;
                } else {
#pragma unroll 1
                    for (uint8_t i = 0; i < sym_len; ++i) {
                        output_bytes[out_pos + i] = (uint8_t)(symbol >> (8 * i));
                    }
                }
                in_pos += 1;
                out_pos += sym_len;
            }
        }
    }
}
