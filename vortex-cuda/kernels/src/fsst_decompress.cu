// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// FSST thread-per-string decompression.
//
// symbols[i] is the 8-byte symbol for code i, stored little-endian in a u64:
// byte 0 lives in bits 0-7, byte 1 in bits 8-15, etc. symbol_lengths[i] is the
// symbol's valid byte count (1-8). Code 255 is the escape marker: the next
// input byte is emitted as a literal.
//
// Grid-stride: each block handles blockDim.x * ELEMENTS_PER_THREAD strings.
//
// The 256-entry symbol table is cooperatively loaded into shared memory before
// decoding begins, so every per-code lookup in the inner loop hits SRAM.
//
// Symbol emission uses an 8-byte fat store whenever (a) `out_pos` is 8-byte
// aligned (CUDA hardware rejects misaligned u64 stores with
// CUDA_ERROR_MISALIGNED_ADDRESS) and (b) at least 8 bytes of this thread's
// own output region remain. The store writes the full symbol u64 (valid
// bytes plus garbage from the top); `out_pos` advances by sym_len so a
// subsequent fat store overwrites the garbage. Misaligned positions and the
// last up-to-7 bytes of each string fall back to a scalar byte loop.
extern "C" __global__ void fsst_decompress(const uint8_t *__restrict codes_bytes,
                                           const int32_t *__restrict codes_offsets,
                                           const uint64_t *__restrict symbols,
                                           const uint8_t *__restrict symbol_lengths,
                                           const int32_t *__restrict output_offsets,
                                           uint8_t *__restrict output_bytes,
                                           uint64_t num_strings) {
    __shared__ uint64_t sm_symbols[256];
    __shared__ uint8_t sm_symbol_lengths[256];

    for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) {
        sm_symbols[i] = symbols[i];
        sm_symbol_lengths[i] = symbol_lengths[i];
    }
    __syncthreads();

    const uint64_t elements_per_block = (uint64_t)blockDim.x * ELEMENTS_PER_THREAD;
    const uint64_t block_start = (uint64_t)blockIdx.x * elements_per_block;
    const uint64_t block_end = (block_start + elements_per_block < num_strings)
                                   ? (block_start + elements_per_block)
                                   : num_strings;

    for (uint64_t tid = block_start + threadIdx.x; tid < block_end; tid += blockDim.x) {
        const int32_t in_end = codes_offsets[tid + 1];
        const int32_t out_end = output_offsets[tid + 1];
        int32_t in_pos = codes_offsets[tid];
        int32_t out_pos = output_offsets[tid];

        while (in_pos < in_end) {
            const uint8_t code = codes_bytes[in_pos];
            if (code == 255) {
                // Escape: the next input byte is emitted as a literal.
                output_bytes[out_pos] = codes_bytes[in_pos + 1];
                in_pos += 2;
                out_pos += 1;
            } else {
                const uint64_t symbol = sm_symbols[code];
                const uint8_t sym_len = sm_symbol_lengths[code];
                if ((out_pos & 7) == 0 && out_pos + 8 <= out_end) {
                    *reinterpret_cast<uint64_t *>(output_bytes + out_pos) = symbol;
                } else {
                    // `#pragma unroll 1` prevents nvcc at -O3 from unrolling and
                    // fusing these byte stores into a wider store that would hit
                    // the alignment fault we took this branch to avoid.
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
