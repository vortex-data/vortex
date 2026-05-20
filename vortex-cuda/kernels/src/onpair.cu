// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair dict-decompress, thread-per-row variant. Mirrors the CPU decoder in
// `vortex-onpair/src/decode.rs::decode_rows_unchecked`. A warp-per-row variant
// lives in `onpair_warp.cu` with the SAME ABI; the benchmark compares the
// two by loading both modules and timing only the kernel itself.
//
// Assumptions (all host-staged before launch; see
// `vortex-cuda/benches/onpair_cuda.rs`):
//   * `codes`         already bit-unpacked into a contiguous uint16_t array
//                     of length codes_offsets[num_rows].
//   * `dict_bytes`    padded with >= 16 trailing zero bytes so the per-token
//                     16-byte over-copy stays in-bounds (compress.rs pads to
//                     MAX_TOKEN_SIZE = 16).
//   * `dict_table`    one uint64_t per dict entry, packed as (off << 16) | len.
//                     len <= 16 (MAX_TOKEN_SIZE), so 16 bits suffice.
//   * `output_bytes`  has at least output_offsets[num_rows] + 16 writable
//                     bytes. The host slices off the trailing 16 pad bytes
//                     before returning to the caller; they absorb the
//                     over-copy from the penultimate token in the last row.
//   * `output_offsets` prefix-sum of `uncompressed_lengths` (host-built).
//
// Per token: read code, look up (off, len) from dict_table, copy 16 bytes
// from dict_bytes+off to output_bytes+out_pos, advance out_pos by the true
// `len`. The over-copy tail is overwritten by the next token in the SAME
// row, so within-row writes are race-free.
//
// To avoid an INTER-row write race, every token copy is capped to the row's
// remaining output bytes. That preserves the fast 16-byte over-copy through
// most of the row while preventing a near-tail token from clobbering the next
// row before the row's final token runs.

template <typename OffT>
struct OnPairArgs {
    const uint16_t *__restrict codes;
    const OffT *__restrict codes_offsets;
    const uint64_t *__restrict dict_table;
    const uint8_t *__restrict dict_bytes;
    const uint64_t *__restrict output_offsets;
    const uint8_t *__restrict validity_bits;
    uint8_t *__restrict output_bytes;
};

template <typename OffT>
__device__ inline void onpair_decode_row(const OnPairArgs<OffT> &args, uint64_t sid) {
    if (((args.validity_bits[sid >> 3] >> (sid & 7u)) & 1u) == 0u) {
        return;
    }

    OffT in_pos = args.codes_offsets[sid];
    const OffT in_end = args.codes_offsets[sid + 1];
    if (in_pos >= in_end) {
        return;
    }
    uint64_t out_pos = args.output_offsets[sid];
    const uint64_t out_end = args.output_offsets[sid + 1];

    // Keep the fixed 16-byte over-copy except near the row tail, where it
    // would otherwise spill into the next row's output bytes.
    while (in_pos + 1 < in_end) {
        const uint16_t code = args.codes[in_pos];
        const uint64_t entry = args.dict_table[code];
        const uint32_t off = (uint32_t)(entry >> 16);
        const uint32_t len = (uint32_t)(entry & 0xffffu);
        const uint32_t remaining = (uint32_t)(out_end - out_pos);
        const uint32_t copy_len = remaining < 16u ? remaining : 16u;
        memcpy(args.output_bytes + out_pos, args.dict_bytes + off, copy_len);
        out_pos += len;
        in_pos += (OffT)1;
    }

    // Last token: write only its true length.
    const uint16_t code = args.codes[in_pos];
    const uint64_t entry = args.dict_table[code];
    const uint32_t off = (uint32_t)(entry >> 16);
    const uint32_t len = (uint32_t)(entry & 0xffffu);
    memcpy(args.output_bytes + out_pos, args.dict_bytes + off, len);
}

#define GENERATE_ONPAIR_KERNEL(suffix, OffT)                                                       \
    extern "C" __global__ void onpair_##suffix(const uint16_t *__restrict codes,                   \
                                               const OffT *__restrict codes_offsets,               \
                                               const uint64_t *__restrict dict_table,              \
                                               const uint8_t *__restrict dict_bytes,               \
                                               const uint64_t *__restrict output_offsets,          \
                                               const uint8_t *__restrict validity_bits,            \
                                               uint8_t *__restrict output_bytes,                   \
                                               uint64_t num_rows) {                                \
        const OnPairArgs<OffT> args = {                                                            \
            codes,          codes_offsets, dict_table,   dict_bytes,                               \
            output_offsets, validity_bits, output_bytes,                                           \
        };                                                                                         \
                                                                                                   \
        const uint64_t elements_per_block = (uint64_t)blockDim.x * ELEMENTS_PER_THREAD;            \
        const uint64_t block_start = (uint64_t)blockIdx.x * elements_per_block;                    \
        const uint64_t block_end = (block_start + elements_per_block < num_rows)                   \
                                       ? (block_start + elements_per_block)                        \
                                       : num_rows;                                                 \
                                                                                                   \
        for (uint64_t sid = block_start + threadIdx.x; sid < block_end; sid += blockDim.x) {       \
            onpair_decode_row<OffT>(args, sid);                                                    \
        }                                                                                          \
    }

GENERATE_ONPAIR_KERNEL(u8, uint8_t)
GENERATE_ONPAIR_KERNEL(u16, uint16_t)
GENERATE_ONPAIR_KERNEL(u32, uint32_t)
GENERATE_ONPAIR_KERNEL(u64, uint64_t)
