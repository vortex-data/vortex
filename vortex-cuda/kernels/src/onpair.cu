// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"

#include <stdint.h>

// Support kernels for OnPair GPU decompression. The per-batch output-offsets
// regeneration lives in the CUB shim (`cub/kernels/onpair.cu`), where the
// reduction and exclusive scan run as one fused sweep; this module holds the
// view-construction kernels launched after the decode.

// Arrow/Vortex variable-length view records are 16 bytes. Values up to 12 bytes
// are stored inline after the u32 length. Longer values store their first four
// bytes, backing-buffer index, and byte offset.
constexpr uint32_t MAX_INLINED_SIZE = 12;

// Build one BinaryView over the flat decoded byte stream. Row `rid`'s bytes are
// `output_bytes[row_offsets[rid]..row_offsets[rid + 1])`. The Rust caller only
// launches this when every offset fits the view's u32 fields and the decoded
// heap is exposed as backing buffer zero.
__device__ inline void onpair_write_view(const uint64_t *__restrict row_offsets,
                                         const uint8_t *__restrict output_bytes,
                                         uint4 *__restrict views,
                                         uint64_t rid) {
    const uint64_t start = row_offsets[rid];
    const uint32_t len = (uint32_t)(row_offsets[rid + 1] - start);
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
    views[rid] = make_uint4(len, prefix, 0, (uint32_t)start);
}

extern "C" __global__ void onpair_build_views(const uint64_t *__restrict row_offsets,
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
