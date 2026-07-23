// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"

#include <stdint.h>

// Support kernels for OnPair GPU decompression.
//
// The decode kernel (`onpair_shmem_4tpt_split8read.cu`) consumes per-batch output
// offsets: `chunk_offsets[b]` is the count of decoded bytes preceding the b-th
// 128-token batch. Vortex does not store those offsets; they are regenerated on
// the GPU at decode time. `onpair_batch_sizes` reduces each batch's decoded size
// from the codes and the per-token length LUT, and a CUB exclusive scan over the
// result yields `chunk_offsets`. This scans only the compressed codes — it
// touches neither the dictionary bytes nor the output.

// Tokens per decode batch: one warp of the decode kernel emits 128 tokens
// (4 tokens/thread). Must match the decode kernel's layout.
constexpr uint32_t ONPAIR_TOKENS_PER_BATCH = 128;

// One warp per 128-token batch: sums `lens[codes[t]]` over the batch's (up to)
// 128 tokens and writes the total to `batch_sizes[b]`. Code reads are
// lane-consecutive (coalesced); the length LUT is small and cache-resident.
//
// A code outside the dictionary raises `status` to 1 and contributes zero
// bytes: the host must check the flag before trusting `batch_sizes` and before
// launching the decode kernel, whose dictionary gathers are unchecked.
extern "C" __global__ void onpair_batch_sizes(const uint16_t *__restrict codes,
                                              const uint8_t *__restrict lens,
                                              uint32_t dict_size, uint64_t total_tokens,
                                              uint64_t *__restrict batch_sizes,
                                              uint32_t *__restrict status) {
    const int lane = threadIdx.x & 31;
    const uint32_t warp = threadIdx.x >> 5;
    const uint64_t b = (uint64_t)blockIdx.x * (uint64_t)(blockDim.x >> 5) + (uint64_t)warp;
    const uint64_t base = b * (uint64_t)ONPAIR_TOKENS_PER_BATCH;
    if (base >= total_tokens) {
        return;
    }
    uint32_t s = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint64_t i = base + (uint64_t)lane + (uint64_t)(k * 32);
        if (i < total_tokens) {
            const uint32_t code = (uint32_t)codes[i];
            if (code < dict_size) {
                s += (uint32_t)lens[code];
            } else {
                atomicMax(status, 1u);
            }
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, offset);
    }
    if (lane == 0) {
        batch_sizes[b] = (uint64_t)s;
    }
}

// Widen the per-row decoded lengths to the u64 scan input `row_sizes`. A CUB
// exclusive scan over the result (with one extra zeroed slot) yields the u64
// per-row output offsets and, in the last slot, the total decoded byte count.
// A negative length raises `status` to 2 and contributes zero bytes; the host
// must check the flag before trusting the offsets.
template <typename T>
__device__ inline void onpair_row_sizes_impl(const T *__restrict lengths,
                                             uint64_t *__restrict row_sizes,
                                             uint32_t *__restrict status, uint64_t num_rows) {
    const uint64_t elements_per_block = (uint64_t)blockDim.x * ELEMENTS_PER_THREAD;
    const uint64_t block_start = (uint64_t)blockIdx.x * elements_per_block;
    const uint64_t block_end =
        (block_start + elements_per_block < num_rows) ? (block_start + elements_per_block) : num_rows;
    for (uint64_t i = block_start + threadIdx.x; i < block_end; i += blockDim.x) {
        T len = lengths[i];
        if constexpr (static_cast<T>(-1) < static_cast<T>(0)) {
            if (len < static_cast<T>(0)) {
                atomicMax(status, 2u);
                len = static_cast<T>(0);
            }
        }
        row_sizes[i] = (uint64_t)len;
    }
}

#define GENERATE_ONPAIR_ROW_SIZES_KERNEL(suffix, Type)                                             \
    extern "C" __global__ void onpair_row_sizes_##suffix(                                          \
        const Type *__restrict lengths, uint64_t *__restrict row_sizes,                            \
        uint32_t *__restrict status, uint64_t num_rows) {                                          \
        onpair_row_sizes_impl<Type>(lengths, row_sizes, status, num_rows);                         \
    }

GENERATE_ONPAIR_ROW_SIZES_KERNEL(u8, uint8_t)
GENERATE_ONPAIR_ROW_SIZES_KERNEL(u16, uint16_t)
GENERATE_ONPAIR_ROW_SIZES_KERNEL(u32, uint32_t)
GENERATE_ONPAIR_ROW_SIZES_KERNEL(u64, uint64_t)
GENERATE_ONPAIR_ROW_SIZES_KERNEL(i8, int8_t)
GENERATE_ONPAIR_ROW_SIZES_KERNEL(i16, int16_t)
GENERATE_ONPAIR_ROW_SIZES_KERNEL(i32, int32_t)
GENERATE_ONPAIR_ROW_SIZES_KERNEL(i64, int64_t)

// Narrow the u64 row offsets to the i32 Arrow `Utf8`/`Binary` offsets buffer.
// The host only launches this after checking the total decoded size fits i32,
// and offsets are nondecreasing, so every value fits.
extern "C" __global__ void onpair_offsets_to_i32(const uint64_t *__restrict row_offsets,
                                                 int32_t *__restrict arrow_offsets,
                                                 uint64_t num_offsets) {
    const uint64_t elements_per_block = (uint64_t)blockDim.x * ELEMENTS_PER_THREAD;
    const uint64_t block_start = (uint64_t)blockIdx.x * elements_per_block;
    const uint64_t block_end = (block_start + elements_per_block < num_offsets)
                                   ? (block_start + elements_per_block)
                                   : num_offsets;
    for (uint64_t i = block_start + threadIdx.x; i < block_end; i += blockDim.x) {
        arrow_offsets[i] = (int32_t)row_offsets[i];
    }
}

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
                                         uint4 *__restrict views, uint64_t rid) {
    const uint64_t start = row_offsets[rid];
    const uint32_t len = (uint32_t)(row_offsets[rid + 1] - start);
    if (len <= MAX_INLINED_SIZE) {
        uint32_t words[3] = {0, 0, 0};
#pragma unroll
        for (uint32_t i = 0; i < MAX_INLINED_SIZE; i++) {
            if (i < len) {
                words[i >> 2] |= (uint32_t)output_bytes[start + i] << (8u * (i & 3u));
            }
        }
        views[rid] = make_uint4(len, words[0], words[1], words[2]);
        return;
    }

    const uint32_t prefix =
        (uint32_t)output_bytes[start] | ((uint32_t)output_bytes[start + 1] << 8u) |
        ((uint32_t)output_bytes[start + 2] << 16u) | ((uint32_t)output_bytes[start + 3] << 24u);
    views[rid] = make_uint4(len, prefix, 0, (uint32_t)start);
}

extern "C" __global__ void onpair_build_views(const uint64_t *__restrict row_offsets,
                                              const uint8_t *__restrict output_bytes,
                                              uint4 *__restrict views, uint64_t num_rows) {
    const uint64_t elements_per_block = (uint64_t)blockDim.x * ELEMENTS_PER_THREAD;
    const uint64_t block_start = (uint64_t)blockIdx.x * elements_per_block;
    const uint64_t block_end =
        (block_start + elements_per_block < num_rows) ? (block_start + elements_per_block) : num_rows;
    for (uint64_t rid = block_start + threadIdx.x; rid < block_end; rid += blockDim.x) {
        onpair_write_view(row_offsets, output_bytes, views, rid);
    }
}
