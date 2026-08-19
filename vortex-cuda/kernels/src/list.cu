// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include "types.cuh"

// Converts Arrow-style `List` offsets into `ListView` offset/size pairs.
//
// `List` stores `list_len + 1` monotonically increasing offsets; a `ListView` stores one offset
// and one size per list. Both outputs are written by the same thread so the two views of a list
// are always produced together.
template <typename OffsetT>
__device__ void list_views(const OffsetT *const __restrict offsets,
                           OffsetT *const __restrict out_offsets,
                           OffsetT *const __restrict out_sizes,
                           uint64_t list_len) {
    const uint32_t elements_per_block = blockDim.x * ELEMENTS_PER_THREAD;
    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * elements_per_block;
    const uint64_t block_end = min(block_start + elements_per_block, list_len);

    for (uint64_t idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
        const OffsetT start = offsets[idx];
        out_offsets[idx] = start;
        out_sizes[idx] = static_cast<OffsetT>(offsets[idx + 1] - start);
    }
}

#define GENERATE_LIST_VIEWS_KERNEL(offset_suffix, OffsetT)                                                   \
    extern "C" __global__ void list_views_##offset_suffix(const OffsetT *const __restrict offsets,           \
                                                          OffsetT *const __restrict out_offsets,             \
                                                          OffsetT *const __restrict out_sizes,               \
                                                          uint64_t list_len) {                               \
        list_views<OffsetT>(offsets, out_offsets, out_sizes, list_len);                                      \
    }

FOR_EACH_INTEGER(GENERATE_LIST_VIEWS_KERNEL)
