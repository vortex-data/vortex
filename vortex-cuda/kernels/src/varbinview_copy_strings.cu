// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include "varbinview.cuh"

// Lookup a string from a binary view, copying it into
// a destination buffer.
__device__ void copy_string_to_dst(
    BinaryView& view,
    Buffer *buffers,
    uint8_t *dst
) {
    int32_t size = view.inlined.size;
    uint8_t *src;
    if (size <= MAX_INLINED_SIZE) {
        // TODO(aduffy): use uint64_t loads instead?
        src = view.inlined.data;
    } else {
        auto ref = view.ref;
        src = buffers[ref.index] + ref.offset;
    }
    memcpy(dst, src, size);
}

extern "C" __global__ void varbinview_copy_strings(
    int64_t len,
    BinaryView* views,
    Buffer* buffers,
    Buffer dst_buffer,
    Offsets dst_offsets
) {
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread is responsible for copying a single string.
    // Any excess threads do no work.
    if (tid >= len) {
        return;
    }

    auto view = views[tid];
    int32_t offset = dst_offsets[tid];
    uint8_t *dst = &dst_buffer[offset];

    copy_string_to_dst(view, buffers, dst);
}