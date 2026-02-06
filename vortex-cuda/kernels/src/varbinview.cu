// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "varbinview.cuh"

// Lookup a string from a binary view, copying it into
// a destination buffer.
__device__ void copy_string_to_dst(
    BinaryView& view,
    Buffer[] buffers,
    uint8_t *dst
) {
    int32_t size = view.inlined.size;
    uint8_t *src;
    if (size <= MAX_INLINED_SIZE) {
        src = view.inlined.data;
    } else {
        auto ref = view.ref;
        src = buffers[ref.index] + ref.offset;
    }
    memcpy(dst, src, size);
}

// single-threaded, compute offsets
extern "C" __global__ void compute_offsets(
    const BinaryView[] views,
    int32_t num_strings,
    Offsets out_offsets
) {
    int32_t offset = 0;
    out_offsets[0] = 0;
    for (int i = 0; i < num_strings; i++) {
        offset += views[i].inlined.size;
        out_offsets[i + 1] = offset;
    }
}

extern "C" __global__ void copy_strings(
    int64_t len,
    BinaryView[] views,
    Buffer[] buffers,
    Buffer *dst_buffer,
    Offsets *dst_offsets
) {
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread is responsible for copying a single string.
    // Any excess threads do no work.
    if (tid >= len) {
        return;
    }

    auto view = views[tid];
    auto offset = dst_offsets[tid];
    uint8_t *dst = dst_buffer + offset;

    copy_string_to_dst(view, buffers, dst);
}