// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include "varbinview.cuh"

// single-threaded, compute offsets
extern "C" __global__ void varbinview_compute_offsets(
    const BinaryView *views,
    int64_t num_strings,
    Offsets out_offsets,
    int32_t *last_offset
) {
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // force execution to be single-threaded to compute the prefix
    // sum.
    // TODO(aduffy): we could do this with a CUB kernel instead.
    //  Check the profiles later to see where this shows up.
    if (tid != 0) {
        return;
    }

    int32_t offset = 0;
    out_offsets[0] = 0;
    for (int i = 0; i < num_strings; i++) {
        offset += views[i].inlined.size;
        out_offsets[i + 1] = offset;
    }

    *last_offset = offset;
}
