// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// GENERATED — do not edit by hand. See `generate_cp_unpack_u32_stencils`
// in `vortex-cuda/build.rs`. Copy-and-Patch unpack stencil for u32
// element width with `BW = 0` baked in as a compile-time constant.

#include "bit_unpack_32_lanes.cuh"
#include "fastlanes_common.cuh"
#include <stdint.h>

extern "C" __device__ void
cp_unpack(const uint32_t *__restrict in, uint32_t *__restrict enc_buf) {
    _bit_unpack_32_lane<0>(in, enc_buf, /*reference=*/0u, /*lane=*/threadIdx.x);
}
