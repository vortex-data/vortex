// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Bit-unpack stencil for u32 element width. Operates on one 1024-element
// FastLanes chunk cooperatively across a 32-thread warp; each thread handles
// one FastLanes lane. Writes the unpacked values into `enc_buf` in linear
// (original-element) order — the `INDEX(row, lane)` macro inside the lane
// decoder maps the packed bit-iteration coordinate to its original position.
//
// The runtime `bit_width` parameter dispatches via the switch in
// `bit_unpack_32_lane`, which inlines the right specialization. A pure
// Copy-and-Patch implementation would emit one PTX per bit width and bake
// the constant in; this prototype shows the variant-selection mechanism
// for the ALP and post-op stencils first.

#include "bit_unpack_32_lanes.cuh"
#include "fastlanes_common.cuh"
#include <stdint.h>

// `bit_unpack_32_lane` is defined in dynamic_dispatch.cu — but we can't
// depend on that link unit. We re-declare a thin device dispatcher inline
// here, calling the per-BW lane decoders that live in the included header.
__device__ __noinline__ static void
cp_unpack_dispatch(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane, uint32_t bw) {
    switch (bw) {
    case 0: _bit_unpack_32_lane<0>(in, out, 0u, lane); break;
    case 1: _bit_unpack_32_lane<1>(in, out, 0u, lane); break;
    case 2: _bit_unpack_32_lane<2>(in, out, 0u, lane); break;
    case 3: _bit_unpack_32_lane<3>(in, out, 0u, lane); break;
    case 4: _bit_unpack_32_lane<4>(in, out, 0u, lane); break;
    case 5: _bit_unpack_32_lane<5>(in, out, 0u, lane); break;
    case 6: _bit_unpack_32_lane<6>(in, out, 0u, lane); break;
    case 7: _bit_unpack_32_lane<7>(in, out, 0u, lane); break;
    case 8: _bit_unpack_32_lane<8>(in, out, 0u, lane); break;
    case 9: _bit_unpack_32_lane<9>(in, out, 0u, lane); break;
    case 10: _bit_unpack_32_lane<10>(in, out, 0u, lane); break;
    case 11: _bit_unpack_32_lane<11>(in, out, 0u, lane); break;
    case 12: _bit_unpack_32_lane<12>(in, out, 0u, lane); break;
    case 13: _bit_unpack_32_lane<13>(in, out, 0u, lane); break;
    case 14: _bit_unpack_32_lane<14>(in, out, 0u, lane); break;
    case 15: _bit_unpack_32_lane<15>(in, out, 0u, lane); break;
    case 16: _bit_unpack_32_lane<16>(in, out, 0u, lane); break;
    case 17: _bit_unpack_32_lane<17>(in, out, 0u, lane); break;
    case 18: _bit_unpack_32_lane<18>(in, out, 0u, lane); break;
    case 19: _bit_unpack_32_lane<19>(in, out, 0u, lane); break;
    case 20: _bit_unpack_32_lane<20>(in, out, 0u, lane); break;
    case 21: _bit_unpack_32_lane<21>(in, out, 0u, lane); break;
    case 22: _bit_unpack_32_lane<22>(in, out, 0u, lane); break;
    case 23: _bit_unpack_32_lane<23>(in, out, 0u, lane); break;
    case 24: _bit_unpack_32_lane<24>(in, out, 0u, lane); break;
    case 25: _bit_unpack_32_lane<25>(in, out, 0u, lane); break;
    case 26: _bit_unpack_32_lane<26>(in, out, 0u, lane); break;
    case 27: _bit_unpack_32_lane<27>(in, out, 0u, lane); break;
    case 28: _bit_unpack_32_lane<28>(in, out, 0u, lane); break;
    case 29: _bit_unpack_32_lane<29>(in, out, 0u, lane); break;
    case 30: _bit_unpack_32_lane<30>(in, out, 0u, lane); break;
    case 31: _bit_unpack_32_lane<31>(in, out, 0u, lane); break;
    case 32: _bit_unpack_32_lane<32>(in, out, 0u, lane); break;
    }
}

extern "C" __device__ void
cp_unpack(const uint32_t *__restrict in, uint32_t *__restrict enc_buf, uint32_t bit_width) {
    cp_unpack_dispatch(in, enc_buf, threadIdx.x, bit_width);
}
