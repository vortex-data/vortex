// AUTO-GENERATED. Do not edit by hand!
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "fastlanes_common.cuh"

template <int BW>
__device__ void _bit_unpack_8_lane(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t reference, unsigned int lane);

template <>
__device__ void _bit_unpack_8_lane<0>(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t reference, unsigned int lane) {
    #pragma unroll
    for (int row = 0; row < 8; row++) {
        out[INDEX(row, lane)] = reference;
    }
}

template <>
__device__ void _bit_unpack_8_lane<1>(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t reference, unsigned int lane) {
    unsigned int LANE_COUNT = 128;
    uint8_t src;
    uint8_t tmp;
    src = in[lane];
    tmp = (src >> 0) & MASK(uint8_t, 1);
    out[INDEX(0, lane)] = tmp + reference;
    tmp = (src >> 1) & MASK(uint8_t, 1);
    out[INDEX(1, lane)] = tmp + reference;
    tmp = (src >> 2) & MASK(uint8_t, 1);
    out[INDEX(2, lane)] = tmp + reference;
    tmp = (src >> 3) & MASK(uint8_t, 1);
    out[INDEX(3, lane)] = tmp + reference;
    tmp = (src >> 4) & MASK(uint8_t, 1);
    out[INDEX(4, lane)] = tmp + reference;
    tmp = (src >> 5) & MASK(uint8_t, 1);
    out[INDEX(5, lane)] = tmp + reference;
    tmp = (src >> 6) & MASK(uint8_t, 1);
    out[INDEX(6, lane)] = tmp + reference;
    tmp = (src >> 7) & MASK(uint8_t, 1);
    out[INDEX(7, lane)] = tmp + reference;
}

template <>
__device__ void _bit_unpack_8_lane<2>(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t reference, unsigned int lane) {
    unsigned int LANE_COUNT = 128;
    uint8_t src;
    uint8_t tmp;
    src = in[lane];
    tmp = (src >> 0) & MASK(uint8_t, 2);
    out[INDEX(0, lane)] = tmp + reference;
    tmp = (src >> 2) & MASK(uint8_t, 2);
    out[INDEX(1, lane)] = tmp + reference;
    tmp = (src >> 4) & MASK(uint8_t, 2);
    out[INDEX(2, lane)] = tmp + reference;
    tmp = (src >> 6) & MASK(uint8_t, 2);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint8_t, 0)) << 2;
    out[INDEX(3, lane)] = tmp + reference;
    tmp = (src >> 0) & MASK(uint8_t, 2);
    out[INDEX(4, lane)] = tmp + reference;
    tmp = (src >> 2) & MASK(uint8_t, 2);
    out[INDEX(5, lane)] = tmp + reference;
    tmp = (src >> 4) & MASK(uint8_t, 2);
    out[INDEX(6, lane)] = tmp + reference;
    tmp = (src >> 6) & MASK(uint8_t, 2);
    out[INDEX(7, lane)] = tmp + reference;
}

template <>
__device__ void _bit_unpack_8_lane<3>(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t reference, unsigned int lane) {
    unsigned int LANE_COUNT = 128;
    uint8_t src;
    uint8_t tmp;
    src = in[lane];
    tmp = (src >> 0) & MASK(uint8_t, 3);
    out[INDEX(0, lane)] = tmp + reference;
    tmp = (src >> 3) & MASK(uint8_t, 3);
    out[INDEX(1, lane)] = tmp + reference;
    tmp = (src >> 6) & MASK(uint8_t, 2);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint8_t, 1)) << 2;
    out[INDEX(2, lane)] = tmp + reference;
    tmp = (src >> 1) & MASK(uint8_t, 3);
    out[INDEX(3, lane)] = tmp + reference;
    tmp = (src >> 4) & MASK(uint8_t, 3);
    out[INDEX(4, lane)] = tmp + reference;
    tmp = (src >> 7) & MASK(uint8_t, 1);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint8_t, 2)) << 1;
    out[INDEX(5, lane)] = tmp + reference;
    tmp = (src >> 2) & MASK(uint8_t, 3);
    out[INDEX(6, lane)] = tmp + reference;
    tmp = (src >> 5) & MASK(uint8_t, 3);
    out[INDEX(7, lane)] = tmp + reference;
}

template <>
__device__ void _bit_unpack_8_lane<4>(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t reference, unsigned int lane) {
    unsigned int LANE_COUNT = 128;
    uint8_t src;
    uint8_t tmp;
    src = in[lane];
    tmp = (src >> 0) & MASK(uint8_t, 4);
    out[INDEX(0, lane)] = tmp + reference;
    tmp = (src >> 4) & MASK(uint8_t, 4);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint8_t, 0)) << 4;
    out[INDEX(1, lane)] = tmp + reference;
    tmp = (src >> 0) & MASK(uint8_t, 4);
    out[INDEX(2, lane)] = tmp + reference;
    tmp = (src >> 4) & MASK(uint8_t, 4);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint8_t, 0)) << 4;
    out[INDEX(3, lane)] = tmp + reference;
    tmp = (src >> 0) & MASK(uint8_t, 4);
    out[INDEX(4, lane)] = tmp + reference;
    tmp = (src >> 4) & MASK(uint8_t, 4);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint8_t, 0)) << 4;
    out[INDEX(5, lane)] = tmp + reference;
    tmp = (src >> 0) & MASK(uint8_t, 4);
    out[INDEX(6, lane)] = tmp + reference;
    tmp = (src >> 4) & MASK(uint8_t, 4);
    out[INDEX(7, lane)] = tmp + reference;
}

template <>
__device__ void _bit_unpack_8_lane<5>(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t reference, unsigned int lane) {
    unsigned int LANE_COUNT = 128;
    uint8_t src;
    uint8_t tmp;
    src = in[lane];
    tmp = (src >> 0) & MASK(uint8_t, 5);
    out[INDEX(0, lane)] = tmp + reference;
    tmp = (src >> 5) & MASK(uint8_t, 3);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint8_t, 2)) << 3;
    out[INDEX(1, lane)] = tmp + reference;
    tmp = (src >> 2) & MASK(uint8_t, 5);
    out[INDEX(2, lane)] = tmp + reference;
    tmp = (src >> 7) & MASK(uint8_t, 1);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint8_t, 4)) << 1;
    out[INDEX(3, lane)] = tmp + reference;
    tmp = (src >> 4) & MASK(uint8_t, 4);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint8_t, 1)) << 4;
    out[INDEX(4, lane)] = tmp + reference;
    tmp = (src >> 1) & MASK(uint8_t, 5);
    out[INDEX(5, lane)] = tmp + reference;
    tmp = (src >> 6) & MASK(uint8_t, 2);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint8_t, 3)) << 2;
    out[INDEX(6, lane)] = tmp + reference;
    tmp = (src >> 3) & MASK(uint8_t, 5);
    out[INDEX(7, lane)] = tmp + reference;
}

template <>
__device__ void _bit_unpack_8_lane<6>(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t reference, unsigned int lane) {
    unsigned int LANE_COUNT = 128;
    uint8_t src;
    uint8_t tmp;
    src = in[lane];
    tmp = (src >> 0) & MASK(uint8_t, 6);
    out[INDEX(0, lane)] = tmp + reference;
    tmp = (src >> 6) & MASK(uint8_t, 2);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint8_t, 4)) << 2;
    out[INDEX(1, lane)] = tmp + reference;
    tmp = (src >> 4) & MASK(uint8_t, 4);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint8_t, 2)) << 4;
    out[INDEX(2, lane)] = tmp + reference;
    tmp = (src >> 2) & MASK(uint8_t, 6);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint8_t, 0)) << 6;
    out[INDEX(3, lane)] = tmp + reference;
    tmp = (src >> 0) & MASK(uint8_t, 6);
    out[INDEX(4, lane)] = tmp + reference;
    tmp = (src >> 6) & MASK(uint8_t, 2);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint8_t, 4)) << 2;
    out[INDEX(5, lane)] = tmp + reference;
    tmp = (src >> 4) & MASK(uint8_t, 4);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint8_t, 2)) << 4;
    out[INDEX(6, lane)] = tmp + reference;
    tmp = (src >> 2) & MASK(uint8_t, 6);
    out[INDEX(7, lane)] = tmp + reference;
}

template <>
__device__ void _bit_unpack_8_lane<7>(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t reference, unsigned int lane) {
    unsigned int LANE_COUNT = 128;
    uint8_t src;
    uint8_t tmp;
    src = in[lane];
    tmp = (src >> 0) & MASK(uint8_t, 7);
    out[INDEX(0, lane)] = tmp + reference;
    tmp = (src >> 7) & MASK(uint8_t, 1);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint8_t, 6)) << 1;
    out[INDEX(1, lane)] = tmp + reference;
    tmp = (src >> 6) & MASK(uint8_t, 2);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint8_t, 5)) << 2;
    out[INDEX(2, lane)] = tmp + reference;
    tmp = (src >> 5) & MASK(uint8_t, 3);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint8_t, 4)) << 3;
    out[INDEX(3, lane)] = tmp + reference;
    tmp = (src >> 4) & MASK(uint8_t, 4);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint8_t, 3)) << 4;
    out[INDEX(4, lane)] = tmp + reference;
    tmp = (src >> 3) & MASK(uint8_t, 5);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint8_t, 2)) << 5;
    out[INDEX(5, lane)] = tmp + reference;
    tmp = (src >> 2) & MASK(uint8_t, 6);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint8_t, 1)) << 6;
    out[INDEX(6, lane)] = tmp + reference;
    tmp = (src >> 1) & MASK(uint8_t, 7);
    out[INDEX(7, lane)] = tmp + reference;
}

template <>
__device__ void _bit_unpack_8_lane<8>(const uint8_t *__restrict in, uint8_t *__restrict out, uint8_t reference, unsigned int lane) {
    unsigned int LANE_COUNT = 128;
    #pragma unroll
    for (int row = 0; row < 8; row++) {
        out[INDEX(row, lane)] = in[LANE_COUNT * row + lane] + reference;
    }
}

/// Runtime dispatch to the optimized lane decoder for the given bit width.
__device__ __noinline__ void bit_unpack_8_lane(
    const uint8_t *__restrict in,
    uint8_t *__restrict out,
    uint8_t reference,
    unsigned int lane,
    uint32_t bit_width
) {
    switch (bit_width) {
        case 0: _bit_unpack_8_lane<0>(in, out, reference, lane); break;
        case 1: _bit_unpack_8_lane<1>(in, out, reference, lane); break;
        case 2: _bit_unpack_8_lane<2>(in, out, reference, lane); break;
        case 3: _bit_unpack_8_lane<3>(in, out, reference, lane); break;
        case 4: _bit_unpack_8_lane<4>(in, out, reference, lane); break;
        case 5: _bit_unpack_8_lane<5>(in, out, reference, lane); break;
        case 6: _bit_unpack_8_lane<6>(in, out, reference, lane); break;
        case 7: _bit_unpack_8_lane<7>(in, out, reference, lane); break;
        case 8: _bit_unpack_8_lane<8>(in, out, reference, lane); break;
    }
}

