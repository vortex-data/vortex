// AUTO-GENERATED. Do not edit by hand!
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "fastlanes_common.cuh"

__device__ void _bit_unpack_8_0bw_lane(const uint8_t *__restrict in,
                                       uint8_t *__restrict out,
                                       uint8_t reference,
                                       unsigned int lane) {
    unsigned int LANE_COUNT = 128;

    out[INDEX(0, lane)] = reference;
    out[INDEX(1, lane)] = reference;
    out[INDEX(2, lane)] = reference;
    out[INDEX(3, lane)] = reference;
    out[INDEX(4, lane)] = reference;
    out[INDEX(5, lane)] = reference;
    out[INDEX(6, lane)] = reference;
    out[INDEX(7, lane)] = reference;
}

__device__ void _bit_unpack_8_1bw_lane(const uint8_t *__restrict in,
                                       uint8_t *__restrict out,
                                       uint8_t reference,
                                       unsigned int lane) {
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

__device__ void _bit_unpack_8_2bw_lane(const uint8_t *__restrict in,
                                       uint8_t *__restrict out,
                                       uint8_t reference,
                                       unsigned int lane) {
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

__device__ void _bit_unpack_8_3bw_lane(const uint8_t *__restrict in,
                                       uint8_t *__restrict out,
                                       uint8_t reference,
                                       unsigned int lane) {
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

__device__ void _bit_unpack_8_4bw_lane(const uint8_t *__restrict in,
                                       uint8_t *__restrict out,
                                       uint8_t reference,
                                       unsigned int lane) {
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

__device__ void _bit_unpack_8_5bw_lane(const uint8_t *__restrict in,
                                       uint8_t *__restrict out,
                                       uint8_t reference,
                                       unsigned int lane) {
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

__device__ void _bit_unpack_8_6bw_lane(const uint8_t *__restrict in,
                                       uint8_t *__restrict out,
                                       uint8_t reference,
                                       unsigned int lane) {
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

__device__ void _bit_unpack_8_7bw_lane(const uint8_t *__restrict in,
                                       uint8_t *__restrict out,
                                       uint8_t reference,
                                       unsigned int lane) {
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

__device__ void _bit_unpack_8_8bw_lane(const uint8_t *__restrict in,
                                       uint8_t *__restrict out,
                                       uint8_t reference,
                                       unsigned int lane) {
    unsigned int LANE_COUNT = 128;

    out[INDEX(0, lane)] = in[LANE_COUNT * 0 + lane] + reference;
    out[INDEX(1, lane)] = in[LANE_COUNT * 1 + lane] + reference;
    out[INDEX(2, lane)] = in[LANE_COUNT * 2 + lane] + reference;
    out[INDEX(3, lane)] = in[LANE_COUNT * 3 + lane] + reference;
    out[INDEX(4, lane)] = in[LANE_COUNT * 4 + lane] + reference;
    out[INDEX(5, lane)] = in[LANE_COUNT * 5 + lane] + reference;
    out[INDEX(6, lane)] = in[LANE_COUNT * 6 + lane] + reference;
    out[INDEX(7, lane)] = in[LANE_COUNT * 7 + lane] + reference;
}

/// Runtime dispatch to the optimized lane decoder for the given bit width.
__device__ inline void bit_unpack_8_lane(const uint8_t *__restrict in,
                                         uint8_t *__restrict out,
                                         uint8_t reference,
                                         unsigned int lane,
                                         uint32_t bit_width) {
    switch (bit_width) {
    case 0:
        _bit_unpack_8_0bw_lane(in, out, reference, lane);
        break;
    case 1:
        _bit_unpack_8_1bw_lane(in, out, reference, lane);
        break;
    case 2:
        _bit_unpack_8_2bw_lane(in, out, reference, lane);
        break;
    case 3:
        _bit_unpack_8_3bw_lane(in, out, reference, lane);
        break;
    case 4:
        _bit_unpack_8_4bw_lane(in, out, reference, lane);
        break;
    case 5:
        _bit_unpack_8_5bw_lane(in, out, reference, lane);
        break;
    case 6:
        _bit_unpack_8_6bw_lane(in, out, reference, lane);
        break;
    case 7:
        _bit_unpack_8_7bw_lane(in, out, reference, lane);
        break;
    case 8:
        _bit_unpack_8_8bw_lane(in, out, reference, lane);
        break;
    }
}

__device__ void _bit_unpack_8_0bw_32t(const uint8_t *__restrict in,
                                      uint8_t *__restrict out,
                                      uint8_t reference,
                                      int thread_idx) {
    __shared__ uint8_t shared_out[1024];
    _bit_unpack_8_0bw_lane(in, shared_out, reference, thread_idx * 4 + 0);
    _bit_unpack_8_0bw_lane(in, shared_out, reference, thread_idx * 4 + 1);
    _bit_unpack_8_0bw_lane(in, shared_out, reference, thread_idx * 4 + 2);
    _bit_unpack_8_0bw_lane(in, shared_out, reference, thread_idx * 4 + 3);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void
bit_unpack_8_0bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 0 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_0bw_32t(in, out, reference, thread_idx);
}

__device__ void _bit_unpack_8_1bw_32t(const uint8_t *__restrict in,
                                      uint8_t *__restrict out,
                                      uint8_t reference,
                                      int thread_idx) {
    __shared__ uint8_t shared_out[1024];
    _bit_unpack_8_1bw_lane(in, shared_out, reference, thread_idx * 4 + 0);
    _bit_unpack_8_1bw_lane(in, shared_out, reference, thread_idx * 4 + 1);
    _bit_unpack_8_1bw_lane(in, shared_out, reference, thread_idx * 4 + 2);
    _bit_unpack_8_1bw_lane(in, shared_out, reference, thread_idx * 4 + 3);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void
bit_unpack_8_1bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 1 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_1bw_32t(in, out, reference, thread_idx);
}

__device__ void _bit_unpack_8_2bw_32t(const uint8_t *__restrict in,
                                      uint8_t *__restrict out,
                                      uint8_t reference,
                                      int thread_idx) {
    __shared__ uint8_t shared_out[1024];
    _bit_unpack_8_2bw_lane(in, shared_out, reference, thread_idx * 4 + 0);
    _bit_unpack_8_2bw_lane(in, shared_out, reference, thread_idx * 4 + 1);
    _bit_unpack_8_2bw_lane(in, shared_out, reference, thread_idx * 4 + 2);
    _bit_unpack_8_2bw_lane(in, shared_out, reference, thread_idx * 4 + 3);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void
bit_unpack_8_2bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 2 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_2bw_32t(in, out, reference, thread_idx);
}

__device__ void _bit_unpack_8_3bw_32t(const uint8_t *__restrict in,
                                      uint8_t *__restrict out,
                                      uint8_t reference,
                                      int thread_idx) {
    __shared__ uint8_t shared_out[1024];
    _bit_unpack_8_3bw_lane(in, shared_out, reference, thread_idx * 4 + 0);
    _bit_unpack_8_3bw_lane(in, shared_out, reference, thread_idx * 4 + 1);
    _bit_unpack_8_3bw_lane(in, shared_out, reference, thread_idx * 4 + 2);
    _bit_unpack_8_3bw_lane(in, shared_out, reference, thread_idx * 4 + 3);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void
bit_unpack_8_3bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 3 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_3bw_32t(in, out, reference, thread_idx);
}

__device__ void _bit_unpack_8_4bw_32t(const uint8_t *__restrict in,
                                      uint8_t *__restrict out,
                                      uint8_t reference,
                                      int thread_idx) {
    __shared__ uint8_t shared_out[1024];
    _bit_unpack_8_4bw_lane(in, shared_out, reference, thread_idx * 4 + 0);
    _bit_unpack_8_4bw_lane(in, shared_out, reference, thread_idx * 4 + 1);
    _bit_unpack_8_4bw_lane(in, shared_out, reference, thread_idx * 4 + 2);
    _bit_unpack_8_4bw_lane(in, shared_out, reference, thread_idx * 4 + 3);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void
bit_unpack_8_4bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 4 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_4bw_32t(in, out, reference, thread_idx);
}

__device__ void _bit_unpack_8_5bw_32t(const uint8_t *__restrict in,
                                      uint8_t *__restrict out,
                                      uint8_t reference,
                                      int thread_idx) {
    __shared__ uint8_t shared_out[1024];
    _bit_unpack_8_5bw_lane(in, shared_out, reference, thread_idx * 4 + 0);
    _bit_unpack_8_5bw_lane(in, shared_out, reference, thread_idx * 4 + 1);
    _bit_unpack_8_5bw_lane(in, shared_out, reference, thread_idx * 4 + 2);
    _bit_unpack_8_5bw_lane(in, shared_out, reference, thread_idx * 4 + 3);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void
bit_unpack_8_5bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 5 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_5bw_32t(in, out, reference, thread_idx);
}

__device__ void _bit_unpack_8_6bw_32t(const uint8_t *__restrict in,
                                      uint8_t *__restrict out,
                                      uint8_t reference,
                                      int thread_idx) {
    __shared__ uint8_t shared_out[1024];
    _bit_unpack_8_6bw_lane(in, shared_out, reference, thread_idx * 4 + 0);
    _bit_unpack_8_6bw_lane(in, shared_out, reference, thread_idx * 4 + 1);
    _bit_unpack_8_6bw_lane(in, shared_out, reference, thread_idx * 4 + 2);
    _bit_unpack_8_6bw_lane(in, shared_out, reference, thread_idx * 4 + 3);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void
bit_unpack_8_6bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 6 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_6bw_32t(in, out, reference, thread_idx);
}

__device__ void _bit_unpack_8_7bw_32t(const uint8_t *__restrict in,
                                      uint8_t *__restrict out,
                                      uint8_t reference,
                                      int thread_idx) {
    __shared__ uint8_t shared_out[1024];
    _bit_unpack_8_7bw_lane(in, shared_out, reference, thread_idx * 4 + 0);
    _bit_unpack_8_7bw_lane(in, shared_out, reference, thread_idx * 4 + 1);
    _bit_unpack_8_7bw_lane(in, shared_out, reference, thread_idx * 4 + 2);
    _bit_unpack_8_7bw_lane(in, shared_out, reference, thread_idx * 4 + 3);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void
bit_unpack_8_7bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 7 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_7bw_32t(in, out, reference, thread_idx);
}

__device__ void _bit_unpack_8_8bw_32t(const uint8_t *__restrict in,
                                      uint8_t *__restrict out,
                                      uint8_t reference,
                                      int thread_idx) {
    __shared__ uint8_t shared_out[1024];
    _bit_unpack_8_8bw_lane(in, shared_out, reference, thread_idx * 4 + 0);
    _bit_unpack_8_8bw_lane(in, shared_out, reference, thread_idx * 4 + 1);
    _bit_unpack_8_8bw_lane(in, shared_out, reference, thread_idx * 4 + 2);
    _bit_unpack_8_8bw_lane(in, shared_out, reference, thread_idx * 4 + 3);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void
bit_unpack_8_8bw_32t(const uint8_t *__restrict full_in, uint8_t *__restrict full_out, uint8_t reference) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 8 / sizeof(uint8_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_8_8bw_32t(in, out, reference, thread_idx);
}
