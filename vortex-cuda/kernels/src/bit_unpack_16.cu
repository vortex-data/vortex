// AUTO-GENERATED. Do not edit by hand!
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "fastlanes_common.cuh"

__device__ void _bit_unpack_16_0bw_lane(const uint16_t *__restrict in, uint16_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 64;
    uint16_t zero = 0ULL;
    
    out[INDEX(0, lane)] = zero;
    out[INDEX(1, lane)] = zero;
    out[INDEX(2, lane)] = zero;
    out[INDEX(3, lane)] = zero;
    out[INDEX(4, lane)] = zero;
    out[INDEX(5, lane)] = zero;
    out[INDEX(6, lane)] = zero;
    out[INDEX(7, lane)] = zero;
    out[INDEX(8, lane)] = zero;
    out[INDEX(9, lane)] = zero;
    out[INDEX(10, lane)] = zero;
    out[INDEX(11, lane)] = zero;
    out[INDEX(12, lane)] = zero;
    out[INDEX(13, lane)] = zero;
    out[INDEX(14, lane)] = zero;
    out[INDEX(15, lane)] = zero;
}

__device__ void _bit_unpack_16_0bw_32t(const uint16_t *__restrict in, uint16_t *__restrict out, int thread_idx) {
    _bit_unpack_16_0bw_lane(in, out, thread_idx * 2 + 0);
    _bit_unpack_16_0bw_lane(in, out, thread_idx * 2 + 1);
}

extern "C" __global__ void bit_unpack_16_0bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 0 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_0bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_16_1bw_lane(const uint16_t *__restrict in, uint16_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 64;
    uint16_t src;
    uint16_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint16_t, 1);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint16_t, 1);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint16_t, 1);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint16_t, 1);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 1);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint16_t, 1);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint16_t, 1);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint16_t, 1);
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 1);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint16_t, 1);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint16_t, 1);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint16_t, 1);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 1);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint16_t, 1);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint16_t, 1);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint16_t, 1);
    out[INDEX(15, lane)] = tmp;
}

__device__ void _bit_unpack_16_1bw_32t(const uint16_t *__restrict in, uint16_t *__restrict out, int thread_idx) {
    _bit_unpack_16_1bw_lane(in, out, thread_idx * 2 + 0);
    _bit_unpack_16_1bw_lane(in, out, thread_idx * 2 + 1);
}

extern "C" __global__ void bit_unpack_16_1bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 1 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_1bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_16_2bw_lane(const uint16_t *__restrict in, uint16_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 64;
    uint16_t src;
    uint16_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint16_t, 2);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint16_t, 2);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 2);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint16_t, 2);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 2);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint16_t, 2);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 2);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint16_t, 2);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint16_t, 0)) << 2;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint16_t, 2);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint16_t, 2);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 2);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint16_t, 2);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 2);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint16_t, 2);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 2);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint16_t, 2);
    out[INDEX(15, lane)] = tmp;
}

__device__ void _bit_unpack_16_2bw_32t(const uint16_t *__restrict in, uint16_t *__restrict out, int thread_idx) {
    _bit_unpack_16_2bw_lane(in, out, thread_idx * 2 + 0);
    _bit_unpack_16_2bw_lane(in, out, thread_idx * 2 + 1);
}

extern "C" __global__ void bit_unpack_16_2bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 2 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_2bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_16_3bw_lane(const uint16_t *__restrict in, uint16_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 64;
    uint16_t src;
    uint16_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint16_t, 3);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint16_t, 3);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint16_t, 3);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint16_t, 3);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 3);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint16_t, 1);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint16_t, 2)) << 1;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint16_t, 3);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint16_t, 3);
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 3);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint16_t, 3);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint16_t, 2);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint16_t, 1)) << 2;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint16_t, 3);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 3);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint16_t, 3);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint16_t, 3);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint16_t, 3);
    out[INDEX(15, lane)] = tmp;
}

__device__ void _bit_unpack_16_3bw_32t(const uint16_t *__restrict in, uint16_t *__restrict out, int thread_idx) {
    _bit_unpack_16_3bw_lane(in, out, thread_idx * 2 + 0);
    _bit_unpack_16_3bw_lane(in, out, thread_idx * 2 + 1);
}

extern "C" __global__ void bit_unpack_16_3bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 3 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_3bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_16_4bw_lane(const uint16_t *__restrict in, uint16_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 64;
    uint16_t src;
    uint16_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint16_t, 4);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 4);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 4);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint16_t, 0)) << 4;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint16_t, 4);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 4);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 4);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint16_t, 0)) << 4;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint16_t, 4);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 4);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 4);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint16_t, 0)) << 4;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint16_t, 4);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 4);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 4);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    out[INDEX(15, lane)] = tmp;
}

__device__ void _bit_unpack_16_4bw_32t(const uint16_t *__restrict in, uint16_t *__restrict out, int thread_idx) {
    _bit_unpack_16_4bw_lane(in, out, thread_idx * 2 + 0);
    _bit_unpack_16_4bw_lane(in, out, thread_idx * 2 + 1);
}

extern "C" __global__ void bit_unpack_16_4bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 4 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_4bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_16_5bw_lane(const uint16_t *__restrict in, uint16_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 64;
    uint16_t src;
    uint16_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint16_t, 5);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint16_t, 5);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint16_t, 5);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint16_t, 1);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint16_t, 4)) << 1;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 5);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint16_t, 5);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint16_t, 2);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint16_t, 3)) << 2;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint16_t, 5);
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 5);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint16_t, 3);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint16_t, 2)) << 3;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint16_t, 5);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint16_t, 5);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint16_t, 1)) << 4;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint16_t, 5);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint16_t, 5);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint16_t, 5);
    out[INDEX(15, lane)] = tmp;
}

__device__ void _bit_unpack_16_5bw_32t(const uint16_t *__restrict in, uint16_t *__restrict out, int thread_idx) {
    _bit_unpack_16_5bw_lane(in, out, thread_idx * 2 + 0);
    _bit_unpack_16_5bw_lane(in, out, thread_idx * 2 + 1);
}

extern "C" __global__ void bit_unpack_16_5bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 5 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_5bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_16_6bw_lane(const uint16_t *__restrict in, uint16_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 64;
    uint16_t src;
    uint16_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint16_t, 6);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint16_t, 6);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint16_t, 2)) << 4;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint16_t, 6);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 6);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint16_t, 2);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint16_t, 4)) << 2;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 6);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint16_t, 6);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint16_t, 0)) << 6;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint16_t, 6);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint16_t, 6);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint16_t, 2)) << 4;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint16_t, 6);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 6);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint16_t, 2);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint16_t, 4)) << 2;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 6);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint16_t, 6);
    out[INDEX(15, lane)] = tmp;
}

__device__ void _bit_unpack_16_6bw_32t(const uint16_t *__restrict in, uint16_t *__restrict out, int thread_idx) {
    _bit_unpack_16_6bw_lane(in, out, thread_idx * 2 + 0);
    _bit_unpack_16_6bw_lane(in, out, thread_idx * 2 + 1);
}

extern "C" __global__ void bit_unpack_16_6bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 6 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_6bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_16_7bw_lane(const uint16_t *__restrict in, uint16_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 64;
    uint16_t src;
    uint16_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint16_t, 7);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint16_t, 7);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint16_t, 2);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint16_t, 5)) << 2;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint16_t, 7);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint16_t, 3)) << 4;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint16_t, 7);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint16_t, 6);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint16_t, 1)) << 6;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint16_t, 7);
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 7);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint16_t, 1);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint16_t, 6)) << 1;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint16_t, 7);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint16_t, 3);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint16_t, 4)) << 3;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 7);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint16_t, 5);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint16_t, 2)) << 5;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint16_t, 7);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint16_t, 7);
    out[INDEX(15, lane)] = tmp;
}

__device__ void _bit_unpack_16_7bw_32t(const uint16_t *__restrict in, uint16_t *__restrict out, int thread_idx) {
    _bit_unpack_16_7bw_lane(in, out, thread_idx * 2 + 0);
    _bit_unpack_16_7bw_lane(in, out, thread_idx * 2 + 1);
}

extern "C" __global__ void bit_unpack_16_7bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 7 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_7bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_16_8bw_lane(const uint16_t *__restrict in, uint16_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 64;
    uint16_t src;
    uint16_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint16_t, 8);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint16_t, 0)) << 8;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint16_t, 8);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint16_t, 0)) << 8;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint16_t, 8);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint16_t, 0)) << 8;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint16_t, 8);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint16_t, 0)) << 8;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint16_t, 8);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint16_t, 0)) << 8;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint16_t, 8);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint16_t, 0)) << 8;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint16_t, 8);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint16_t, 0)) << 8;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint16_t, 8);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    out[INDEX(15, lane)] = tmp;
}

__device__ void _bit_unpack_16_8bw_32t(const uint16_t *__restrict in, uint16_t *__restrict out, int thread_idx) {
    _bit_unpack_16_8bw_lane(in, out, thread_idx * 2 + 0);
    _bit_unpack_16_8bw_lane(in, out, thread_idx * 2 + 1);
}

extern "C" __global__ void bit_unpack_16_8bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 8 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_8bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_16_9bw_lane(const uint16_t *__restrict in, uint16_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 64;
    uint16_t src;
    uint16_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint16_t, 9);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint16_t, 7);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint16_t, 2)) << 7;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint16_t, 9);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint16_t, 5);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint16_t, 4)) << 5;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 9);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint16_t, 3);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint16_t, 6)) << 3;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint16_t, 9);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint16_t, 1);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint16_t, 8)) << 1;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint16_t, 1)) << 8;
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint16_t, 9);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint16_t, 6);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint16_t, 3)) << 6;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint16_t, 9);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint16_t, 5)) << 4;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint16_t, 9);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint16_t, 2);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint16_t, 7)) << 2;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint16_t, 9);
    out[INDEX(15, lane)] = tmp;
}

__device__ void _bit_unpack_16_9bw_32t(const uint16_t *__restrict in, uint16_t *__restrict out, int thread_idx) {
    _bit_unpack_16_9bw_lane(in, out, thread_idx * 2 + 0);
    _bit_unpack_16_9bw_lane(in, out, thread_idx * 2 + 1);
}

extern "C" __global__ void bit_unpack_16_9bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 9 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_9bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_16_10bw_lane(const uint16_t *__restrict in, uint16_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 64;
    uint16_t src;
    uint16_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint16_t, 10);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint16_t, 6);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint16_t, 4)) << 6;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 10);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint16_t, 2);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint16_t, 8)) << 2;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint16_t, 2)) << 8;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint16_t, 10);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint16_t, 6)) << 4;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint16_t, 10);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint16_t, 0)) << 10;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint16_t, 10);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint16_t, 6);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint16_t, 4)) << 6;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 10);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint16_t, 2);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint16_t, 8)) << 2;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint16_t, 2)) << 8;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint16_t, 10);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint16_t, 6)) << 4;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint16_t, 10);
    out[INDEX(15, lane)] = tmp;
}

__device__ void _bit_unpack_16_10bw_32t(const uint16_t *__restrict in, uint16_t *__restrict out, int thread_idx) {
    _bit_unpack_16_10bw_lane(in, out, thread_idx * 2 + 0);
    _bit_unpack_16_10bw_lane(in, out, thread_idx * 2 + 1);
}

extern "C" __global__ void bit_unpack_16_10bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 10 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_10bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_16_11bw_lane(const uint16_t *__restrict in, uint16_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 64;
    uint16_t src;
    uint16_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint16_t, 11);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint16_t, 5);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint16_t, 6)) << 5;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint16_t, 10);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint16_t, 1)) << 10;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint16_t, 11);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint16_t, 7)) << 4;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint16_t, 9);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint16_t, 2)) << 9;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint16_t, 11);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint16_t, 3);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint16_t, 8)) << 3;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint16_t, 3)) << 8;
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint16_t, 11);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint16_t, 2);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint16_t, 9)) << 2;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint16_t, 7);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint16_t, 4)) << 7;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 11);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint16_t, 1);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint16_t, 10)) << 1;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint16_t, 6);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint16_t, 5)) << 6;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint16_t, 11);
    out[INDEX(15, lane)] = tmp;
}

__device__ void _bit_unpack_16_11bw_32t(const uint16_t *__restrict in, uint16_t *__restrict out, int thread_idx) {
    _bit_unpack_16_11bw_lane(in, out, thread_idx * 2 + 0);
    _bit_unpack_16_11bw_lane(in, out, thread_idx * 2 + 1);
}

extern "C" __global__ void bit_unpack_16_11bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 11 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_11bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_16_12bw_lane(const uint16_t *__restrict in, uint16_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 64;
    uint16_t src;
    uint16_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint16_t, 12);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint16_t, 8)) << 4;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint16_t, 4)) << 8;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 12);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint16_t, 0)) << 12;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint16_t, 12);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint16_t, 8)) << 4;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint16_t, 4)) << 8;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 12);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint16_t, 0)) << 12;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint16_t, 12);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint16_t, 8)) << 4;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint16_t, 4)) << 8;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 12);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint16_t, 0)) << 12;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint16_t, 12);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint16_t, 8)) << 4;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint16_t, 4)) << 8;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 12);
    out[INDEX(15, lane)] = tmp;
}

__device__ void _bit_unpack_16_12bw_32t(const uint16_t *__restrict in, uint16_t *__restrict out, int thread_idx) {
    _bit_unpack_16_12bw_lane(in, out, thread_idx * 2 + 0);
    _bit_unpack_16_12bw_lane(in, out, thread_idx * 2 + 1);
}

extern "C" __global__ void bit_unpack_16_12bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 12 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_12bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_16_13bw_lane(const uint16_t *__restrict in, uint16_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 64;
    uint16_t src;
    uint16_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint16_t, 13);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint16_t, 3);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint16_t, 10)) << 3;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint16_t, 6);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint16_t, 7)) << 6;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint16_t, 9);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint16_t, 4)) << 9;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 12);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint16_t, 1)) << 12;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint16_t, 13);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint16_t, 2);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint16_t, 11)) << 2;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint16_t, 5);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint16_t, 8)) << 5;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint16_t, 5)) << 8;
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint16_t, 11);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint16_t, 2)) << 11;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint16_t, 13);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint16_t, 1);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint16_t, 12)) << 1;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint16_t, 9)) << 4;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint16_t, 7);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint16_t, 6)) << 7;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint16_t, 10);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint16_t, 3)) << 10;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint16_t, 13);
    out[INDEX(15, lane)] = tmp;
}

__device__ void _bit_unpack_16_13bw_32t(const uint16_t *__restrict in, uint16_t *__restrict out, int thread_idx) {
    _bit_unpack_16_13bw_lane(in, out, thread_idx * 2 + 0);
    _bit_unpack_16_13bw_lane(in, out, thread_idx * 2 + 1);
}

extern "C" __global__ void bit_unpack_16_13bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 13 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_13bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_16_14bw_lane(const uint16_t *__restrict in, uint16_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 64;
    uint16_t src;
    uint16_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint16_t, 14);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint16_t, 2);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint16_t, 12)) << 2;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint16_t, 10)) << 4;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint16_t, 6);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint16_t, 8)) << 6;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint16_t, 6)) << 8;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint16_t, 10);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint16_t, 4)) << 10;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 12);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint16_t, 2)) << 12;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint16_t, 14);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint16_t, 0)) << 14;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint16_t, 14);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint16_t, 2);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint16_t, 12)) << 2;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint16_t, 10)) << 4;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint16_t, 6);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint16_t, 8)) << 6;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint16_t, 6)) << 8;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint16_t, 10);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint16_t, 4)) << 10;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 12);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint16_t, 2)) << 12;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint16_t, 14);
    out[INDEX(15, lane)] = tmp;
}

__device__ void _bit_unpack_16_14bw_32t(const uint16_t *__restrict in, uint16_t *__restrict out, int thread_idx) {
    _bit_unpack_16_14bw_lane(in, out, thread_idx * 2 + 0);
    _bit_unpack_16_14bw_lane(in, out, thread_idx * 2 + 1);
}

extern "C" __global__ void bit_unpack_16_14bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 14 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_14bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_16_15bw_lane(const uint16_t *__restrict in, uint16_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 64;
    uint16_t src;
    uint16_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint16_t, 15);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint16_t, 1);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint16_t, 14)) << 1;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint16_t, 2);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint16_t, 13)) << 2;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint16_t, 3);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint16_t, 12)) << 3;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint16_t, 4);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint16_t, 11)) << 4;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint16_t, 5);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint16_t, 10)) << 5;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint16_t, 6);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint16_t, 9)) << 6;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint16_t, 7);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint16_t, 8)) << 7;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint16_t, 8);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint16_t, 7)) << 8;
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint16_t, 9);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint16_t, 6)) << 9;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint16_t, 10);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint16_t, 5)) << 10;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint16_t, 11);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint16_t, 4)) << 11;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint16_t, 12);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint16_t, 3)) << 12;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint16_t, 13);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint16_t, 2)) << 13;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint16_t, 14);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint16_t, 1)) << 14;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint16_t, 15);
    out[INDEX(15, lane)] = tmp;
}

__device__ void _bit_unpack_16_15bw_32t(const uint16_t *__restrict in, uint16_t *__restrict out, int thread_idx) {
    _bit_unpack_16_15bw_lane(in, out, thread_idx * 2 + 0);
    _bit_unpack_16_15bw_lane(in, out, thread_idx * 2 + 1);
}

extern "C" __global__ void bit_unpack_16_15bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 15 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_15bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_16_16bw_lane(const uint16_t *__restrict in, uint16_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 64;
    
    out[INDEX(0, lane)] = in[LANE_COUNT * 0 + lane];
    out[INDEX(1, lane)] = in[LANE_COUNT * 1 + lane];
    out[INDEX(2, lane)] = in[LANE_COUNT * 2 + lane];
    out[INDEX(3, lane)] = in[LANE_COUNT * 3 + lane];
    out[INDEX(4, lane)] = in[LANE_COUNT * 4 + lane];
    out[INDEX(5, lane)] = in[LANE_COUNT * 5 + lane];
    out[INDEX(6, lane)] = in[LANE_COUNT * 6 + lane];
    out[INDEX(7, lane)] = in[LANE_COUNT * 7 + lane];
    out[INDEX(8, lane)] = in[LANE_COUNT * 8 + lane];
    out[INDEX(9, lane)] = in[LANE_COUNT * 9 + lane];
    out[INDEX(10, lane)] = in[LANE_COUNT * 10 + lane];
    out[INDEX(11, lane)] = in[LANE_COUNT * 11 + lane];
    out[INDEX(12, lane)] = in[LANE_COUNT * 12 + lane];
    out[INDEX(13, lane)] = in[LANE_COUNT * 13 + lane];
    out[INDEX(14, lane)] = in[LANE_COUNT * 14 + lane];
    out[INDEX(15, lane)] = in[LANE_COUNT * 15 + lane];
}

__device__ void _bit_unpack_16_16bw_32t(const uint16_t *__restrict in, uint16_t *__restrict out, int thread_idx) {
    _bit_unpack_16_16bw_lane(in, out, thread_idx * 2 + 0);
    _bit_unpack_16_16bw_lane(in, out, thread_idx * 2 + 1);
}

extern "C" __global__ void bit_unpack_16_16bw_32t(const uint16_t *__restrict full_in, uint16_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 16 / sizeof(uint16_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_16_16bw_32t(in, out, thread_idx);
}

