// AUTO-GENERATED. Do not edit by hand!
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "fastlanes_common.cuh"

__device__ void _bit_unpack_32_0bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t zero = 0ULL;
    
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
    out[INDEX(16, lane)] = zero;
    out[INDEX(17, lane)] = zero;
    out[INDEX(18, lane)] = zero;
    out[INDEX(19, lane)] = zero;
    out[INDEX(20, lane)] = zero;
    out[INDEX(21, lane)] = zero;
    out[INDEX(22, lane)] = zero;
    out[INDEX(23, lane)] = zero;
    out[INDEX(24, lane)] = zero;
    out[INDEX(25, lane)] = zero;
    out[INDEX(26, lane)] = zero;
    out[INDEX(27, lane)] = zero;
    out[INDEX(28, lane)] = zero;
    out[INDEX(29, lane)] = zero;
    out[INDEX(30, lane)] = zero;
    out[INDEX(31, lane)] = zero;
}

__device__ void _bit_unpack_32_0bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_0bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_0bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 0 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_0bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_1bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 1);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint32_t, 1);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 1);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint32_t, 1);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 1);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint32_t, 1);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 1);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint32_t, 1);
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 1);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint32_t, 1);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 1);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint32_t, 1);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 1);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint32_t, 1);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 1);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint32_t, 1);
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 1);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 17) & MASK(uint32_t, 1);
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 1);
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 19) & MASK(uint32_t, 1);
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 1);
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 21) & MASK(uint32_t, 1);
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 1);
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 23) & MASK(uint32_t, 1);
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 1);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 25) & MASK(uint32_t, 1);
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 1);
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 27) & MASK(uint32_t, 1);
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 1);
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 29) & MASK(uint32_t, 1);
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 1);
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 31) & MASK(uint32_t, 1);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_1bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_1bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_1bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 1 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_1bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_2bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 2);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 2);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 2);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 2);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 2);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 2);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 2);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 2);
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 2);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 2);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 2);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 2);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 2);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 2);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 2);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 0)) << 2;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 2);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 2);
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 2);
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 2);
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 2);
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 2);
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 2);
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 2);
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 2);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 2);
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 2);
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 2);
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 2);
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 2);
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 2);
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_2bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_2bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_2bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 2 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_2bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_3bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 3);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint32_t, 3);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 3);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint32_t, 3);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 3);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint32_t, 3);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 3);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 21) & MASK(uint32_t, 3);
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 3);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 27) & MASK(uint32_t, 3);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 1)) << 2;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint32_t, 3);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 3);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint32_t, 3);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 3);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint32_t, 3);
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 3);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 19) & MASK(uint32_t, 3);
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 3);
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 25) & MASK(uint32_t, 3);
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 3);
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 31) & MASK(uint32_t, 1);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 2)) << 1;
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 3);
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint32_t, 3);
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 3);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint32_t, 3);
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 3);
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 17) & MASK(uint32_t, 3);
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 3);
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 23) & MASK(uint32_t, 3);
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 3);
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 29) & MASK(uint32_t, 3);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_3bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_3bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_3bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 3 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_3bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_4bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 4);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 4);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 4);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 4);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 4);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 4);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 4);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 0)) << 4;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 4);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 4);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 4);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 4);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 4);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 4);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 4);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 0)) << 4;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 4);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 4);
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 4);
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 4);
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 4);
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 4);
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 4);
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 0)) << 4;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 4);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 4);
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 4);
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 4);
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 4);
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 4);
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 4);
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_4bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_4bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_4bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 4 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_4bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_5bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 5);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint32_t, 5);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 5);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint32_t, 5);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 5);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 25) & MASK(uint32_t, 5);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 3)) << 2;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint32_t, 5);
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 5);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint32_t, 5);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 5);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 23) & MASK(uint32_t, 5);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 1)) << 4;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint32_t, 5);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 5);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint32_t, 5);
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 5);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 21) & MASK(uint32_t, 5);
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 5);
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 31) & MASK(uint32_t, 1);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 4)) << 1;
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 5);
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint32_t, 5);
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 5);
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 19) & MASK(uint32_t, 5);
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 5);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 29) & MASK(uint32_t, 3);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 2)) << 3;
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 5);
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint32_t, 5);
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 5);
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 17) & MASK(uint32_t, 5);
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 5);
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 27) & MASK(uint32_t, 5);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_5bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_5bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_5bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 5 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_5bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_6bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 6);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 6);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 6);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 6);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 6);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 4)) << 2;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 6);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 6);
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 6);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 6);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 2)) << 4;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 6);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 6);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 6);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 6);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 0)) << 6;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 6);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 6);
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 6);
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 6);
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 6);
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 4)) << 2;
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 6);
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 6);
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 6);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 6);
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 2)) << 4;
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 6);
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 6);
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 6);
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 6);
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_6bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_6bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_6bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 6 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_6bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_7bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 7);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint32_t, 7);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 7);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 21) & MASK(uint32_t, 7);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 3)) << 4;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint32_t, 7);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 7);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 17) & MASK(uint32_t, 7);
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 7);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 31) & MASK(uint32_t, 1);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 6)) << 1;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 7);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint32_t, 7);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 7);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 27) & MASK(uint32_t, 5);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 2)) << 5;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 7);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint32_t, 7);
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 7);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 23) & MASK(uint32_t, 7);
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 5)) << 2;
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint32_t, 7);
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 7);
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 19) & MASK(uint32_t, 7);
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 1)) << 6;
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint32_t, 7);
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 7);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint32_t, 7);
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 7);
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 29) & MASK(uint32_t, 3);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 4)) << 3;
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 7);
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint32_t, 7);
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 7);
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 25) & MASK(uint32_t, 7);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_7bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_7bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_7bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 7 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_7bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_8bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 8);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 8);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 8);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 0)) << 8;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 8);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 8);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 8);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 0)) << 8;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 8);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 8);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 8);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 0)) << 8;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 8);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 8);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 8);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 0)) << 8;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 8);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 8);
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 8);
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 0)) << 8;
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 8);
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 8);
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 8);
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 0)) << 8;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 8);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 8);
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 8);
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 0)) << 8;
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 8);
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 8);
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 8);
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_8bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_8bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_8bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 8 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_8bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_9bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 9);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint32_t, 9);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 9);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 27) & MASK(uint32_t, 5);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 4)) << 5;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 9);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint32_t, 9);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 9);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 31) & MASK(uint32_t, 1);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 8)) << 1;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 9);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 17) & MASK(uint32_t, 9);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 3)) << 6;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint32_t, 9);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 9);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 21) & MASK(uint32_t, 9);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 7)) << 2;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint32_t, 9);
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 9);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 25) & MASK(uint32_t, 7);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 2)) << 7;
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 9);
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint32_t, 9);
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 9);
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 29) & MASK(uint32_t, 3);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 6)) << 3;
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 9);
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint32_t, 9);
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 1)) << 8;
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint32_t, 9);
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 9);
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 19) & MASK(uint32_t, 9);
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 5)) << 4;
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint32_t, 9);
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 9);
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 23) & MASK(uint32_t, 9);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_9bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_9bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_9bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 9 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_9bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_10bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 10);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 10);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 10);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 8)) << 2;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 10);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 10);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 6)) << 4;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 10);
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 10);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 4)) << 6;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 10);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 10);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 2)) << 8;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 10);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 10);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 0)) << 10;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 10);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 10);
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 10);
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 8)) << 2;
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 10);
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 10);
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 6)) << 4;
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 10);
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 10);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 4)) << 6;
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 10);
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 10);
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 2)) << 8;
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 10);
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 10);
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_10bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_10bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_10bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 10 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_10bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_11bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 11);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint32_t, 11);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 1)) << 10;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint32_t, 11);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 11);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 23) & MASK(uint32_t, 9);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 2)) << 9;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 11);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint32_t, 11);
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 3)) << 8;
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint32_t, 11);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 11);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 25) & MASK(uint32_t, 7);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 4)) << 7;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 11);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint32_t, 11);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 5)) << 6;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint32_t, 11);
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 11);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 27) & MASK(uint32_t, 5);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 6)) << 5;
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 11);
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 17) & MASK(uint32_t, 11);
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 7)) << 4;
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint32_t, 11);
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 11);
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 29) & MASK(uint32_t, 3);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 8)) << 3;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 11);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 19) & MASK(uint32_t, 11);
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 9)) << 2;
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint32_t, 11);
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 11);
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 31) & MASK(uint32_t, 1);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 10)) << 1;
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 11);
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 21) & MASK(uint32_t, 11);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_11bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_11bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_11bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 11 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_11bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_12bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 12);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 12);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 4)) << 8;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 12);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 12);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 8)) << 4;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 12);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 0)) << 12;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 12);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 12);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 4)) << 8;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 12);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 12);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 8)) << 4;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 12);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 0)) << 12;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 12);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 12);
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 4)) << 8;
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 12);
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 12);
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 8)) << 4;
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 12);
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 0)) << 12;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 12);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 12);
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 4)) << 8;
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 12);
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 12);
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 8)) << 4;
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 12);
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_12bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_12bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_12bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 12 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_12bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_13bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 13);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint32_t, 13);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 7)) << 6;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint32_t, 13);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 1)) << 12;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint32_t, 13);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 13);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 27) & MASK(uint32_t, 5);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 8)) << 5;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 13);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 21) & MASK(uint32_t, 11);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 2)) << 11;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 13);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint32_t, 13);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 9)) << 4;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint32_t, 13);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 3)) << 10;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint32_t, 13);
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 13);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 29) & MASK(uint32_t, 3);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 10)) << 3;
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 13);
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 23) & MASK(uint32_t, 9);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 4)) << 9;
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 13);
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 17) & MASK(uint32_t, 13);
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 11)) << 2;
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint32_t, 13);
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 5)) << 8;
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint32_t, 13);
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 13);
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 31) & MASK(uint32_t, 1);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 12)) << 1;
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 13);
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 25) & MASK(uint32_t, 7);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 6)) << 7;
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 13);
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 19) & MASK(uint32_t, 13);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_13bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_13bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_13bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 13 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_13bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_14bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 14);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 14);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 10)) << 4;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 14);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 6)) << 8;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 14);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 2)) << 12;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 14);
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 14);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 12)) << 2;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 14);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 8)) << 6;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 14);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 4)) << 10;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 14);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 0)) << 14;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 14);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 14);
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 10)) << 4;
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 14);
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 6)) << 8;
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 14);
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 2)) << 12;
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 14);
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 14);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 12)) << 2;
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 14);
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 8)) << 6;
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 14);
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 4)) << 10;
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 14);
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_14bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_14bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_14bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 14 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_14bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_15bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 15);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint32_t, 15);
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 13)) << 2;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint32_t, 15);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 11)) << 4;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint32_t, 15);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 9)) << 6;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint32_t, 15);
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 7)) << 8;
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint32_t, 15);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 5)) << 10;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint32_t, 15);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 3)) << 12;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint32_t, 15);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 1)) << 14;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint32_t, 15);
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 15);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 31) & MASK(uint32_t, 1);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 14)) << 1;
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 15);
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 29) & MASK(uint32_t, 3);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 12)) << 3;
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 15);
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 27) & MASK(uint32_t, 5);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 10)) << 5;
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 15);
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 25) & MASK(uint32_t, 7);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 8)) << 7;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 15);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 23) & MASK(uint32_t, 9);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 6)) << 9;
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 15);
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 21) & MASK(uint32_t, 11);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 4)) << 11;
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 15);
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 19) & MASK(uint32_t, 13);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint32_t, 2)) << 13;
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 15);
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 17) & MASK(uint32_t, 15);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_15bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_15bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_15bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 15 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_15bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_16bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 16);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 0)) << 16;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 16);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 0)) << 16;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 16);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 0)) << 16;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 16);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 0)) << 16;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 16);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 0)) << 16;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 16);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 0)) << 16;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 16);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 0)) << 16;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 16);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 0)) << 16;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 16);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 0)) << 16;
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 16);
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 0)) << 16;
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 16);
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 0)) << 16;
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 16);
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 0)) << 16;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 16);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 0)) << 16;
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 16);
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint32_t, 0)) << 16;
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 16);
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 15];
    tmp |= (src & MASK(uint32_t, 0)) << 16;
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 16);
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_16bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_16bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_16bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 16 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_16bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_17bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 17);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 17) & MASK(uint32_t, 15);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 2)) << 15;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 17);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 19) & MASK(uint32_t, 13);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 4)) << 13;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 17);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 21) & MASK(uint32_t, 11);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 6)) << 11;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 17);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 23) & MASK(uint32_t, 9);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 8)) << 9;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 17);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 25) & MASK(uint32_t, 7);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 10)) << 7;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 17);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 27) & MASK(uint32_t, 5);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 12)) << 5;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 17);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 29) & MASK(uint32_t, 3);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 14)) << 3;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 17);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 31) & MASK(uint32_t, 1);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 16)) << 1;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 1)) << 16;
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint32_t, 17);
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 3)) << 14;
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint32_t, 17);
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 5)) << 12;
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint32_t, 17);
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 7)) << 10;
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint32_t, 17);
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 9)) << 8;
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint32_t, 17);
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint32_t, 11)) << 6;
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint32_t, 17);
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 15];
    tmp |= (src & MASK(uint32_t, 13)) << 4;
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint32_t, 17);
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 16];
    tmp |= (src & MASK(uint32_t, 15)) << 2;
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint32_t, 17);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_17bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_17bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_17bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 17 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_17bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_18bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 18);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 4)) << 14;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 18);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 8)) << 10;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 18);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 12)) << 6;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 18);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 16)) << 2;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 2)) << 16;
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 18);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 6)) << 12;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 18);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 10)) << 8;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 18);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 14)) << 4;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 18);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 0)) << 18;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 18);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 4)) << 14;
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 18);
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 8)) << 10;
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 18);
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 12)) << 6;
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 18);
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 16)) << 2;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint32_t, 2)) << 16;
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 18);
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 15];
    tmp |= (src & MASK(uint32_t, 6)) << 12;
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 18);
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 16];
    tmp |= (src & MASK(uint32_t, 10)) << 8;
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 18);
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 17];
    tmp |= (src & MASK(uint32_t, 14)) << 4;
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 18);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_18bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_18bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_18bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 18 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_18bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_19bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 19);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 19) & MASK(uint32_t, 13);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 6)) << 13;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 19);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 25) & MASK(uint32_t, 7);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 12)) << 7;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 19);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 31) & MASK(uint32_t, 1);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 18)) << 1;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 5)) << 14;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint32_t, 19);
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 11)) << 8;
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint32_t, 19);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 17)) << 2;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 17) & MASK(uint32_t, 15);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 4)) << 15;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 19);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 23) & MASK(uint32_t, 9);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 10)) << 9;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 19);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 29) & MASK(uint32_t, 3);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 16)) << 3;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 3)) << 16;
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint32_t, 19);
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 9)) << 10;
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint32_t, 19);
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 15)) << 4;
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint32_t, 17);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 2)) << 17;
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 19);
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 21) & MASK(uint32_t, 11);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint32_t, 8)) << 11;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 19);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 27) & MASK(uint32_t, 5);
    src = in[lane + LANE_COUNT * 15];
    tmp |= (src & MASK(uint32_t, 14)) << 5;
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 18);
    src = in[lane + LANE_COUNT * 16];
    tmp |= (src & MASK(uint32_t, 1)) << 18;
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint32_t, 19);
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 17];
    tmp |= (src & MASK(uint32_t, 7)) << 12;
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint32_t, 19);
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 18];
    tmp |= (src & MASK(uint32_t, 13)) << 6;
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint32_t, 19);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_19bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_19bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_19bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 19 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_19bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_20bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 20);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 8)) << 12;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 20);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 16)) << 4;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 4)) << 16;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 20);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 12)) << 8;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 0)) << 20;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 20);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 8)) << 12;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 20);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 16)) << 4;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 4)) << 16;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 20);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 12)) << 8;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 0)) << 20;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 20);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 8)) << 12;
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 20);
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 16)) << 4;
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 4)) << 16;
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 20);
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint32_t, 12)) << 8;
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 15];
    tmp |= (src & MASK(uint32_t, 0)) << 20;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 20);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 16];
    tmp |= (src & MASK(uint32_t, 8)) << 12;
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 20);
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 17];
    tmp |= (src & MASK(uint32_t, 16)) << 4;
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 18];
    tmp |= (src & MASK(uint32_t, 4)) << 16;
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 20);
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 19];
    tmp |= (src & MASK(uint32_t, 12)) << 8;
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_20bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_20bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_20bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 20 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_20bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_21bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 21);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 21) & MASK(uint32_t, 11);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 10)) << 11;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 21);
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 31) & MASK(uint32_t, 1);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 20)) << 1;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 9)) << 12;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint32_t, 21);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 19)) << 2;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 19) & MASK(uint32_t, 13);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 8)) << 13;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 21);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 29) & MASK(uint32_t, 3);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 18)) << 3;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 7)) << 14;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint32_t, 21);
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 17)) << 4;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 17) & MASK(uint32_t, 15);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 6)) << 15;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 21);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 27) & MASK(uint32_t, 5);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 16)) << 5;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 5)) << 16;
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint32_t, 21);
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 15)) << 6;
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint32_t, 17);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 4)) << 17;
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 21);
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 25) & MASK(uint32_t, 7);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint32_t, 14)) << 7;
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 18);
    src = in[lane + LANE_COUNT * 15];
    tmp |= (src & MASK(uint32_t, 3)) << 18;
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint32_t, 21);
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 16];
    tmp |= (src & MASK(uint32_t, 13)) << 8;
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint32_t, 19);
    src = in[lane + LANE_COUNT * 17];
    tmp |= (src & MASK(uint32_t, 2)) << 19;
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 21);
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 23) & MASK(uint32_t, 9);
    src = in[lane + LANE_COUNT * 18];
    tmp |= (src & MASK(uint32_t, 12)) << 9;
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 19];
    tmp |= (src & MASK(uint32_t, 1)) << 20;
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint32_t, 21);
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 20];
    tmp |= (src & MASK(uint32_t, 11)) << 10;
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint32_t, 21);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_21bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_21bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_21bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 21 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_21bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_22bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 22);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 12)) << 10;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 2)) << 20;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 22);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 14)) << 8;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 18);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 4)) << 18;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 22);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 16)) << 6;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 6)) << 16;
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 22);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 18)) << 4;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 8)) << 14;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 22);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 20)) << 2;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 10)) << 12;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 22);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 0)) << 22;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 22);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 12)) << 10;
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 2)) << 20;
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 22);
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint32_t, 14)) << 8;
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 18);
    src = in[lane + LANE_COUNT * 15];
    tmp |= (src & MASK(uint32_t, 4)) << 18;
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 22);
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 16];
    tmp |= (src & MASK(uint32_t, 16)) << 6;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 17];
    tmp |= (src & MASK(uint32_t, 6)) << 16;
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 22);
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 18];
    tmp |= (src & MASK(uint32_t, 18)) << 4;
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 19];
    tmp |= (src & MASK(uint32_t, 8)) << 14;
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 22);
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 20];
    tmp |= (src & MASK(uint32_t, 20)) << 2;
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 21];
    tmp |= (src & MASK(uint32_t, 10)) << 12;
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 22);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_22bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_22bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_22bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 22 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_22bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_23bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 23);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 23) & MASK(uint32_t, 9);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 14)) << 9;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 18);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 5)) << 18;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint32_t, 23);
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 19)) << 4;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 19) & MASK(uint32_t, 13);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 10)) << 13;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 22);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 1)) << 22;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint32_t, 23);
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 15)) << 8;
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint32_t, 17);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 6)) << 17;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 23);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 29) & MASK(uint32_t, 3);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 20)) << 3;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 11)) << 12;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint32_t, 21);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 2)) << 21;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 23);
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 25) & MASK(uint32_t, 7);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 16)) << 7;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 7)) << 16;
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint32_t, 23);
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 21)) << 2;
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 21) & MASK(uint32_t, 11);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint32_t, 12)) << 11;
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 15];
    tmp |= (src & MASK(uint32_t, 3)) << 20;
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint32_t, 23);
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 16];
    tmp |= (src & MASK(uint32_t, 17)) << 6;
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 17) & MASK(uint32_t, 15);
    src = in[lane + LANE_COUNT * 17];
    tmp |= (src & MASK(uint32_t, 8)) << 15;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 23);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 31) & MASK(uint32_t, 1);
    src = in[lane + LANE_COUNT * 18];
    tmp |= (src & MASK(uint32_t, 22)) << 1;
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 19];
    tmp |= (src & MASK(uint32_t, 13)) << 10;
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint32_t, 19);
    src = in[lane + LANE_COUNT * 20];
    tmp |= (src & MASK(uint32_t, 4)) << 19;
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 23);
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 27) & MASK(uint32_t, 5);
    src = in[lane + LANE_COUNT * 21];
    tmp |= (src & MASK(uint32_t, 18)) << 5;
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 22];
    tmp |= (src & MASK(uint32_t, 9)) << 14;
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint32_t, 23);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_23bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_23bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_23bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 23 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_23bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_24bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 24);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 16)) << 8;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 8)) << 16;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 0)) << 24;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 24);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 16)) << 8;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 8)) << 16;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 0)) << 24;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 24);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 16)) << 8;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 8)) << 16;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 0)) << 24;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 24);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 16)) << 8;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 8)) << 16;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 0)) << 24;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 24);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 16)) << 8;
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint32_t, 8)) << 16;
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 15];
    tmp |= (src & MASK(uint32_t, 0)) << 24;
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 24);
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 16];
    tmp |= (src & MASK(uint32_t, 16)) << 8;
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 17];
    tmp |= (src & MASK(uint32_t, 8)) << 16;
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 18];
    tmp |= (src & MASK(uint32_t, 0)) << 24;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 24);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 19];
    tmp |= (src & MASK(uint32_t, 16)) << 8;
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 20];
    tmp |= (src & MASK(uint32_t, 8)) << 16;
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 21];
    tmp |= (src & MASK(uint32_t, 0)) << 24;
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 24);
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 22];
    tmp |= (src & MASK(uint32_t, 16)) << 8;
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 23];
    tmp |= (src & MASK(uint32_t, 8)) << 16;
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_24bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_24bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_24bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 24 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_24bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_25bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 25);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 25) & MASK(uint32_t, 7);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 18)) << 7;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 11)) << 14;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint32_t, 21);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 4)) << 21;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 25);
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 29) & MASK(uint32_t, 3);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 22)) << 3;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 15)) << 10;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint32_t, 17);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 8)) << 17;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 1)) << 24;
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint32_t, 25);
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 19)) << 6;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 19) & MASK(uint32_t, 13);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 12)) << 13;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 5)) << 20;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint32_t, 25);
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 23)) << 2;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 23) & MASK(uint32_t, 9);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 16)) << 9;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 9)) << 16;
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint32_t, 23);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint32_t, 2)) << 23;
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 25);
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 27) & MASK(uint32_t, 5);
    src = in[lane + LANE_COUNT * 15];
    tmp |= (src & MASK(uint32_t, 20)) << 5;
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 16];
    tmp |= (src & MASK(uint32_t, 13)) << 12;
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint32_t, 19);
    src = in[lane + LANE_COUNT * 17];
    tmp |= (src & MASK(uint32_t, 6)) << 19;
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 25);
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 31) & MASK(uint32_t, 1);
    src = in[lane + LANE_COUNT * 18];
    tmp |= (src & MASK(uint32_t, 24)) << 1;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 19];
    tmp |= (src & MASK(uint32_t, 17)) << 8;
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 17) & MASK(uint32_t, 15);
    src = in[lane + LANE_COUNT * 20];
    tmp |= (src & MASK(uint32_t, 10)) << 15;
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 22);
    src = in[lane + LANE_COUNT * 21];
    tmp |= (src & MASK(uint32_t, 3)) << 22;
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint32_t, 25);
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 22];
    tmp |= (src & MASK(uint32_t, 21)) << 4;
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 21) & MASK(uint32_t, 11);
    src = in[lane + LANE_COUNT * 23];
    tmp |= (src & MASK(uint32_t, 14)) << 11;
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 18);
    src = in[lane + LANE_COUNT * 24];
    tmp |= (src & MASK(uint32_t, 7)) << 18;
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint32_t, 25);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_25bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_25bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_25bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 25 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_25bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_26bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 26);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 20)) << 6;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 14)) << 12;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 18);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 8)) << 18;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 2)) << 24;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 26);
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 22)) << 4;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 16)) << 10;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 10)) << 16;
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 22);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 4)) << 22;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 26);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 24)) << 2;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 18)) << 8;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 12)) << 14;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 6)) << 20;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 26);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 0)) << 26;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 26);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint32_t, 20)) << 6;
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 15];
    tmp |= (src & MASK(uint32_t, 14)) << 12;
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 18);
    src = in[lane + LANE_COUNT * 16];
    tmp |= (src & MASK(uint32_t, 8)) << 18;
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 17];
    tmp |= (src & MASK(uint32_t, 2)) << 24;
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 26);
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 18];
    tmp |= (src & MASK(uint32_t, 22)) << 4;
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 19];
    tmp |= (src & MASK(uint32_t, 16)) << 10;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 20];
    tmp |= (src & MASK(uint32_t, 10)) << 16;
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 22);
    src = in[lane + LANE_COUNT * 21];
    tmp |= (src & MASK(uint32_t, 4)) << 22;
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 26);
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 22];
    tmp |= (src & MASK(uint32_t, 24)) << 2;
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 23];
    tmp |= (src & MASK(uint32_t, 18)) << 8;
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 24];
    tmp |= (src & MASK(uint32_t, 12)) << 14;
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 25];
    tmp |= (src & MASK(uint32_t, 6)) << 20;
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 26);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_26bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_26bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_26bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 26 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_26bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_27bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 27);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 27) & MASK(uint32_t, 5);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 22)) << 5;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 17)) << 10;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 17) & MASK(uint32_t, 15);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 12)) << 15;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 7)) << 20;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint32_t, 25);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 2)) << 25;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 27);
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 29) & MASK(uint32_t, 3);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 24)) << 3;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 19)) << 8;
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 19) & MASK(uint32_t, 13);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 14)) << 13;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 18);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 9)) << 18;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint32_t, 23);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 4)) << 23;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 27);
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 31) & MASK(uint32_t, 1);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 26)) << 1;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 21)) << 6;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 21) & MASK(uint32_t, 11);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 16)) << 11;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint32_t, 11)) << 16;
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint32_t, 21);
    src = in[lane + LANE_COUNT * 15];
    tmp |= (src & MASK(uint32_t, 6)) << 21;
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 26);
    src = in[lane + LANE_COUNT * 16];
    tmp |= (src & MASK(uint32_t, 1)) << 26;
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint32_t, 27);
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 17];
    tmp |= (src & MASK(uint32_t, 23)) << 4;
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 23) & MASK(uint32_t, 9);
    src = in[lane + LANE_COUNT * 18];
    tmp |= (src & MASK(uint32_t, 18)) << 9;
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 19];
    tmp |= (src & MASK(uint32_t, 13)) << 14;
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint32_t, 19);
    src = in[lane + LANE_COUNT * 20];
    tmp |= (src & MASK(uint32_t, 8)) << 19;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 21];
    tmp |= (src & MASK(uint32_t, 3)) << 24;
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint32_t, 27);
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 22];
    tmp |= (src & MASK(uint32_t, 25)) << 2;
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 25) & MASK(uint32_t, 7);
    src = in[lane + LANE_COUNT * 23];
    tmp |= (src & MASK(uint32_t, 20)) << 7;
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 24];
    tmp |= (src & MASK(uint32_t, 15)) << 12;
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint32_t, 17);
    src = in[lane + LANE_COUNT * 25];
    tmp |= (src & MASK(uint32_t, 10)) << 17;
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 22);
    src = in[lane + LANE_COUNT * 26];
    tmp |= (src & MASK(uint32_t, 5)) << 22;
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint32_t, 27);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_27bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_27bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_27bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 27 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_27bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_28bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 28);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 24)) << 4;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 20)) << 8;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 16)) << 12;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 12)) << 16;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 8)) << 20;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 4)) << 24;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 28);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 0)) << 28;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 28);
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 24)) << 4;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 20)) << 8;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 16)) << 12;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 12)) << 16;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 8)) << 20;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 4)) << 24;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 28);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint32_t, 0)) << 28;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 28);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 15];
    tmp |= (src & MASK(uint32_t, 24)) << 4;
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 16];
    tmp |= (src & MASK(uint32_t, 20)) << 8;
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 17];
    tmp |= (src & MASK(uint32_t, 16)) << 12;
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 18];
    tmp |= (src & MASK(uint32_t, 12)) << 16;
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 19];
    tmp |= (src & MASK(uint32_t, 8)) << 20;
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 20];
    tmp |= (src & MASK(uint32_t, 4)) << 24;
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 28);
    src = in[lane + LANE_COUNT * 21];
    tmp |= (src & MASK(uint32_t, 0)) << 28;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 28);
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 22];
    tmp |= (src & MASK(uint32_t, 24)) << 4;
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 23];
    tmp |= (src & MASK(uint32_t, 20)) << 8;
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 24];
    tmp |= (src & MASK(uint32_t, 16)) << 12;
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 25];
    tmp |= (src & MASK(uint32_t, 12)) << 16;
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 26];
    tmp |= (src & MASK(uint32_t, 8)) << 20;
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 27];
    tmp |= (src & MASK(uint32_t, 4)) << 24;
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 28);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_28bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_28bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_28bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 28 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_28bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_29bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 29);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 29) & MASK(uint32_t, 3);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 26)) << 3;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 23)) << 6;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 23) & MASK(uint32_t, 9);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 20)) << 9;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 17)) << 12;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 17) & MASK(uint32_t, 15);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 14)) << 15;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 18);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 11)) << 18;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint32_t, 21);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 8)) << 21;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 5)) << 24;
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint32_t, 27);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 2)) << 27;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 29);
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 31) & MASK(uint32_t, 1);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 28)) << 1;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 25)) << 4;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 25) & MASK(uint32_t, 7);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 22)) << 7;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 19)) << 10;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 19) & MASK(uint32_t, 13);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint32_t, 16)) << 13;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 15];
    tmp |= (src & MASK(uint32_t, 13)) << 16;
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint32_t, 19);
    src = in[lane + LANE_COUNT * 16];
    tmp |= (src & MASK(uint32_t, 10)) << 19;
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 22);
    src = in[lane + LANE_COUNT * 17];
    tmp |= (src & MASK(uint32_t, 7)) << 22;
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint32_t, 25);
    src = in[lane + LANE_COUNT * 18];
    tmp |= (src & MASK(uint32_t, 4)) << 25;
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 28);
    src = in[lane + LANE_COUNT * 19];
    tmp |= (src & MASK(uint32_t, 1)) << 28;
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint32_t, 29);
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 20];
    tmp |= (src & MASK(uint32_t, 27)) << 2;
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 27) & MASK(uint32_t, 5);
    src = in[lane + LANE_COUNT * 21];
    tmp |= (src & MASK(uint32_t, 24)) << 5;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 22];
    tmp |= (src & MASK(uint32_t, 21)) << 8;
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 21) & MASK(uint32_t, 11);
    src = in[lane + LANE_COUNT * 23];
    tmp |= (src & MASK(uint32_t, 18)) << 11;
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 24];
    tmp |= (src & MASK(uint32_t, 15)) << 14;
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint32_t, 17);
    src = in[lane + LANE_COUNT * 25];
    tmp |= (src & MASK(uint32_t, 12)) << 17;
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 26];
    tmp |= (src & MASK(uint32_t, 9)) << 20;
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint32_t, 23);
    src = in[lane + LANE_COUNT * 27];
    tmp |= (src & MASK(uint32_t, 6)) << 23;
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 26);
    src = in[lane + LANE_COUNT * 28];
    tmp |= (src & MASK(uint32_t, 3)) << 26;
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint32_t, 29);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_29bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_29bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_29bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 29 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_29bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_30bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 30);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 28)) << 2;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 26)) << 4;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 24)) << 6;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 22)) << 8;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 20)) << 10;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 18)) << 12;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 16)) << 14;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 14)) << 16;
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 18);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 12)) << 18;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 10)) << 20;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 22);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 8)) << 22;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 6)) << 24;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 26);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 4)) << 26;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 28);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint32_t, 2)) << 28;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 30);
    src = in[lane + LANE_COUNT * 15];
    tmp |= (src & MASK(uint32_t, 0)) << 30;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 0) & MASK(uint32_t, 30);
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 16];
    tmp |= (src & MASK(uint32_t, 28)) << 2;
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 17];
    tmp |= (src & MASK(uint32_t, 26)) << 4;
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 18];
    tmp |= (src & MASK(uint32_t, 24)) << 6;
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 19];
    tmp |= (src & MASK(uint32_t, 22)) << 8;
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 20];
    tmp |= (src & MASK(uint32_t, 20)) << 10;
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 21];
    tmp |= (src & MASK(uint32_t, 18)) << 12;
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 22];
    tmp |= (src & MASK(uint32_t, 16)) << 14;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 23];
    tmp |= (src & MASK(uint32_t, 14)) << 16;
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 18);
    src = in[lane + LANE_COUNT * 24];
    tmp |= (src & MASK(uint32_t, 12)) << 18;
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 25];
    tmp |= (src & MASK(uint32_t, 10)) << 20;
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 22);
    src = in[lane + LANE_COUNT * 26];
    tmp |= (src & MASK(uint32_t, 8)) << 22;
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 27];
    tmp |= (src & MASK(uint32_t, 6)) << 24;
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 26);
    src = in[lane + LANE_COUNT * 28];
    tmp |= (src & MASK(uint32_t, 4)) << 26;
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 28);
    src = in[lane + LANE_COUNT * 29];
    tmp |= (src & MASK(uint32_t, 2)) << 28;
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 30);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_30bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_30bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_30bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 30 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_30bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_31bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;
    
    src = in[lane];
    tmp = (src >> 0) & MASK(uint32_t, 31);
    out[INDEX(0, lane)] = tmp;
    tmp = (src >> 31) & MASK(uint32_t, 1);
    src = in[lane + LANE_COUNT * 1];
    tmp |= (src & MASK(uint32_t, 30)) << 1;
    out[INDEX(1, lane)] = tmp;
    tmp = (src >> 30) & MASK(uint32_t, 2);
    src = in[lane + LANE_COUNT * 2];
    tmp |= (src & MASK(uint32_t, 29)) << 2;
    out[INDEX(2, lane)] = tmp;
    tmp = (src >> 29) & MASK(uint32_t, 3);
    src = in[lane + LANE_COUNT * 3];
    tmp |= (src & MASK(uint32_t, 28)) << 3;
    out[INDEX(3, lane)] = tmp;
    tmp = (src >> 28) & MASK(uint32_t, 4);
    src = in[lane + LANE_COUNT * 4];
    tmp |= (src & MASK(uint32_t, 27)) << 4;
    out[INDEX(4, lane)] = tmp;
    tmp = (src >> 27) & MASK(uint32_t, 5);
    src = in[lane + LANE_COUNT * 5];
    tmp |= (src & MASK(uint32_t, 26)) << 5;
    out[INDEX(5, lane)] = tmp;
    tmp = (src >> 26) & MASK(uint32_t, 6);
    src = in[lane + LANE_COUNT * 6];
    tmp |= (src & MASK(uint32_t, 25)) << 6;
    out[INDEX(6, lane)] = tmp;
    tmp = (src >> 25) & MASK(uint32_t, 7);
    src = in[lane + LANE_COUNT * 7];
    tmp |= (src & MASK(uint32_t, 24)) << 7;
    out[INDEX(7, lane)] = tmp;
    tmp = (src >> 24) & MASK(uint32_t, 8);
    src = in[lane + LANE_COUNT * 8];
    tmp |= (src & MASK(uint32_t, 23)) << 8;
    out[INDEX(8, lane)] = tmp;
    tmp = (src >> 23) & MASK(uint32_t, 9);
    src = in[lane + LANE_COUNT * 9];
    tmp |= (src & MASK(uint32_t, 22)) << 9;
    out[INDEX(9, lane)] = tmp;
    tmp = (src >> 22) & MASK(uint32_t, 10);
    src = in[lane + LANE_COUNT * 10];
    tmp |= (src & MASK(uint32_t, 21)) << 10;
    out[INDEX(10, lane)] = tmp;
    tmp = (src >> 21) & MASK(uint32_t, 11);
    src = in[lane + LANE_COUNT * 11];
    tmp |= (src & MASK(uint32_t, 20)) << 11;
    out[INDEX(11, lane)] = tmp;
    tmp = (src >> 20) & MASK(uint32_t, 12);
    src = in[lane + LANE_COUNT * 12];
    tmp |= (src & MASK(uint32_t, 19)) << 12;
    out[INDEX(12, lane)] = tmp;
    tmp = (src >> 19) & MASK(uint32_t, 13);
    src = in[lane + LANE_COUNT * 13];
    tmp |= (src & MASK(uint32_t, 18)) << 13;
    out[INDEX(13, lane)] = tmp;
    tmp = (src >> 18) & MASK(uint32_t, 14);
    src = in[lane + LANE_COUNT * 14];
    tmp |= (src & MASK(uint32_t, 17)) << 14;
    out[INDEX(14, lane)] = tmp;
    tmp = (src >> 17) & MASK(uint32_t, 15);
    src = in[lane + LANE_COUNT * 15];
    tmp |= (src & MASK(uint32_t, 16)) << 15;
    out[INDEX(15, lane)] = tmp;
    tmp = (src >> 16) & MASK(uint32_t, 16);
    src = in[lane + LANE_COUNT * 16];
    tmp |= (src & MASK(uint32_t, 15)) << 16;
    out[INDEX(16, lane)] = tmp;
    tmp = (src >> 15) & MASK(uint32_t, 17);
    src = in[lane + LANE_COUNT * 17];
    tmp |= (src & MASK(uint32_t, 14)) << 17;
    out[INDEX(17, lane)] = tmp;
    tmp = (src >> 14) & MASK(uint32_t, 18);
    src = in[lane + LANE_COUNT * 18];
    tmp |= (src & MASK(uint32_t, 13)) << 18;
    out[INDEX(18, lane)] = tmp;
    tmp = (src >> 13) & MASK(uint32_t, 19);
    src = in[lane + LANE_COUNT * 19];
    tmp |= (src & MASK(uint32_t, 12)) << 19;
    out[INDEX(19, lane)] = tmp;
    tmp = (src >> 12) & MASK(uint32_t, 20);
    src = in[lane + LANE_COUNT * 20];
    tmp |= (src & MASK(uint32_t, 11)) << 20;
    out[INDEX(20, lane)] = tmp;
    tmp = (src >> 11) & MASK(uint32_t, 21);
    src = in[lane + LANE_COUNT * 21];
    tmp |= (src & MASK(uint32_t, 10)) << 21;
    out[INDEX(21, lane)] = tmp;
    tmp = (src >> 10) & MASK(uint32_t, 22);
    src = in[lane + LANE_COUNT * 22];
    tmp |= (src & MASK(uint32_t, 9)) << 22;
    out[INDEX(22, lane)] = tmp;
    tmp = (src >> 9) & MASK(uint32_t, 23);
    src = in[lane + LANE_COUNT * 23];
    tmp |= (src & MASK(uint32_t, 8)) << 23;
    out[INDEX(23, lane)] = tmp;
    tmp = (src >> 8) & MASK(uint32_t, 24);
    src = in[lane + LANE_COUNT * 24];
    tmp |= (src & MASK(uint32_t, 7)) << 24;
    out[INDEX(24, lane)] = tmp;
    tmp = (src >> 7) & MASK(uint32_t, 25);
    src = in[lane + LANE_COUNT * 25];
    tmp |= (src & MASK(uint32_t, 6)) << 25;
    out[INDEX(25, lane)] = tmp;
    tmp = (src >> 6) & MASK(uint32_t, 26);
    src = in[lane + LANE_COUNT * 26];
    tmp |= (src & MASK(uint32_t, 5)) << 26;
    out[INDEX(26, lane)] = tmp;
    tmp = (src >> 5) & MASK(uint32_t, 27);
    src = in[lane + LANE_COUNT * 27];
    tmp |= (src & MASK(uint32_t, 4)) << 27;
    out[INDEX(27, lane)] = tmp;
    tmp = (src >> 4) & MASK(uint32_t, 28);
    src = in[lane + LANE_COUNT * 28];
    tmp |= (src & MASK(uint32_t, 3)) << 28;
    out[INDEX(28, lane)] = tmp;
    tmp = (src >> 3) & MASK(uint32_t, 29);
    src = in[lane + LANE_COUNT * 29];
    tmp |= (src & MASK(uint32_t, 2)) << 29;
    out[INDEX(29, lane)] = tmp;
    tmp = (src >> 2) & MASK(uint32_t, 30);
    src = in[lane + LANE_COUNT * 30];
    tmp |= (src & MASK(uint32_t, 1)) << 30;
    out[INDEX(30, lane)] = tmp;
    tmp = (src >> 1) & MASK(uint32_t, 31);
    out[INDEX(31, lane)] = tmp;
}

__device__ void _bit_unpack_32_31bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_31bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_31bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 31 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_31bw_32t(in, out, thread_idx);
}

__device__ void _bit_unpack_32_32bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    
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
    out[INDEX(16, lane)] = in[LANE_COUNT * 16 + lane];
    out[INDEX(17, lane)] = in[LANE_COUNT * 17 + lane];
    out[INDEX(18, lane)] = in[LANE_COUNT * 18 + lane];
    out[INDEX(19, lane)] = in[LANE_COUNT * 19 + lane];
    out[INDEX(20, lane)] = in[LANE_COUNT * 20 + lane];
    out[INDEX(21, lane)] = in[LANE_COUNT * 21 + lane];
    out[INDEX(22, lane)] = in[LANE_COUNT * 22 + lane];
    out[INDEX(23, lane)] = in[LANE_COUNT * 23 + lane];
    out[INDEX(24, lane)] = in[LANE_COUNT * 24 + lane];
    out[INDEX(25, lane)] = in[LANE_COUNT * 25 + lane];
    out[INDEX(26, lane)] = in[LANE_COUNT * 26 + lane];
    out[INDEX(27, lane)] = in[LANE_COUNT * 27 + lane];
    out[INDEX(28, lane)] = in[LANE_COUNT * 28 + lane];
    out[INDEX(29, lane)] = in[LANE_COUNT * 29 + lane];
    out[INDEX(30, lane)] = in[LANE_COUNT * 30 + lane];
    out[INDEX(31, lane)] = in[LANE_COUNT * 31 + lane];
}

__device__ void _bit_unpack_32_32bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_32bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_32bw_32t(const uint32_t *__restrict full_in, uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 32 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_32bw_32t(in, out, thread_idx);
}

