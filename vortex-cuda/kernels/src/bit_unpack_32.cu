// AUTO-GENERATED. Do not edit by hand!
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "fastlanes_common.cuh"

__device__ void
_bit_unpack_32_0bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
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

__device__ void
_bit_unpack_32_1bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 1);
    out[INDEX(0, lane)] = tmp;
    tmp = BFE(src, 1, 1);
    out[INDEX(1, lane)] = tmp;
    tmp = BFE(src, 2, 1);
    out[INDEX(2, lane)] = tmp;
    tmp = BFE(src, 3, 1);
    out[INDEX(3, lane)] = tmp;
    tmp = BFE(src, 4, 1);
    out[INDEX(4, lane)] = tmp;
    tmp = BFE(src, 5, 1);
    out[INDEX(5, lane)] = tmp;
    tmp = BFE(src, 6, 1);
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 7, 1);
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 8, 1);
    out[INDEX(8, lane)] = tmp;
    tmp = BFE(src, 9, 1);
    out[INDEX(9, lane)] = tmp;
    tmp = BFE(src, 10, 1);
    out[INDEX(10, lane)] = tmp;
    tmp = BFE(src, 11, 1);
    out[INDEX(11, lane)] = tmp;
    tmp = BFE(src, 12, 1);
    out[INDEX(12, lane)] = tmp;
    tmp = BFE(src, 13, 1);
    out[INDEX(13, lane)] = tmp;
    tmp = BFE(src, 14, 1);
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 15, 1);
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 16, 1);
    out[INDEX(16, lane)] = tmp;
    tmp = BFE(src, 17, 1);
    out[INDEX(17, lane)] = tmp;
    tmp = BFE(src, 18, 1);
    out[INDEX(18, lane)] = tmp;
    tmp = BFE(src, 19, 1);
    out[INDEX(19, lane)] = tmp;
    tmp = BFE(src, 20, 1);
    out[INDEX(20, lane)] = tmp;
    tmp = BFE(src, 21, 1);
    out[INDEX(21, lane)] = tmp;
    tmp = BFE(src, 22, 1);
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 23, 1);
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 24, 1);
    out[INDEX(24, lane)] = tmp;
    tmp = BFE(src, 25, 1);
    out[INDEX(25, lane)] = tmp;
    tmp = BFE(src, 26, 1);
    out[INDEX(26, lane)] = tmp;
    tmp = BFE(src, 27, 1);
    out[INDEX(27, lane)] = tmp;
    tmp = BFE(src, 28, 1);
    out[INDEX(28, lane)] = tmp;
    tmp = BFE(src, 29, 1);
    out[INDEX(29, lane)] = tmp;
    tmp = BFE(src, 30, 1);
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 31, 1);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_2bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 2);
    out[INDEX(0, lane)] = tmp;
    tmp = BFE(src, 2, 2);
    out[INDEX(1, lane)] = tmp;
    tmp = BFE(src, 4, 2);
    out[INDEX(2, lane)] = tmp;
    tmp = BFE(src, 6, 2);
    out[INDEX(3, lane)] = tmp;
    tmp = BFE(src, 8, 2);
    out[INDEX(4, lane)] = tmp;
    tmp = BFE(src, 10, 2);
    out[INDEX(5, lane)] = tmp;
    tmp = BFE(src, 12, 2);
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 14, 2);
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 16, 2);
    out[INDEX(8, lane)] = tmp;
    tmp = BFE(src, 18, 2);
    out[INDEX(9, lane)] = tmp;
    tmp = BFE(src, 20, 2);
    out[INDEX(10, lane)] = tmp;
    tmp = BFE(src, 22, 2);
    out[INDEX(11, lane)] = tmp;
    tmp = BFE(src, 24, 2);
    out[INDEX(12, lane)] = tmp;
    tmp = BFE(src, 26, 2);
    out[INDEX(13, lane)] = tmp;
    tmp = BFE(src, 28, 2);
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 30, 2);
    src = in[lane + LANE_COUNT * 1];
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 0, 2);
    out[INDEX(16, lane)] = tmp;
    tmp = BFE(src, 2, 2);
    out[INDEX(17, lane)] = tmp;
    tmp = BFE(src, 4, 2);
    out[INDEX(18, lane)] = tmp;
    tmp = BFE(src, 6, 2);
    out[INDEX(19, lane)] = tmp;
    tmp = BFE(src, 8, 2);
    out[INDEX(20, lane)] = tmp;
    tmp = BFE(src, 10, 2);
    out[INDEX(21, lane)] = tmp;
    tmp = BFE(src, 12, 2);
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 14, 2);
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 16, 2);
    out[INDEX(24, lane)] = tmp;
    tmp = BFE(src, 18, 2);
    out[INDEX(25, lane)] = tmp;
    tmp = BFE(src, 20, 2);
    out[INDEX(26, lane)] = tmp;
    tmp = BFE(src, 22, 2);
    out[INDEX(27, lane)] = tmp;
    tmp = BFE(src, 24, 2);
    out[INDEX(28, lane)] = tmp;
    tmp = BFE(src, 26, 2);
    out[INDEX(29, lane)] = tmp;
    tmp = BFE(src, 28, 2);
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 30, 2);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_3bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 3);
    out[INDEX(0, lane)] = tmp;
    tmp = BFE(src, 3, 3);
    out[INDEX(1, lane)] = tmp;
    tmp = BFE(src, 6, 3);
    out[INDEX(2, lane)] = tmp;
    tmp = BFE(src, 9, 3);
    out[INDEX(3, lane)] = tmp;
    tmp = BFE(src, 12, 3);
    out[INDEX(4, lane)] = tmp;
    tmp = BFE(src, 15, 3);
    out[INDEX(5, lane)] = tmp;
    tmp = BFE(src, 18, 3);
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 21, 3);
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 24, 3);
    out[INDEX(8, lane)] = tmp;
    tmp = BFE(src, 27, 3);
    out[INDEX(9, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 3);
        src = next;
    }
    out[INDEX(10, lane)] = tmp;
    tmp = BFE(src, 1, 3);
    out[INDEX(11, lane)] = tmp;
    tmp = BFE(src, 4, 3);
    out[INDEX(12, lane)] = tmp;
    tmp = BFE(src, 7, 3);
    out[INDEX(13, lane)] = tmp;
    tmp = BFE(src, 10, 3);
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 13, 3);
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 16, 3);
    out[INDEX(16, lane)] = tmp;
    tmp = BFE(src, 19, 3);
    out[INDEX(17, lane)] = tmp;
    tmp = BFE(src, 22, 3);
    out[INDEX(18, lane)] = tmp;
    tmp = BFE(src, 25, 3);
    out[INDEX(19, lane)] = tmp;
    tmp = BFE(src, 28, 3);
    out[INDEX(20, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 31) & MASK(uint32_t, 3);
        src = next;
    }
    out[INDEX(21, lane)] = tmp;
    tmp = BFE(src, 2, 3);
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 5, 3);
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 8, 3);
    out[INDEX(24, lane)] = tmp;
    tmp = BFE(src, 11, 3);
    out[INDEX(25, lane)] = tmp;
    tmp = BFE(src, 14, 3);
    out[INDEX(26, lane)] = tmp;
    tmp = BFE(src, 17, 3);
    out[INDEX(27, lane)] = tmp;
    tmp = BFE(src, 20, 3);
    out[INDEX(28, lane)] = tmp;
    tmp = BFE(src, 23, 3);
    out[INDEX(29, lane)] = tmp;
    tmp = BFE(src, 26, 3);
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 29, 3);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_4bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 4);
    out[INDEX(0, lane)] = tmp;
    tmp = BFE(src, 4, 4);
    out[INDEX(1, lane)] = tmp;
    tmp = BFE(src, 8, 4);
    out[INDEX(2, lane)] = tmp;
    tmp = BFE(src, 12, 4);
    out[INDEX(3, lane)] = tmp;
    tmp = BFE(src, 16, 4);
    out[INDEX(4, lane)] = tmp;
    tmp = BFE(src, 20, 4);
    out[INDEX(5, lane)] = tmp;
    tmp = BFE(src, 24, 4);
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 28, 4);
    src = in[lane + LANE_COUNT * 1];
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 0, 4);
    out[INDEX(8, lane)] = tmp;
    tmp = BFE(src, 4, 4);
    out[INDEX(9, lane)] = tmp;
    tmp = BFE(src, 8, 4);
    out[INDEX(10, lane)] = tmp;
    tmp = BFE(src, 12, 4);
    out[INDEX(11, lane)] = tmp;
    tmp = BFE(src, 16, 4);
    out[INDEX(12, lane)] = tmp;
    tmp = BFE(src, 20, 4);
    out[INDEX(13, lane)] = tmp;
    tmp = BFE(src, 24, 4);
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 28, 4);
    src = in[lane + LANE_COUNT * 2];
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 0, 4);
    out[INDEX(16, lane)] = tmp;
    tmp = BFE(src, 4, 4);
    out[INDEX(17, lane)] = tmp;
    tmp = BFE(src, 8, 4);
    out[INDEX(18, lane)] = tmp;
    tmp = BFE(src, 12, 4);
    out[INDEX(19, lane)] = tmp;
    tmp = BFE(src, 16, 4);
    out[INDEX(20, lane)] = tmp;
    tmp = BFE(src, 20, 4);
    out[INDEX(21, lane)] = tmp;
    tmp = BFE(src, 24, 4);
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 28, 4);
    src = in[lane + LANE_COUNT * 3];
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 0, 4);
    out[INDEX(24, lane)] = tmp;
    tmp = BFE(src, 4, 4);
    out[INDEX(25, lane)] = tmp;
    tmp = BFE(src, 8, 4);
    out[INDEX(26, lane)] = tmp;
    tmp = BFE(src, 12, 4);
    out[INDEX(27, lane)] = tmp;
    tmp = BFE(src, 16, 4);
    out[INDEX(28, lane)] = tmp;
    tmp = BFE(src, 20, 4);
    out[INDEX(29, lane)] = tmp;
    tmp = BFE(src, 24, 4);
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 28, 4);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_5bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 5);
    out[INDEX(0, lane)] = tmp;
    tmp = BFE(src, 5, 5);
    out[INDEX(1, lane)] = tmp;
    tmp = BFE(src, 10, 5);
    out[INDEX(2, lane)] = tmp;
    tmp = BFE(src, 15, 5);
    out[INDEX(3, lane)] = tmp;
    tmp = BFE(src, 20, 5);
    out[INDEX(4, lane)] = tmp;
    tmp = BFE(src, 25, 5);
    out[INDEX(5, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 5);
        src = next;
    }
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 3, 5);
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 8, 5);
    out[INDEX(8, lane)] = tmp;
    tmp = BFE(src, 13, 5);
    out[INDEX(9, lane)] = tmp;
    tmp = BFE(src, 18, 5);
    out[INDEX(10, lane)] = tmp;
    tmp = BFE(src, 23, 5);
    out[INDEX(11, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 5);
        src = next;
    }
    out[INDEX(12, lane)] = tmp;
    tmp = BFE(src, 1, 5);
    out[INDEX(13, lane)] = tmp;
    tmp = BFE(src, 6, 5);
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 11, 5);
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 16, 5);
    out[INDEX(16, lane)] = tmp;
    tmp = BFE(src, 21, 5);
    out[INDEX(17, lane)] = tmp;
    tmp = BFE(src, 26, 5);
    out[INDEX(18, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 31) & MASK(uint32_t, 5);
        src = next;
    }
    out[INDEX(19, lane)] = tmp;
    tmp = BFE(src, 4, 5);
    out[INDEX(20, lane)] = tmp;
    tmp = BFE(src, 9, 5);
    out[INDEX(21, lane)] = tmp;
    tmp = BFE(src, 14, 5);
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 19, 5);
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 24, 5);
    out[INDEX(24, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 29) & MASK(uint32_t, 5);
        src = next;
    }
    out[INDEX(25, lane)] = tmp;
    tmp = BFE(src, 2, 5);
    out[INDEX(26, lane)] = tmp;
    tmp = BFE(src, 7, 5);
    out[INDEX(27, lane)] = tmp;
    tmp = BFE(src, 12, 5);
    out[INDEX(28, lane)] = tmp;
    tmp = BFE(src, 17, 5);
    out[INDEX(29, lane)] = tmp;
    tmp = BFE(src, 22, 5);
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 27, 5);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_6bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 6);
    out[INDEX(0, lane)] = tmp;
    tmp = BFE(src, 6, 6);
    out[INDEX(1, lane)] = tmp;
    tmp = BFE(src, 12, 6);
    out[INDEX(2, lane)] = tmp;
    tmp = BFE(src, 18, 6);
    out[INDEX(3, lane)] = tmp;
    tmp = BFE(src, 24, 6);
    out[INDEX(4, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 6);
        src = next;
    }
    out[INDEX(5, lane)] = tmp;
    tmp = BFE(src, 4, 6);
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 10, 6);
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 16, 6);
    out[INDEX(8, lane)] = tmp;
    tmp = BFE(src, 22, 6);
    out[INDEX(9, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 6);
        src = next;
    }
    out[INDEX(10, lane)] = tmp;
    tmp = BFE(src, 2, 6);
    out[INDEX(11, lane)] = tmp;
    tmp = BFE(src, 8, 6);
    out[INDEX(12, lane)] = tmp;
    tmp = BFE(src, 14, 6);
    out[INDEX(13, lane)] = tmp;
    tmp = BFE(src, 20, 6);
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 26, 6);
    src = in[lane + LANE_COUNT * 3];
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 0, 6);
    out[INDEX(16, lane)] = tmp;
    tmp = BFE(src, 6, 6);
    out[INDEX(17, lane)] = tmp;
    tmp = BFE(src, 12, 6);
    out[INDEX(18, lane)] = tmp;
    tmp = BFE(src, 18, 6);
    out[INDEX(19, lane)] = tmp;
    tmp = BFE(src, 24, 6);
    out[INDEX(20, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 6);
        src = next;
    }
    out[INDEX(21, lane)] = tmp;
    tmp = BFE(src, 4, 6);
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 10, 6);
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 16, 6);
    out[INDEX(24, lane)] = tmp;
    tmp = BFE(src, 22, 6);
    out[INDEX(25, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 6);
        src = next;
    }
    out[INDEX(26, lane)] = tmp;
    tmp = BFE(src, 2, 6);
    out[INDEX(27, lane)] = tmp;
    tmp = BFE(src, 8, 6);
    out[INDEX(28, lane)] = tmp;
    tmp = BFE(src, 14, 6);
    out[INDEX(29, lane)] = tmp;
    tmp = BFE(src, 20, 6);
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 26, 6);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_7bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 7);
    out[INDEX(0, lane)] = tmp;
    tmp = BFE(src, 7, 7);
    out[INDEX(1, lane)] = tmp;
    tmp = BFE(src, 14, 7);
    out[INDEX(2, lane)] = tmp;
    tmp = BFE(src, 21, 7);
    out[INDEX(3, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 7);
        src = next;
    }
    out[INDEX(4, lane)] = tmp;
    tmp = BFE(src, 3, 7);
    out[INDEX(5, lane)] = tmp;
    tmp = BFE(src, 10, 7);
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 17, 7);
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 24, 7);
    out[INDEX(8, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 31) & MASK(uint32_t, 7);
        src = next;
    }
    out[INDEX(9, lane)] = tmp;
    tmp = BFE(src, 6, 7);
    out[INDEX(10, lane)] = tmp;
    tmp = BFE(src, 13, 7);
    out[INDEX(11, lane)] = tmp;
    tmp = BFE(src, 20, 7);
    out[INDEX(12, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 27) & MASK(uint32_t, 7);
        src = next;
    }
    out[INDEX(13, lane)] = tmp;
    tmp = BFE(src, 2, 7);
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 9, 7);
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 16, 7);
    out[INDEX(16, lane)] = tmp;
    tmp = BFE(src, 23, 7);
    out[INDEX(17, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 7);
        src = next;
    }
    out[INDEX(18, lane)] = tmp;
    tmp = BFE(src, 5, 7);
    out[INDEX(19, lane)] = tmp;
    tmp = BFE(src, 12, 7);
    out[INDEX(20, lane)] = tmp;
    tmp = BFE(src, 19, 7);
    out[INDEX(21, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 7);
        src = next;
    }
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 1, 7);
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 8, 7);
    out[INDEX(24, lane)] = tmp;
    tmp = BFE(src, 15, 7);
    out[INDEX(25, lane)] = tmp;
    tmp = BFE(src, 22, 7);
    out[INDEX(26, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 29) & MASK(uint32_t, 7);
        src = next;
    }
    out[INDEX(27, lane)] = tmp;
    tmp = BFE(src, 4, 7);
    out[INDEX(28, lane)] = tmp;
    tmp = BFE(src, 11, 7);
    out[INDEX(29, lane)] = tmp;
    tmp = BFE(src, 18, 7);
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 25, 7);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_8bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 8);
    out[INDEX(0, lane)] = tmp;
    tmp = BFE(src, 8, 8);
    out[INDEX(1, lane)] = tmp;
    tmp = BFE(src, 16, 8);
    out[INDEX(2, lane)] = tmp;
    tmp = BFE(src, 24, 8);
    src = in[lane + LANE_COUNT * 1];
    out[INDEX(3, lane)] = tmp;
    tmp = BFE(src, 0, 8);
    out[INDEX(4, lane)] = tmp;
    tmp = BFE(src, 8, 8);
    out[INDEX(5, lane)] = tmp;
    tmp = BFE(src, 16, 8);
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 24, 8);
    src = in[lane + LANE_COUNT * 2];
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 0, 8);
    out[INDEX(8, lane)] = tmp;
    tmp = BFE(src, 8, 8);
    out[INDEX(9, lane)] = tmp;
    tmp = BFE(src, 16, 8);
    out[INDEX(10, lane)] = tmp;
    tmp = BFE(src, 24, 8);
    src = in[lane + LANE_COUNT * 3];
    out[INDEX(11, lane)] = tmp;
    tmp = BFE(src, 0, 8);
    out[INDEX(12, lane)] = tmp;
    tmp = BFE(src, 8, 8);
    out[INDEX(13, lane)] = tmp;
    tmp = BFE(src, 16, 8);
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 24, 8);
    src = in[lane + LANE_COUNT * 4];
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 0, 8);
    out[INDEX(16, lane)] = tmp;
    tmp = BFE(src, 8, 8);
    out[INDEX(17, lane)] = tmp;
    tmp = BFE(src, 16, 8);
    out[INDEX(18, lane)] = tmp;
    tmp = BFE(src, 24, 8);
    src = in[lane + LANE_COUNT * 5];
    out[INDEX(19, lane)] = tmp;
    tmp = BFE(src, 0, 8);
    out[INDEX(20, lane)] = tmp;
    tmp = BFE(src, 8, 8);
    out[INDEX(21, lane)] = tmp;
    tmp = BFE(src, 16, 8);
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 24, 8);
    src = in[lane + LANE_COUNT * 6];
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 0, 8);
    out[INDEX(24, lane)] = tmp;
    tmp = BFE(src, 8, 8);
    out[INDEX(25, lane)] = tmp;
    tmp = BFE(src, 16, 8);
    out[INDEX(26, lane)] = tmp;
    tmp = BFE(src, 24, 8);
    src = in[lane + LANE_COUNT * 7];
    out[INDEX(27, lane)] = tmp;
    tmp = BFE(src, 0, 8);
    out[INDEX(28, lane)] = tmp;
    tmp = BFE(src, 8, 8);
    out[INDEX(29, lane)] = tmp;
    tmp = BFE(src, 16, 8);
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 24, 8);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_9bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 9);
    out[INDEX(0, lane)] = tmp;
    tmp = BFE(src, 9, 9);
    out[INDEX(1, lane)] = tmp;
    tmp = BFE(src, 18, 9);
    out[INDEX(2, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 27) & MASK(uint32_t, 9);
        src = next;
    }
    out[INDEX(3, lane)] = tmp;
    tmp = BFE(src, 4, 9);
    out[INDEX(4, lane)] = tmp;
    tmp = BFE(src, 13, 9);
    out[INDEX(5, lane)] = tmp;
    tmp = BFE(src, 22, 9);
    out[INDEX(6, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 31) & MASK(uint32_t, 9);
        src = next;
    }
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 8, 9);
    out[INDEX(8, lane)] = tmp;
    tmp = BFE(src, 17, 9);
    out[INDEX(9, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 9);
        src = next;
    }
    out[INDEX(10, lane)] = tmp;
    tmp = BFE(src, 3, 9);
    out[INDEX(11, lane)] = tmp;
    tmp = BFE(src, 12, 9);
    out[INDEX(12, lane)] = tmp;
    tmp = BFE(src, 21, 9);
    out[INDEX(13, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 9);
        src = next;
    }
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 7, 9);
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 16, 9);
    out[INDEX(16, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 25) & MASK(uint32_t, 9);
        src = next;
    }
    out[INDEX(17, lane)] = tmp;
    tmp = BFE(src, 2, 9);
    out[INDEX(18, lane)] = tmp;
    tmp = BFE(src, 11, 9);
    out[INDEX(19, lane)] = tmp;
    tmp = BFE(src, 20, 9);
    out[INDEX(20, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 29) & MASK(uint32_t, 9);
        src = next;
    }
    out[INDEX(21, lane)] = tmp;
    tmp = BFE(src, 6, 9);
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 15, 9);
    out[INDEX(23, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 9);
        src = next;
    }
    out[INDEX(24, lane)] = tmp;
    tmp = BFE(src, 1, 9);
    out[INDEX(25, lane)] = tmp;
    tmp = BFE(src, 10, 9);
    out[INDEX(26, lane)] = tmp;
    tmp = BFE(src, 19, 9);
    out[INDEX(27, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 9);
        src = next;
    }
    out[INDEX(28, lane)] = tmp;
    tmp = BFE(src, 5, 9);
    out[INDEX(29, lane)] = tmp;
    tmp = BFE(src, 14, 9);
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 23, 9);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_10bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 10);
    out[INDEX(0, lane)] = tmp;
    tmp = BFE(src, 10, 10);
    out[INDEX(1, lane)] = tmp;
    tmp = BFE(src, 20, 10);
    out[INDEX(2, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 10);
        src = next;
    }
    out[INDEX(3, lane)] = tmp;
    tmp = BFE(src, 8, 10);
    out[INDEX(4, lane)] = tmp;
    tmp = BFE(src, 18, 10);
    out[INDEX(5, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 10);
        src = next;
    }
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 6, 10);
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 16, 10);
    out[INDEX(8, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 10);
        src = next;
    }
    out[INDEX(9, lane)] = tmp;
    tmp = BFE(src, 4, 10);
    out[INDEX(10, lane)] = tmp;
    tmp = BFE(src, 14, 10);
    out[INDEX(11, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 10);
        src = next;
    }
    out[INDEX(12, lane)] = tmp;
    tmp = BFE(src, 2, 10);
    out[INDEX(13, lane)] = tmp;
    tmp = BFE(src, 12, 10);
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 22, 10);
    src = in[lane + LANE_COUNT * 5];
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 0, 10);
    out[INDEX(16, lane)] = tmp;
    tmp = BFE(src, 10, 10);
    out[INDEX(17, lane)] = tmp;
    tmp = BFE(src, 20, 10);
    out[INDEX(18, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 10);
        src = next;
    }
    out[INDEX(19, lane)] = tmp;
    tmp = BFE(src, 8, 10);
    out[INDEX(20, lane)] = tmp;
    tmp = BFE(src, 18, 10);
    out[INDEX(21, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 10);
        src = next;
    }
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 6, 10);
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 16, 10);
    out[INDEX(24, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 10);
        src = next;
    }
    out[INDEX(25, lane)] = tmp;
    tmp = BFE(src, 4, 10);
    out[INDEX(26, lane)] = tmp;
    tmp = BFE(src, 14, 10);
    out[INDEX(27, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 10);
        src = next;
    }
    out[INDEX(28, lane)] = tmp;
    tmp = BFE(src, 2, 10);
    out[INDEX(29, lane)] = tmp;
    tmp = BFE(src, 12, 10);
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 22, 10);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_11bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 11);
    out[INDEX(0, lane)] = tmp;
    tmp = BFE(src, 11, 11);
    out[INDEX(1, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 11);
        src = next;
    }
    out[INDEX(2, lane)] = tmp;
    tmp = BFE(src, 1, 11);
    out[INDEX(3, lane)] = tmp;
    tmp = BFE(src, 12, 11);
    out[INDEX(4, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 23) & MASK(uint32_t, 11);
        src = next;
    }
    out[INDEX(5, lane)] = tmp;
    tmp = BFE(src, 2, 11);
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 13, 11);
    out[INDEX(7, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 11);
        src = next;
    }
    out[INDEX(8, lane)] = tmp;
    tmp = BFE(src, 3, 11);
    out[INDEX(9, lane)] = tmp;
    tmp = BFE(src, 14, 11);
    out[INDEX(10, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 25) & MASK(uint32_t, 11);
        src = next;
    }
    out[INDEX(11, lane)] = tmp;
    tmp = BFE(src, 4, 11);
    out[INDEX(12, lane)] = tmp;
    tmp = BFE(src, 15, 11);
    out[INDEX(13, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 11);
        src = next;
    }
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 5, 11);
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 16, 11);
    out[INDEX(16, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 27) & MASK(uint32_t, 11);
        src = next;
    }
    out[INDEX(17, lane)] = tmp;
    tmp = BFE(src, 6, 11);
    out[INDEX(18, lane)] = tmp;
    tmp = BFE(src, 17, 11);
    out[INDEX(19, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 11);
        src = next;
    }
    out[INDEX(20, lane)] = tmp;
    tmp = BFE(src, 7, 11);
    out[INDEX(21, lane)] = tmp;
    tmp = BFE(src, 18, 11);
    out[INDEX(22, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 29) & MASK(uint32_t, 11);
        src = next;
    }
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 8, 11);
    out[INDEX(24, lane)] = tmp;
    tmp = BFE(src, 19, 11);
    out[INDEX(25, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 11);
        src = next;
    }
    out[INDEX(26, lane)] = tmp;
    tmp = BFE(src, 9, 11);
    out[INDEX(27, lane)] = tmp;
    tmp = BFE(src, 20, 11);
    out[INDEX(28, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 31) & MASK(uint32_t, 11);
        src = next;
    }
    out[INDEX(29, lane)] = tmp;
    tmp = BFE(src, 10, 11);
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 21, 11);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_12bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 12);
    out[INDEX(0, lane)] = tmp;
    tmp = BFE(src, 12, 12);
    out[INDEX(1, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 12);
        src = next;
    }
    out[INDEX(2, lane)] = tmp;
    tmp = BFE(src, 4, 12);
    out[INDEX(3, lane)] = tmp;
    tmp = BFE(src, 16, 12);
    out[INDEX(4, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 12);
        src = next;
    }
    out[INDEX(5, lane)] = tmp;
    tmp = BFE(src, 8, 12);
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 20, 12);
    src = in[lane + LANE_COUNT * 3];
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 0, 12);
    out[INDEX(8, lane)] = tmp;
    tmp = BFE(src, 12, 12);
    out[INDEX(9, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 12);
        src = next;
    }
    out[INDEX(10, lane)] = tmp;
    tmp = BFE(src, 4, 12);
    out[INDEX(11, lane)] = tmp;
    tmp = BFE(src, 16, 12);
    out[INDEX(12, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 12);
        src = next;
    }
    out[INDEX(13, lane)] = tmp;
    tmp = BFE(src, 8, 12);
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 20, 12);
    src = in[lane + LANE_COUNT * 6];
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 0, 12);
    out[INDEX(16, lane)] = tmp;
    tmp = BFE(src, 12, 12);
    out[INDEX(17, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 12);
        src = next;
    }
    out[INDEX(18, lane)] = tmp;
    tmp = BFE(src, 4, 12);
    out[INDEX(19, lane)] = tmp;
    tmp = BFE(src, 16, 12);
    out[INDEX(20, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 12);
        src = next;
    }
    out[INDEX(21, lane)] = tmp;
    tmp = BFE(src, 8, 12);
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 20, 12);
    src = in[lane + LANE_COUNT * 9];
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 0, 12);
    out[INDEX(24, lane)] = tmp;
    tmp = BFE(src, 12, 12);
    out[INDEX(25, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 12);
        src = next;
    }
    out[INDEX(26, lane)] = tmp;
    tmp = BFE(src, 4, 12);
    out[INDEX(27, lane)] = tmp;
    tmp = BFE(src, 16, 12);
    out[INDEX(28, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 12);
        src = next;
    }
    out[INDEX(29, lane)] = tmp;
    tmp = BFE(src, 8, 12);
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 20, 12);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_13bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 13);
    out[INDEX(0, lane)] = tmp;
    tmp = BFE(src, 13, 13);
    out[INDEX(1, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 13);
        src = next;
    }
    out[INDEX(2, lane)] = tmp;
    tmp = BFE(src, 7, 13);
    out[INDEX(3, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 13);
        src = next;
    }
    out[INDEX(4, lane)] = tmp;
    tmp = BFE(src, 1, 13);
    out[INDEX(5, lane)] = tmp;
    tmp = BFE(src, 14, 13);
    out[INDEX(6, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 27) & MASK(uint32_t, 13);
        src = next;
    }
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 8, 13);
    out[INDEX(8, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 21) & MASK(uint32_t, 13);
        src = next;
    }
    out[INDEX(9, lane)] = tmp;
    tmp = BFE(src, 2, 13);
    out[INDEX(10, lane)] = tmp;
    tmp = BFE(src, 15, 13);
    out[INDEX(11, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 13);
        src = next;
    }
    out[INDEX(12, lane)] = tmp;
    tmp = BFE(src, 9, 13);
    out[INDEX(13, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 13);
        src = next;
    }
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 3, 13);
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 16, 13);
    out[INDEX(16, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 29) & MASK(uint32_t, 13);
        src = next;
    }
    out[INDEX(17, lane)] = tmp;
    tmp = BFE(src, 10, 13);
    out[INDEX(18, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 23) & MASK(uint32_t, 13);
        src = next;
    }
    out[INDEX(19, lane)] = tmp;
    tmp = BFE(src, 4, 13);
    out[INDEX(20, lane)] = tmp;
    tmp = BFE(src, 17, 13);
    out[INDEX(21, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 13);
        src = next;
    }
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 11, 13);
    out[INDEX(23, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 13);
        src = next;
    }
    out[INDEX(24, lane)] = tmp;
    tmp = BFE(src, 5, 13);
    out[INDEX(25, lane)] = tmp;
    tmp = BFE(src, 18, 13);
    out[INDEX(26, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 31) & MASK(uint32_t, 13);
        src = next;
    }
    out[INDEX(27, lane)] = tmp;
    tmp = BFE(src, 12, 13);
    out[INDEX(28, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 12];
        tmp = FUNNEL_SHIFT_R(src, next, 25) & MASK(uint32_t, 13);
        src = next;
    }
    out[INDEX(29, lane)] = tmp;
    tmp = BFE(src, 6, 13);
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 19, 13);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_14bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 14);
    out[INDEX(0, lane)] = tmp;
    tmp = BFE(src, 14, 14);
    out[INDEX(1, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 14);
        src = next;
    }
    out[INDEX(2, lane)] = tmp;
    tmp = BFE(src, 10, 14);
    out[INDEX(3, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 14);
        src = next;
    }
    out[INDEX(4, lane)] = tmp;
    tmp = BFE(src, 6, 14);
    out[INDEX(5, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 14);
        src = next;
    }
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 2, 14);
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 16, 14);
    out[INDEX(8, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 14);
        src = next;
    }
    out[INDEX(9, lane)] = tmp;
    tmp = BFE(src, 12, 14);
    out[INDEX(10, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 14);
        src = next;
    }
    out[INDEX(11, lane)] = tmp;
    tmp = BFE(src, 8, 14);
    out[INDEX(12, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 14);
        src = next;
    }
    out[INDEX(13, lane)] = tmp;
    tmp = BFE(src, 4, 14);
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 18, 14);
    src = in[lane + LANE_COUNT * 7];
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 0, 14);
    out[INDEX(16, lane)] = tmp;
    tmp = BFE(src, 14, 14);
    out[INDEX(17, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 14);
        src = next;
    }
    out[INDEX(18, lane)] = tmp;
    tmp = BFE(src, 10, 14);
    out[INDEX(19, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 14);
        src = next;
    }
    out[INDEX(20, lane)] = tmp;
    tmp = BFE(src, 6, 14);
    out[INDEX(21, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 14);
        src = next;
    }
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 2, 14);
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 16, 14);
    out[INDEX(24, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 14);
        src = next;
    }
    out[INDEX(25, lane)] = tmp;
    tmp = BFE(src, 12, 14);
    out[INDEX(26, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 12];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 14);
        src = next;
    }
    out[INDEX(27, lane)] = tmp;
    tmp = BFE(src, 8, 14);
    out[INDEX(28, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 13];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 14);
        src = next;
    }
    out[INDEX(29, lane)] = tmp;
    tmp = BFE(src, 4, 14);
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 18, 14);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_15bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 15);
    out[INDEX(0, lane)] = tmp;
    tmp = BFE(src, 15, 15);
    out[INDEX(1, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 15);
        src = next;
    }
    out[INDEX(2, lane)] = tmp;
    tmp = BFE(src, 13, 15);
    out[INDEX(3, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 15);
        src = next;
    }
    out[INDEX(4, lane)] = tmp;
    tmp = BFE(src, 11, 15);
    out[INDEX(5, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 15);
        src = next;
    }
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 9, 15);
    out[INDEX(7, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 15);
        src = next;
    }
    out[INDEX(8, lane)] = tmp;
    tmp = BFE(src, 7, 15);
    out[INDEX(9, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 15);
        src = next;
    }
    out[INDEX(10, lane)] = tmp;
    tmp = BFE(src, 5, 15);
    out[INDEX(11, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 15);
        src = next;
    }
    out[INDEX(12, lane)] = tmp;
    tmp = BFE(src, 3, 15);
    out[INDEX(13, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 18) & MASK(uint32_t, 15);
        src = next;
    }
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 1, 15);
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 16, 15);
    out[INDEX(16, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 31) & MASK(uint32_t, 15);
        src = next;
    }
    out[INDEX(17, lane)] = tmp;
    tmp = BFE(src, 14, 15);
    out[INDEX(18, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 29) & MASK(uint32_t, 15);
        src = next;
    }
    out[INDEX(19, lane)] = tmp;
    tmp = BFE(src, 12, 15);
    out[INDEX(20, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 27) & MASK(uint32_t, 15);
        src = next;
    }
    out[INDEX(21, lane)] = tmp;
    tmp = BFE(src, 10, 15);
    out[INDEX(22, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 25) & MASK(uint32_t, 15);
        src = next;
    }
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 8, 15);
    out[INDEX(24, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 12];
        tmp = FUNNEL_SHIFT_R(src, next, 23) & MASK(uint32_t, 15);
        src = next;
    }
    out[INDEX(25, lane)] = tmp;
    tmp = BFE(src, 6, 15);
    out[INDEX(26, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 13];
        tmp = FUNNEL_SHIFT_R(src, next, 21) & MASK(uint32_t, 15);
        src = next;
    }
    out[INDEX(27, lane)] = tmp;
    tmp = BFE(src, 4, 15);
    out[INDEX(28, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 14];
        tmp = FUNNEL_SHIFT_R(src, next, 19) & MASK(uint32_t, 15);
        src = next;
    }
    out[INDEX(29, lane)] = tmp;
    tmp = BFE(src, 2, 15);
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 17, 15);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_16bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 16);
    out[INDEX(0, lane)] = tmp;
    tmp = BFE(src, 16, 16);
    src = in[lane + LANE_COUNT * 1];
    out[INDEX(1, lane)] = tmp;
    tmp = BFE(src, 0, 16);
    out[INDEX(2, lane)] = tmp;
    tmp = BFE(src, 16, 16);
    src = in[lane + LANE_COUNT * 2];
    out[INDEX(3, lane)] = tmp;
    tmp = BFE(src, 0, 16);
    out[INDEX(4, lane)] = tmp;
    tmp = BFE(src, 16, 16);
    src = in[lane + LANE_COUNT * 3];
    out[INDEX(5, lane)] = tmp;
    tmp = BFE(src, 0, 16);
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 16, 16);
    src = in[lane + LANE_COUNT * 4];
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 0, 16);
    out[INDEX(8, lane)] = tmp;
    tmp = BFE(src, 16, 16);
    src = in[lane + LANE_COUNT * 5];
    out[INDEX(9, lane)] = tmp;
    tmp = BFE(src, 0, 16);
    out[INDEX(10, lane)] = tmp;
    tmp = BFE(src, 16, 16);
    src = in[lane + LANE_COUNT * 6];
    out[INDEX(11, lane)] = tmp;
    tmp = BFE(src, 0, 16);
    out[INDEX(12, lane)] = tmp;
    tmp = BFE(src, 16, 16);
    src = in[lane + LANE_COUNT * 7];
    out[INDEX(13, lane)] = tmp;
    tmp = BFE(src, 0, 16);
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 16, 16);
    src = in[lane + LANE_COUNT * 8];
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 0, 16);
    out[INDEX(16, lane)] = tmp;
    tmp = BFE(src, 16, 16);
    src = in[lane + LANE_COUNT * 9];
    out[INDEX(17, lane)] = tmp;
    tmp = BFE(src, 0, 16);
    out[INDEX(18, lane)] = tmp;
    tmp = BFE(src, 16, 16);
    src = in[lane + LANE_COUNT * 10];
    out[INDEX(19, lane)] = tmp;
    tmp = BFE(src, 0, 16);
    out[INDEX(20, lane)] = tmp;
    tmp = BFE(src, 16, 16);
    src = in[lane + LANE_COUNT * 11];
    out[INDEX(21, lane)] = tmp;
    tmp = BFE(src, 0, 16);
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 16, 16);
    src = in[lane + LANE_COUNT * 12];
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 0, 16);
    out[INDEX(24, lane)] = tmp;
    tmp = BFE(src, 16, 16);
    src = in[lane + LANE_COUNT * 13];
    out[INDEX(25, lane)] = tmp;
    tmp = BFE(src, 0, 16);
    out[INDEX(26, lane)] = tmp;
    tmp = BFE(src, 16, 16);
    src = in[lane + LANE_COUNT * 14];
    out[INDEX(27, lane)] = tmp;
    tmp = BFE(src, 0, 16);
    out[INDEX(28, lane)] = tmp;
    tmp = BFE(src, 16, 16);
    src = in[lane + LANE_COUNT * 15];
    out[INDEX(29, lane)] = tmp;
    tmp = BFE(src, 0, 16);
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 16, 16);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_17bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 17);
    out[INDEX(0, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 17) & MASK(uint32_t, 17);
        src = next;
    }
    out[INDEX(1, lane)] = tmp;
    tmp = BFE(src, 2, 17);
    out[INDEX(2, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 19) & MASK(uint32_t, 17);
        src = next;
    }
    out[INDEX(3, lane)] = tmp;
    tmp = BFE(src, 4, 17);
    out[INDEX(4, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 21) & MASK(uint32_t, 17);
        src = next;
    }
    out[INDEX(5, lane)] = tmp;
    tmp = BFE(src, 6, 17);
    out[INDEX(6, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 23) & MASK(uint32_t, 17);
        src = next;
    }
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 8, 17);
    out[INDEX(8, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 25) & MASK(uint32_t, 17);
        src = next;
    }
    out[INDEX(9, lane)] = tmp;
    tmp = BFE(src, 10, 17);
    out[INDEX(10, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 27) & MASK(uint32_t, 17);
        src = next;
    }
    out[INDEX(11, lane)] = tmp;
    tmp = BFE(src, 12, 17);
    out[INDEX(12, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 29) & MASK(uint32_t, 17);
        src = next;
    }
    out[INDEX(13, lane)] = tmp;
    tmp = BFE(src, 14, 17);
    out[INDEX(14, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 31) & MASK(uint32_t, 17);
        src = next;
    }
    out[INDEX(15, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 17);
        src = next;
    }
    out[INDEX(16, lane)] = tmp;
    tmp = BFE(src, 1, 17);
    out[INDEX(17, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 18) & MASK(uint32_t, 17);
        src = next;
    }
    out[INDEX(18, lane)] = tmp;
    tmp = BFE(src, 3, 17);
    out[INDEX(19, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 17);
        src = next;
    }
    out[INDEX(20, lane)] = tmp;
    tmp = BFE(src, 5, 17);
    out[INDEX(21, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 12];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 17);
        src = next;
    }
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 7, 17);
    out[INDEX(23, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 13];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 17);
        src = next;
    }
    out[INDEX(24, lane)] = tmp;
    tmp = BFE(src, 9, 17);
    out[INDEX(25, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 14];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 17);
        src = next;
    }
    out[INDEX(26, lane)] = tmp;
    tmp = BFE(src, 11, 17);
    out[INDEX(27, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 15];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 17);
        src = next;
    }
    out[INDEX(28, lane)] = tmp;
    tmp = BFE(src, 13, 17);
    out[INDEX(29, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 16];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 17);
        src = next;
    }
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 15, 17);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_18bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 18);
    out[INDEX(0, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 18) & MASK(uint32_t, 18);
        src = next;
    }
    out[INDEX(1, lane)] = tmp;
    tmp = BFE(src, 4, 18);
    out[INDEX(2, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 18);
        src = next;
    }
    out[INDEX(3, lane)] = tmp;
    tmp = BFE(src, 8, 18);
    out[INDEX(4, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 18);
        src = next;
    }
    out[INDEX(5, lane)] = tmp;
    tmp = BFE(src, 12, 18);
    out[INDEX(6, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 18);
        src = next;
    }
    out[INDEX(7, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 18);
        src = next;
    }
    out[INDEX(8, lane)] = tmp;
    tmp = BFE(src, 2, 18);
    out[INDEX(9, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 18);
        src = next;
    }
    out[INDEX(10, lane)] = tmp;
    tmp = BFE(src, 6, 18);
    out[INDEX(11, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 18);
        src = next;
    }
    out[INDEX(12, lane)] = tmp;
    tmp = BFE(src, 10, 18);
    out[INDEX(13, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 18);
        src = next;
    }
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 14, 18);
    src = in[lane + LANE_COUNT * 9];
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 0, 18);
    out[INDEX(16, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 18) & MASK(uint32_t, 18);
        src = next;
    }
    out[INDEX(17, lane)] = tmp;
    tmp = BFE(src, 4, 18);
    out[INDEX(18, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 18);
        src = next;
    }
    out[INDEX(19, lane)] = tmp;
    tmp = BFE(src, 8, 18);
    out[INDEX(20, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 12];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 18);
        src = next;
    }
    out[INDEX(21, lane)] = tmp;
    tmp = BFE(src, 12, 18);
    out[INDEX(22, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 13];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 18);
        src = next;
    }
    out[INDEX(23, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 14];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 18);
        src = next;
    }
    out[INDEX(24, lane)] = tmp;
    tmp = BFE(src, 2, 18);
    out[INDEX(25, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 15];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 18);
        src = next;
    }
    out[INDEX(26, lane)] = tmp;
    tmp = BFE(src, 6, 18);
    out[INDEX(27, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 16];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 18);
        src = next;
    }
    out[INDEX(28, lane)] = tmp;
    tmp = BFE(src, 10, 18);
    out[INDEX(29, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 17];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 18);
        src = next;
    }
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 14, 18);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_19bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 19);
    out[INDEX(0, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 19) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(1, lane)] = tmp;
    tmp = BFE(src, 6, 19);
    out[INDEX(2, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 25) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(3, lane)] = tmp;
    tmp = BFE(src, 12, 19);
    out[INDEX(4, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 31) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(5, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 18) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 5, 19);
    out[INDEX(7, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(8, lane)] = tmp;
    tmp = BFE(src, 11, 19);
    out[INDEX(9, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(10, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 17) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(11, lane)] = tmp;
    tmp = BFE(src, 4, 19);
    out[INDEX(12, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 23) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(13, lane)] = tmp;
    tmp = BFE(src, 10, 19);
    out[INDEX(14, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 29) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(15, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(16, lane)] = tmp;
    tmp = BFE(src, 3, 19);
    out[INDEX(17, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(18, lane)] = tmp;
    tmp = BFE(src, 9, 19);
    out[INDEX(19, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 12];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(20, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 13];
        tmp = FUNNEL_SHIFT_R(src, next, 15) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(21, lane)] = tmp;
    tmp = BFE(src, 2, 19);
    out[INDEX(22, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 14];
        tmp = FUNNEL_SHIFT_R(src, next, 21) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 8, 19);
    out[INDEX(24, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 15];
        tmp = FUNNEL_SHIFT_R(src, next, 27) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(25, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 16];
        tmp = FUNNEL_SHIFT_R(src, next, 14) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(26, lane)] = tmp;
    tmp = BFE(src, 1, 19);
    out[INDEX(27, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 17];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(28, lane)] = tmp;
    tmp = BFE(src, 7, 19);
    out[INDEX(29, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 18];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 19);
        src = next;
    }
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 13, 19);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_20bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 20);
    out[INDEX(0, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 20);
        src = next;
    }
    out[INDEX(1, lane)] = tmp;
    tmp = BFE(src, 8, 20);
    out[INDEX(2, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 20);
        src = next;
    }
    out[INDEX(3, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 20);
        src = next;
    }
    out[INDEX(4, lane)] = tmp;
    tmp = BFE(src, 4, 20);
    out[INDEX(5, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 20);
        src = next;
    }
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 12, 20);
    src = in[lane + LANE_COUNT * 5];
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 0, 20);
    out[INDEX(8, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 20);
        src = next;
    }
    out[INDEX(9, lane)] = tmp;
    tmp = BFE(src, 8, 20);
    out[INDEX(10, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 20);
        src = next;
    }
    out[INDEX(11, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 20);
        src = next;
    }
    out[INDEX(12, lane)] = tmp;
    tmp = BFE(src, 4, 20);
    out[INDEX(13, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 20);
        src = next;
    }
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 12, 20);
    src = in[lane + LANE_COUNT * 10];
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 0, 20);
    out[INDEX(16, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 20);
        src = next;
    }
    out[INDEX(17, lane)] = tmp;
    tmp = BFE(src, 8, 20);
    out[INDEX(18, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 12];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 20);
        src = next;
    }
    out[INDEX(19, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 13];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 20);
        src = next;
    }
    out[INDEX(20, lane)] = tmp;
    tmp = BFE(src, 4, 20);
    out[INDEX(21, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 14];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 20);
        src = next;
    }
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 12, 20);
    src = in[lane + LANE_COUNT * 15];
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 0, 20);
    out[INDEX(24, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 16];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 20);
        src = next;
    }
    out[INDEX(25, lane)] = tmp;
    tmp = BFE(src, 8, 20);
    out[INDEX(26, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 17];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 20);
        src = next;
    }
    out[INDEX(27, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 18];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 20);
        src = next;
    }
    out[INDEX(28, lane)] = tmp;
    tmp = BFE(src, 4, 20);
    out[INDEX(29, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 19];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 20);
        src = next;
    }
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 12, 20);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_21bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 21);
    out[INDEX(0, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 21) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(1, lane)] = tmp;
    tmp = BFE(src, 10, 21);
    out[INDEX(2, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 31) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(3, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(4, lane)] = tmp;
    tmp = BFE(src, 9, 21);
    out[INDEX(5, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(6, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 19) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 8, 21);
    out[INDEX(8, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 29) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(9, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 18) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(10, lane)] = tmp;
    tmp = BFE(src, 7, 21);
    out[INDEX(11, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(12, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 17) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(13, lane)] = tmp;
    tmp = BFE(src, 6, 21);
    out[INDEX(14, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 27) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(15, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(16, lane)] = tmp;
    tmp = BFE(src, 5, 21);
    out[INDEX(17, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 12];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(18, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 13];
        tmp = FUNNEL_SHIFT_R(src, next, 15) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(19, lane)] = tmp;
    tmp = BFE(src, 4, 21);
    out[INDEX(20, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 14];
        tmp = FUNNEL_SHIFT_R(src, next, 25) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(21, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 15];
        tmp = FUNNEL_SHIFT_R(src, next, 14) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 3, 21);
    out[INDEX(23, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 16];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(24, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 17];
        tmp = FUNNEL_SHIFT_R(src, next, 13) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(25, lane)] = tmp;
    tmp = BFE(src, 2, 21);
    out[INDEX(26, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 18];
        tmp = FUNNEL_SHIFT_R(src, next, 23) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(27, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 19];
        tmp = FUNNEL_SHIFT_R(src, next, 12) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(28, lane)] = tmp;
    tmp = BFE(src, 1, 21);
    out[INDEX(29, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 20];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 21);
        src = next;
    }
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 11, 21);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_22bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 22);
    out[INDEX(0, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(1, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 12) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(2, lane)] = tmp;
    tmp = BFE(src, 2, 22);
    out[INDEX(3, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(4, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 14) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(5, lane)] = tmp;
    tmp = BFE(src, 4, 22);
    out[INDEX(6, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(7, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(8, lane)] = tmp;
    tmp = BFE(src, 6, 22);
    out[INDEX(9, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(10, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 18) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(11, lane)] = tmp;
    tmp = BFE(src, 8, 22);
    out[INDEX(12, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(13, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 10, 22);
    src = in[lane + LANE_COUNT * 11];
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 0, 22);
    out[INDEX(16, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 12];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(17, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 13];
        tmp = FUNNEL_SHIFT_R(src, next, 12) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(18, lane)] = tmp;
    tmp = BFE(src, 2, 22);
    out[INDEX(19, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 14];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(20, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 15];
        tmp = FUNNEL_SHIFT_R(src, next, 14) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(21, lane)] = tmp;
    tmp = BFE(src, 4, 22);
    out[INDEX(22, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 16];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(23, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 17];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(24, lane)] = tmp;
    tmp = BFE(src, 6, 22);
    out[INDEX(25, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 18];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(26, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 19];
        tmp = FUNNEL_SHIFT_R(src, next, 18) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(27, lane)] = tmp;
    tmp = BFE(src, 8, 22);
    out[INDEX(28, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 20];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(29, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 21];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 22);
        src = next;
    }
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 10, 22);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_23bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 23);
    out[INDEX(0, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 23) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(1, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 14) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(2, lane)] = tmp;
    tmp = BFE(src, 5, 23);
    out[INDEX(3, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(4, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 19) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(5, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 10) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 1, 23);
    out[INDEX(7, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(8, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 15) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(9, lane)] = tmp;
    tmp = BFE(src, 6, 23);
    out[INDEX(10, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 29) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(11, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(12, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 11) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(13, lane)] = tmp;
    tmp = BFE(src, 2, 23);
    out[INDEX(14, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 25) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(15, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 12];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(16, lane)] = tmp;
    tmp = BFE(src, 7, 23);
    out[INDEX(17, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 13];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(18, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 14];
        tmp = FUNNEL_SHIFT_R(src, next, 21) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(19, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 15];
        tmp = FUNNEL_SHIFT_R(src, next, 12) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(20, lane)] = tmp;
    tmp = BFE(src, 3, 23);
    out[INDEX(21, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 16];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(22, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 17];
        tmp = FUNNEL_SHIFT_R(src, next, 17) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 8, 23);
    out[INDEX(24, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 18];
        tmp = FUNNEL_SHIFT_R(src, next, 31) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(25, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 19];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(26, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 20];
        tmp = FUNNEL_SHIFT_R(src, next, 13) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(27, lane)] = tmp;
    tmp = BFE(src, 4, 23);
    out[INDEX(28, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 21];
        tmp = FUNNEL_SHIFT_R(src, next, 27) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(29, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 22];
        tmp = FUNNEL_SHIFT_R(src, next, 18) & MASK(uint32_t, 23);
        src = next;
    }
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 9, 23);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_24bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 24);
    out[INDEX(0, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 24);
        src = next;
    }
    out[INDEX(1, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 24);
        src = next;
    }
    out[INDEX(2, lane)] = tmp;
    tmp = BFE(src, 8, 24);
    src = in[lane + LANE_COUNT * 3];
    out[INDEX(3, lane)] = tmp;
    tmp = BFE(src, 0, 24);
    out[INDEX(4, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 24);
        src = next;
    }
    out[INDEX(5, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 24);
        src = next;
    }
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 8, 24);
    src = in[lane + LANE_COUNT * 6];
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 0, 24);
    out[INDEX(8, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 24);
        src = next;
    }
    out[INDEX(9, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 24);
        src = next;
    }
    out[INDEX(10, lane)] = tmp;
    tmp = BFE(src, 8, 24);
    src = in[lane + LANE_COUNT * 9];
    out[INDEX(11, lane)] = tmp;
    tmp = BFE(src, 0, 24);
    out[INDEX(12, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 24);
        src = next;
    }
    out[INDEX(13, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 24);
        src = next;
    }
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 8, 24);
    src = in[lane + LANE_COUNT * 12];
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 0, 24);
    out[INDEX(16, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 13];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 24);
        src = next;
    }
    out[INDEX(17, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 14];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 24);
        src = next;
    }
    out[INDEX(18, lane)] = tmp;
    tmp = BFE(src, 8, 24);
    src = in[lane + LANE_COUNT * 15];
    out[INDEX(19, lane)] = tmp;
    tmp = BFE(src, 0, 24);
    out[INDEX(20, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 16];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 24);
        src = next;
    }
    out[INDEX(21, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 17];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 24);
        src = next;
    }
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 8, 24);
    src = in[lane + LANE_COUNT * 18];
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 0, 24);
    out[INDEX(24, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 19];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 24);
        src = next;
    }
    out[INDEX(25, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 20];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 24);
        src = next;
    }
    out[INDEX(26, lane)] = tmp;
    tmp = BFE(src, 8, 24);
    src = in[lane + LANE_COUNT * 21];
    out[INDEX(27, lane)] = tmp;
    tmp = BFE(src, 0, 24);
    out[INDEX(28, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 22];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 24);
        src = next;
    }
    out[INDEX(29, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 23];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 24);
        src = next;
    }
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 8, 24);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_25bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 25);
    out[INDEX(0, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 25) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(1, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 18) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(2, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 11) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(3, lane)] = tmp;
    tmp = BFE(src, 4, 25);
    out[INDEX(4, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 29) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(5, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(6, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 15) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(7, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 8) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(8, lane)] = tmp;
    tmp = BFE(src, 1, 25);
    out[INDEX(9, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(10, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 19) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(11, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 12) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(12, lane)] = tmp;
    tmp = BFE(src, 5, 25);
    out[INDEX(13, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(14, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 12];
        tmp = FUNNEL_SHIFT_R(src, next, 23) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(15, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 13];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(16, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 14];
        tmp = FUNNEL_SHIFT_R(src, next, 9) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(17, lane)] = tmp;
    tmp = BFE(src, 2, 25);
    out[INDEX(18, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 15];
        tmp = FUNNEL_SHIFT_R(src, next, 27) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(19, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 16];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(20, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 17];
        tmp = FUNNEL_SHIFT_R(src, next, 13) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(21, lane)] = tmp;
    tmp = BFE(src, 6, 25);
    out[INDEX(22, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 18];
        tmp = FUNNEL_SHIFT_R(src, next, 31) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(23, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 19];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(24, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 20];
        tmp = FUNNEL_SHIFT_R(src, next, 17) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(25, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 21];
        tmp = FUNNEL_SHIFT_R(src, next, 10) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(26, lane)] = tmp;
    tmp = BFE(src, 3, 25);
    out[INDEX(27, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 22];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(28, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 23];
        tmp = FUNNEL_SHIFT_R(src, next, 21) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(29, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 24];
        tmp = FUNNEL_SHIFT_R(src, next, 14) & MASK(uint32_t, 25);
        src = next;
    }
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 7, 25);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_26bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 26);
    out[INDEX(0, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(1, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(2, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 14) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(3, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 8) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(4, lane)] = tmp;
    tmp = BFE(src, 2, 26);
    out[INDEX(5, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(6, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(7, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(8, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 10) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(9, lane)] = tmp;
    tmp = BFE(src, 4, 26);
    out[INDEX(10, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(11, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(12, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 18) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(13, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 12];
        tmp = FUNNEL_SHIFT_R(src, next, 12) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 6, 26);
    src = in[lane + LANE_COUNT * 13];
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 0, 26);
    out[INDEX(16, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 14];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(17, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 15];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(18, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 16];
        tmp = FUNNEL_SHIFT_R(src, next, 14) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(19, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 17];
        tmp = FUNNEL_SHIFT_R(src, next, 8) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(20, lane)] = tmp;
    tmp = BFE(src, 2, 26);
    out[INDEX(21, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 18];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(22, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 19];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(23, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 20];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(24, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 21];
        tmp = FUNNEL_SHIFT_R(src, next, 10) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(25, lane)] = tmp;
    tmp = BFE(src, 4, 26);
    out[INDEX(26, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 22];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(27, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 23];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(28, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 24];
        tmp = FUNNEL_SHIFT_R(src, next, 18) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(29, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 25];
        tmp = FUNNEL_SHIFT_R(src, next, 12) & MASK(uint32_t, 26);
        src = next;
    }
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 6, 26);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_27bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 27);
    out[INDEX(0, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 27) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(1, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(2, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 17) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(3, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 12) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(4, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 7) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(5, lane)] = tmp;
    tmp = BFE(src, 2, 27);
    out[INDEX(6, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 29) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(7, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(8, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 19) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(9, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 14) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(10, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 9) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(11, lane)] = tmp;
    tmp = BFE(src, 4, 27);
    out[INDEX(12, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 31) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(13, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 12];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(14, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 13];
        tmp = FUNNEL_SHIFT_R(src, next, 21) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(15, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 14];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(16, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 15];
        tmp = FUNNEL_SHIFT_R(src, next, 11) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(17, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 16];
        tmp = FUNNEL_SHIFT_R(src, next, 6) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(18, lane)] = tmp;
    tmp = BFE(src, 1, 27);
    out[INDEX(19, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 17];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(20, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 18];
        tmp = FUNNEL_SHIFT_R(src, next, 23) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(21, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 19];
        tmp = FUNNEL_SHIFT_R(src, next, 18) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(22, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 20];
        tmp = FUNNEL_SHIFT_R(src, next, 13) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(23, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 21];
        tmp = FUNNEL_SHIFT_R(src, next, 8) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(24, lane)] = tmp;
    tmp = BFE(src, 3, 27);
    out[INDEX(25, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 22];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(26, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 23];
        tmp = FUNNEL_SHIFT_R(src, next, 25) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(27, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 24];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(28, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 25];
        tmp = FUNNEL_SHIFT_R(src, next, 15) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(29, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 26];
        tmp = FUNNEL_SHIFT_R(src, next, 10) & MASK(uint32_t, 27);
        src = next;
    }
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 5, 27);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_28bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 28);
    out[INDEX(0, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(1, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(2, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(3, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(4, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 12) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(5, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 8) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(6, lane)] = tmp;
    tmp = BFE(src, 4, 28);
    src = in[lane + LANE_COUNT * 7];
    out[INDEX(7, lane)] = tmp;
    tmp = BFE(src, 0, 28);
    out[INDEX(8, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(9, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(10, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(11, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(12, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 12];
        tmp = FUNNEL_SHIFT_R(src, next, 12) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(13, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 13];
        tmp = FUNNEL_SHIFT_R(src, next, 8) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 4, 28);
    src = in[lane + LANE_COUNT * 14];
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 0, 28);
    out[INDEX(16, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 15];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(17, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 16];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(18, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 17];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(19, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 18];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(20, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 19];
        tmp = FUNNEL_SHIFT_R(src, next, 12) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(21, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 20];
        tmp = FUNNEL_SHIFT_R(src, next, 8) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(22, lane)] = tmp;
    tmp = BFE(src, 4, 28);
    src = in[lane + LANE_COUNT * 21];
    out[INDEX(23, lane)] = tmp;
    tmp = BFE(src, 0, 28);
    out[INDEX(24, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 22];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(25, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 23];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(26, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 24];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(27, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 25];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(28, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 26];
        tmp = FUNNEL_SHIFT_R(src, next, 12) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(29, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 27];
        tmp = FUNNEL_SHIFT_R(src, next, 8) & MASK(uint32_t, 28);
        src = next;
    }
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 4, 28);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_29bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 29);
    out[INDEX(0, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 29) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(1, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(2, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 23) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(3, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(4, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 17) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(5, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 14) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(6, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 11) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(7, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 8) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(8, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 5) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(9, lane)] = tmp;
    tmp = BFE(src, 2, 29);
    out[INDEX(10, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 31) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(11, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(12, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 12];
        tmp = FUNNEL_SHIFT_R(src, next, 25) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(13, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 13];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(14, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 14];
        tmp = FUNNEL_SHIFT_R(src, next, 19) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(15, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 15];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(16, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 16];
        tmp = FUNNEL_SHIFT_R(src, next, 13) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(17, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 17];
        tmp = FUNNEL_SHIFT_R(src, next, 10) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(18, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 18];
        tmp = FUNNEL_SHIFT_R(src, next, 7) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(19, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 19];
        tmp = FUNNEL_SHIFT_R(src, next, 4) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(20, lane)] = tmp;
    tmp = BFE(src, 1, 29);
    out[INDEX(21, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 20];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(22, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 21];
        tmp = FUNNEL_SHIFT_R(src, next, 27) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(23, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 22];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(24, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 23];
        tmp = FUNNEL_SHIFT_R(src, next, 21) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(25, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 24];
        tmp = FUNNEL_SHIFT_R(src, next, 18) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(26, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 25];
        tmp = FUNNEL_SHIFT_R(src, next, 15) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(27, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 26];
        tmp = FUNNEL_SHIFT_R(src, next, 12) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(28, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 27];
        tmp = FUNNEL_SHIFT_R(src, next, 9) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(29, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 28];
        tmp = FUNNEL_SHIFT_R(src, next, 6) & MASK(uint32_t, 29);
        src = next;
    }
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 3, 29);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_30bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 30);
    out[INDEX(0, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(1, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(2, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(3, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(4, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(5, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(6, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 18) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(7, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(8, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 14) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(9, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 12) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(10, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 10) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(11, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 12];
        tmp = FUNNEL_SHIFT_R(src, next, 8) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(12, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 13];
        tmp = FUNNEL_SHIFT_R(src, next, 6) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(13, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 14];
        tmp = FUNNEL_SHIFT_R(src, next, 4) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(14, lane)] = tmp;
    tmp = BFE(src, 2, 30);
    src = in[lane + LANE_COUNT * 15];
    out[INDEX(15, lane)] = tmp;
    tmp = BFE(src, 0, 30);
    out[INDEX(16, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 16];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(17, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 17];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(18, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 18];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(19, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 19];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(20, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 20];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(21, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 21];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(22, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 22];
        tmp = FUNNEL_SHIFT_R(src, next, 18) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(23, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 23];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(24, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 24];
        tmp = FUNNEL_SHIFT_R(src, next, 14) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(25, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 25];
        tmp = FUNNEL_SHIFT_R(src, next, 12) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(26, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 26];
        tmp = FUNNEL_SHIFT_R(src, next, 10) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(27, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 27];
        tmp = FUNNEL_SHIFT_R(src, next, 8) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(28, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 28];
        tmp = FUNNEL_SHIFT_R(src, next, 6) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(29, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 29];
        tmp = FUNNEL_SHIFT_R(src, next, 4) & MASK(uint32_t, 30);
        src = next;
    }
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 2, 30);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_31bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
    unsigned int LANE_COUNT = 32;
    uint32_t src;
    uint32_t tmp;

    src = in[lane];
    tmp = BFE(src, 0, 31);
    out[INDEX(0, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 1];
        tmp = FUNNEL_SHIFT_R(src, next, 31) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(1, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 2];
        tmp = FUNNEL_SHIFT_R(src, next, 30) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(2, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 3];
        tmp = FUNNEL_SHIFT_R(src, next, 29) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(3, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 4];
        tmp = FUNNEL_SHIFT_R(src, next, 28) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(4, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 5];
        tmp = FUNNEL_SHIFT_R(src, next, 27) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(5, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 6];
        tmp = FUNNEL_SHIFT_R(src, next, 26) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(6, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 7];
        tmp = FUNNEL_SHIFT_R(src, next, 25) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(7, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 8];
        tmp = FUNNEL_SHIFT_R(src, next, 24) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(8, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 9];
        tmp = FUNNEL_SHIFT_R(src, next, 23) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(9, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 10];
        tmp = FUNNEL_SHIFT_R(src, next, 22) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(10, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 11];
        tmp = FUNNEL_SHIFT_R(src, next, 21) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(11, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 12];
        tmp = FUNNEL_SHIFT_R(src, next, 20) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(12, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 13];
        tmp = FUNNEL_SHIFT_R(src, next, 19) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(13, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 14];
        tmp = FUNNEL_SHIFT_R(src, next, 18) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(14, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 15];
        tmp = FUNNEL_SHIFT_R(src, next, 17) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(15, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 16];
        tmp = FUNNEL_SHIFT_R(src, next, 16) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(16, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 17];
        tmp = FUNNEL_SHIFT_R(src, next, 15) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(17, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 18];
        tmp = FUNNEL_SHIFT_R(src, next, 14) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(18, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 19];
        tmp = FUNNEL_SHIFT_R(src, next, 13) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(19, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 20];
        tmp = FUNNEL_SHIFT_R(src, next, 12) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(20, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 21];
        tmp = FUNNEL_SHIFT_R(src, next, 11) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(21, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 22];
        tmp = FUNNEL_SHIFT_R(src, next, 10) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(22, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 23];
        tmp = FUNNEL_SHIFT_R(src, next, 9) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(23, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 24];
        tmp = FUNNEL_SHIFT_R(src, next, 8) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(24, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 25];
        tmp = FUNNEL_SHIFT_R(src, next, 7) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(25, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 26];
        tmp = FUNNEL_SHIFT_R(src, next, 6) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(26, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 27];
        tmp = FUNNEL_SHIFT_R(src, next, 5) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(27, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 28];
        tmp = FUNNEL_SHIFT_R(src, next, 4) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(28, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 29];
        tmp = FUNNEL_SHIFT_R(src, next, 3) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(29, lane)] = tmp;
    {
        uint32_t next = in[lane + LANE_COUNT * 30];
        tmp = FUNNEL_SHIFT_R(src, next, 2) & MASK(uint32_t, 31);
        src = next;
    }
    out[INDEX(30, lane)] = tmp;
    tmp = BFE(src, 1, 31);
    out[INDEX(31, lane)] = tmp;
}

__device__ void
_bit_unpack_32_32bw_lane(const uint32_t *__restrict in, uint32_t *__restrict out, unsigned int lane) {
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

/// Runtime dispatch to the optimized lane decoder for the given bit width.
__device__ inline void bit_unpack_32_lane(const uint32_t *__restrict in,
                                          uint32_t *__restrict out,
                                          unsigned int lane,
                                          uint32_t bit_width) {
    switch (bit_width) {
    case 0:
        _bit_unpack_32_0bw_lane(in, out, lane);
        break;
    case 1:
        _bit_unpack_32_1bw_lane(in, out, lane);
        break;
    case 2:
        _bit_unpack_32_2bw_lane(in, out, lane);
        break;
    case 3:
        _bit_unpack_32_3bw_lane(in, out, lane);
        break;
    case 4:
        _bit_unpack_32_4bw_lane(in, out, lane);
        break;
    case 5:
        _bit_unpack_32_5bw_lane(in, out, lane);
        break;
    case 6:
        _bit_unpack_32_6bw_lane(in, out, lane);
        break;
    case 7:
        _bit_unpack_32_7bw_lane(in, out, lane);
        break;
    case 8:
        _bit_unpack_32_8bw_lane(in, out, lane);
        break;
    case 9:
        _bit_unpack_32_9bw_lane(in, out, lane);
        break;
    case 10:
        _bit_unpack_32_10bw_lane(in, out, lane);
        break;
    case 11:
        _bit_unpack_32_11bw_lane(in, out, lane);
        break;
    case 12:
        _bit_unpack_32_12bw_lane(in, out, lane);
        break;
    case 13:
        _bit_unpack_32_13bw_lane(in, out, lane);
        break;
    case 14:
        _bit_unpack_32_14bw_lane(in, out, lane);
        break;
    case 15:
        _bit_unpack_32_15bw_lane(in, out, lane);
        break;
    case 16:
        _bit_unpack_32_16bw_lane(in, out, lane);
        break;
    case 17:
        _bit_unpack_32_17bw_lane(in, out, lane);
        break;
    case 18:
        _bit_unpack_32_18bw_lane(in, out, lane);
        break;
    case 19:
        _bit_unpack_32_19bw_lane(in, out, lane);
        break;
    case 20:
        _bit_unpack_32_20bw_lane(in, out, lane);
        break;
    case 21:
        _bit_unpack_32_21bw_lane(in, out, lane);
        break;
    case 22:
        _bit_unpack_32_22bw_lane(in, out, lane);
        break;
    case 23:
        _bit_unpack_32_23bw_lane(in, out, lane);
        break;
    case 24:
        _bit_unpack_32_24bw_lane(in, out, lane);
        break;
    case 25:
        _bit_unpack_32_25bw_lane(in, out, lane);
        break;
    case 26:
        _bit_unpack_32_26bw_lane(in, out, lane);
        break;
    case 27:
        _bit_unpack_32_27bw_lane(in, out, lane);
        break;
    case 28:
        _bit_unpack_32_28bw_lane(in, out, lane);
        break;
    case 29:
        _bit_unpack_32_29bw_lane(in, out, lane);
        break;
    case 30:
        _bit_unpack_32_30bw_lane(in, out, lane);
        break;
    case 31:
        _bit_unpack_32_31bw_lane(in, out, lane);
        break;
    case 32:
        _bit_unpack_32_32bw_lane(in, out, lane);
        break;
    }
}

__device__ void
_bit_unpack_32_0bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_0bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_0bw_32t(const uint32_t *__restrict full_in,
                                                 uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 0 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_0bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_1bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_1bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_1bw_32t(const uint32_t *__restrict full_in,
                                                 uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 1 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_1bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_2bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_2bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_2bw_32t(const uint32_t *__restrict full_in,
                                                 uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 2 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_2bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_3bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_3bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_3bw_32t(const uint32_t *__restrict full_in,
                                                 uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 3 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_3bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_4bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_4bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_4bw_32t(const uint32_t *__restrict full_in,
                                                 uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 4 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_4bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_5bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_5bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_5bw_32t(const uint32_t *__restrict full_in,
                                                 uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 5 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_5bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_6bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_6bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_6bw_32t(const uint32_t *__restrict full_in,
                                                 uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 6 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_6bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_7bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_7bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_7bw_32t(const uint32_t *__restrict full_in,
                                                 uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 7 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_7bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_8bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_8bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_8bw_32t(const uint32_t *__restrict full_in,
                                                 uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 8 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_8bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_9bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_9bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_9bw_32t(const uint32_t *__restrict full_in,
                                                 uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 9 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_9bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_10bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_10bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_10bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 10 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_10bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_11bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_11bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_11bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 11 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_11bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_12bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_12bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_12bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 12 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_12bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_13bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_13bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_13bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 13 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_13bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_14bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_14bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_14bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 14 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_14bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_15bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_15bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_15bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 15 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_15bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_16bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_16bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_16bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 16 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_16bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_17bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_17bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_17bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 17 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_17bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_18bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_18bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_18bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 18 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_18bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_19bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_19bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_19bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 19 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_19bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_20bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_20bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_20bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 20 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_20bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_21bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_21bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_21bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 21 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_21bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_22bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_22bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_22bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 22 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_22bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_23bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_23bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_23bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 23 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_23bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_24bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_24bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_24bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 24 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_24bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_25bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_25bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_25bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 25 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_25bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_26bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_26bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_26bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 26 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_26bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_27bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_27bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_27bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 27 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_27bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_28bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_28bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_28bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 28 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_28bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_29bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_29bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_29bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 29 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_29bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_30bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_30bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_30bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 30 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_30bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_31bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_31bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_31bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 31 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_31bw_32t(in, out, thread_idx);
}

__device__ void
_bit_unpack_32_32bw_32t(const uint32_t *__restrict in, uint32_t *__restrict out, int thread_idx) {
    __shared__ uint32_t shared_out[1024];
    _bit_unpack_32_32bw_lane(in, shared_out, thread_idx * 1 + 0);
    for (int i = 0; i < 32; i++) {
        auto idx = i * 32 + thread_idx;
        out[idx] = shared_out[idx];
    }
}

extern "C" __global__ void bit_unpack_32_32bw_32t(const uint32_t *__restrict full_in,
                                                  uint32_t *__restrict full_out) {
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * 32 / sizeof(uint32_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_32_32bw_32t(in, out, thread_idx);
}
