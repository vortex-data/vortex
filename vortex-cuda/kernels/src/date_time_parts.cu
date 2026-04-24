// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include "types.cuh"

constexpr int64_t SECONDS_PER_DAY = 86400;

// Combines date/time parts (days, seconds, subseconds) into timestamp values.
template <typename DaysT, typename SecondsT, typename SubsecondsT>
__device__ void date_time_parts(const DaysT *__restrict days,
                                const SecondsT *__restrict seconds,
                                const SubsecondsT *__restrict subseconds,
                                int64_t divisor,
                                int64_t *__restrict output,
                                uint64_t array_len) {
    const int64_t ticks_per_day = SECONDS_PER_DAY * divisor;
    const uint32_t elements_per_block = blockDim.x * ELEMENTS_PER_THREAD;

    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * elements_per_block;
    const uint64_t block_end = min(block_start + elements_per_block, array_len);

    for (uint64_t idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
        output[idx] = static_cast<int64_t>(days[idx]) * ticks_per_day +
                      static_cast<int64_t>(seconds[idx]) * divisor + static_cast<int64_t>(subseconds[idx]);
    }
}

#define GENERATE_DATE_TIME_PARTS_KERNEL(days_suffix,                                                         \
                                        DaysT,                                                               \
                                        seconds_suffix,                                                      \
                                        SecondsT,                                                            \
                                        subseconds_suffix,                                                   \
                                        SubsecondsT)                                                         \
    extern "C" __global__ void date_time_parts_##days_suffix##_##seconds_suffix##_##subseconds_suffix(       \
        const DaysT *__restrict days,                                                                        \
        const SecondsT *__restrict seconds,                                                                  \
        const SubsecondsT *__restrict subseconds,                                                            \
        int64_t divisor,                                                                                     \
        int64_t *__restrict output,                                                                          \
        uint64_t array_len) {                                                                                \
        date_time_parts(days, seconds, subseconds, divisor, output, array_len);                              \
    }

#define EXPAND_DAYS(X)                                                                                       \
    X(i8, int8_t)                                                                                            \
    X(i16, int16_t)                                                                                          \
    X(i32, int32_t)                                                                                          \
    X(i64, int64_t)

#define EXPAND_SUBSECONDS(d, DT, s, ST)                                                                      \
    GENERATE_DATE_TIME_PARTS_KERNEL(d, DT, s, ST, i8, int8_t)                                                \
    GENERATE_DATE_TIME_PARTS_KERNEL(d, DT, s, ST, i16, int16_t)                                              \
    GENERATE_DATE_TIME_PARTS_KERNEL(d, DT, s, ST, i32, int32_t)                                              \
    GENERATE_DATE_TIME_PARTS_KERNEL(d, DT, s, ST, i64, int64_t)

#define EXPAND_SECONDS(d, DT)                                                                                \
    EXPAND_SUBSECONDS(d, DT, i8, int8_t)                                                                     \
    EXPAND_SUBSECONDS(d, DT, i16, int16_t)                                                                   \
    EXPAND_SUBSECONDS(d, DT, i32, int32_t)                                                                   \
    EXPAND_SUBSECONDS(d, DT, i64, int64_t)

// Generate all 64 kernels (4³)
EXPAND_DAYS(EXPAND_SECONDS)
