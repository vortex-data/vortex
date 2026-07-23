// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"

#include <limits.h>
#include <stdint.h>

// Convert integer lengths to CUB scan input. The final zero makes an exclusive scan produce an
// Arrow offsets array of length len + 1.
#define GENERATE_FROM_LENGTHS_KERNEL(suffix, LengthT, is_signed)                                             \
    extern "C" __global__ void arrow_offsets_from_lengths_##suffix(const LengthT *__restrict lengths,        \
                                                                   int32_t *__restrict scan,                 \
                                                                   uint32_t *status,                         \
                                                                   uint64_t len) {                           \
        const uint64_t scan_len = len + 1;                                                                   \
        const uint64_t elements_per_block = (uint64_t)blockDim.x * ELEMENTS_PER_THREAD;                      \
        const uint64_t block_start = (uint64_t)blockIdx.x * elements_per_block;                              \
        const uint64_t block_stop = min(block_start + elements_per_block, scan_len);                         \
        for (uint64_t idx = block_start + threadIdx.x; idx < block_stop; idx += blockDim.x) {                \
            if (idx == len) {                                                                                \
                scan[idx] = 0;                                                                               \
                continue;                                                                                    \
            }                                                                                                \
            const LengthT raw_length = lengths[idx];                                                         \
            if (is_signed && raw_length < 0) {                                                               \
                scan[idx] = 0;                                                                               \
                atomicMax(status, 1u);                                                                       \
                continue;                                                                                    \
            }                                                                                                \
            const uint64_t length = (uint64_t)raw_length;                                                    \
            if (length > (uint64_t)INT32_MAX) {                                                              \
                scan[idx] = 0;                                                                               \
                atomicMax(status, 2u);                                                                       \
                continue;                                                                                    \
            }                                                                                                \
            scan[idx] = (int32_t)length;                                                                     \
        }                                                                                                    \
    }

// The first i32 prefix-sum overflow must be negative because every scan input is nonnegative and at
// most INT32_MAX. Retaining this validation as a separate pass keeps malformed inputs from reaching
// consumers with wrapped Arrow offsets.
extern "C" __global__ void
arrow_offsets_validate(const int32_t *__restrict offsets, uint32_t *status, uint64_t scan_len) {
    const uint64_t elements_per_block = (uint64_t)blockDim.x * ELEMENTS_PER_THREAD;
    const uint64_t block_start = (uint64_t)blockIdx.x * elements_per_block;
    const uint64_t block_stop = min(block_start + elements_per_block, scan_len);
    for (uint64_t idx = block_start + threadIdx.x; idx < block_stop; idx += blockDim.x) {
        if (offsets[idx] < 0) {
            atomicMax(status, 2u);
        }
    }
}

GENERATE_FROM_LENGTHS_KERNEL(i8, int8_t, true)
GENERATE_FROM_LENGTHS_KERNEL(i16, int16_t, true)
GENERATE_FROM_LENGTHS_KERNEL(i32, int32_t, true)
GENERATE_FROM_LENGTHS_KERNEL(i64, int64_t, true)
GENERATE_FROM_LENGTHS_KERNEL(u8, uint8_t, false)
GENERATE_FROM_LENGTHS_KERNEL(u16, uint16_t, false)
GENERATE_FROM_LENGTHS_KERNEL(u32, uint32_t, false)
GENERATE_FROM_LENGTHS_KERNEL(u64, uint64_t, false)
