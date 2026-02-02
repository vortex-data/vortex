// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"

// TODO(aduffy): this is very naive. In the future we need to
//   transpose the patches, see G-ALP paper.
// Apply patches to a source array
template<typename ValueT, typename IndexT>
__device__ void patches(
    ValueT *const values,
    const IndexT *const patchIndices,
    const ValueT *const patchValues,
    uint64_t patchesLen
) {
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t startElem = START_ELEM(worker, patchesLen);
    const uint64_t stopElem = START_ELEM(worker, patchesLen);

    if (startElem >= patchesLen) {
        return;
    }

    for (uint64_t idx = startElem; idx < stopElem; idx++) {
        const IndexT patchIdx = patchIndices[idx];
        const ValueT patchVal = patchValues[idx];

        const size_t valueIdx = static_cast<size_t>(patchIdx);
        values[valueIdx] = patchVal;
    }
}

#define GENERATE_PATCHES_KERNEL(ValueT, value_suffix, IndexT, index_suffix) \
extern "C" __global__ void patches_##value_suffix##_##index_suffix( \
    ValueT *const values, \
    const IndexT *const patchIndices, \
    const ValueT *const patchValues, \
    uint64_t patchesLen \
) { \
    patches(values, patchIndices, patchValues, patchesLen); \
}

#define GENERATE_PATCHES_KERNEL_FOR_VALUE(ValueT, value_suffix) \
     GENERATE_PATCHES_KERNEL(ValueT, value_suffix, uint8_t, u8) \
     GENERATE_PATCHES_KERNEL(ValueT, value_suffix, uint16_t, u16) \
     GENERATE_PATCHES_KERNEL(ValueT, value_suffix, uint32_t, u32) \
     GENERATE_PATCHES_KERNEL(ValueT, value_suffix, uint64_t, u64)


GENERATE_PATCHES_KERNEL_FOR_VALUE(uint8_t, u8)
GENERATE_PATCHES_KERNEL_FOR_VALUE(uint16_t, u16)
GENERATE_PATCHES_KERNEL_FOR_VALUE(uint32_t, u32)
GENERATE_PATCHES_KERNEL_FOR_VALUE(uint64_t, u64)

GENERATE_PATCHES_KERNEL_FOR_VALUE(int8_t, i8)
GENERATE_PATCHES_KERNEL_FOR_VALUE(int16_t, i16)
GENERATE_PATCHES_KERNEL_FOR_VALUE(int32_t, i32)
GENERATE_PATCHES_KERNEL_FOR_VALUE(int64_t, i64)

GENERATE_PATCHES_KERNEL_FOR_VALUE(float, f32)
GENERATE_PATCHES_KERNEL_FOR_VALUE(double, f64)
