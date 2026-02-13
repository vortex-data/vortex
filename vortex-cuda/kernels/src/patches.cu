// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"
#include "types.cuh"

// TODO(aduffy): this is very naive. In the future we need to
//   transpose the patches, see G-ALP paper.
// Apply patches to a source array
template <typename ValueT, typename IndexT>
__device__ void patches(ValueT *const values,
                        const IndexT *const patchIndices,
                        const ValueT *const patchValues,
                        uint64_t patchesLen) {
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t startElem = START_ELEM(worker, patchesLen);
    const uint64_t stopElem = STOP_ELEM(worker, patchesLen);

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

#define GENERATE_PATCHES_KERNEL(ValueT, value_suffix, IndexT, index_suffix)                                  \
    extern "C" __global__ void patches_##value_suffix##_##index_suffix(ValueT *const values,                 \
                                                                       const IndexT *const patchIndices,     \
                                                                       const ValueT *const patchValues,      \
                                                                       uint64_t patchesLen) {                \
        patches(values, patchIndices, patchValues, patchesLen);                                              \
    }

// Generate patches kernel for all index types (unsigned integers) for a given value type
#define GENERATE_PATCHES_FOR_ALL_INDICES(value_suffix, ValueT)                                               \
    GENERATE_PATCHES_KERNEL(ValueT, value_suffix, uint8_t, u8)                                               \
    GENERATE_PATCHES_KERNEL(ValueT, value_suffix, uint16_t, u16)                                             \
    GENERATE_PATCHES_KERNEL(ValueT, value_suffix, uint32_t, u32)                                             \
    GENERATE_PATCHES_KERNEL(ValueT, value_suffix, uint64_t, u64)

// Generate for all native SIMD ptypes
FOR_EACH_NATIVE_SIMD_PTYPE(GENERATE_PATCHES_FOR_ALL_INDICES)
