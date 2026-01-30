// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Apply patches to a source array
template<typename ValuesT, typename IndexT>
__device__ void patches_apply_inplace(
    ValuesT *const values,
    const IndexT *const patchIndices,
    const ValueT *const patchValues,
    uint64_t valuesLen,
    uint64_t patchesLen,
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > patchesLen) {
        return;
    }

    const IndexT patchIdx = patchIndices[idx];
    const ValueT patchVal = patchValues[idx];

    const size_t valueIdx = static_cast<size_t>(patchIdx);
    values[valueIdx] = patchVal;
}

#define GENERATE_PATCHES_KERNEL(ValuesT, IndicesT) \
extern "C" __global__ patches_apply_inplace