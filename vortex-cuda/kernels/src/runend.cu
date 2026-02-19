// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include "config.cuh"
#include "types.cuh"

constexpr uint32_t MAX_CACHED_RUNS = 512;

/// Binary search for the first element strictly greater than `value`.
///
/// Uses `thrust::upper_bound` with sequential execution policy. `thrust::seq`
/// is chosen as the binary search runs on a single GPU thread. This is
/// preferred over `thrust::device` as this would spawn an additional kernel
/// launch.
/// See:
/// https://nvidia.github.io/cccl/thrust/api/group__binary__search_1gac85cc9ea00f4bdd8f80ad25fff16741d.html#thrust-upper-bound
///
/// Returns the index of the first element that is greater than `value`, or
/// `len` if no such element exists.
template <typename T>
__device__ inline uint64_t upper_bound(const T *data, uint64_t len, uint64_t value) {

    auto it = thrust::upper_bound(thrust::seq, data, data + len, value);
    return it - data;
}

// Decodes run-end encoded data on the GPU.
//
// Run-end stores data as pairs of (value, end_position) where each run contains
// repeated values from the previous end position to the current end position.
//
// Steps:
// 1. Each CUDA block processes a contiguous chunk of output elements (elements_per_block).
//
// 2. Block Initialization (Thread 0 only):
//    - Compute the global position range [block_start + offset, block_end + offset) for this block
//    - Use binary search (upper_bound) to find the first and last runs that overlap this range
//    - Store the run range in shared memory (block_first_run, block_num_runs)
//
// 3. Shared Memory Caching:
//    - If the number of runs for this block fits in shared memory (< MAX_CACHED_RUNS),
//      all threads cooperatively load the relevant ends[] and values[] into shared memory
//    - This is to reduce global memory access during decoding
//
// 4. Decoding:
//    a) Cached path: Each thread decodes multiple elements using a forward scan.
//       Since thread positions are strided (idx += blockDim.x), and positions are monotonically
//       increasing across iterations, we maintain a current_run index that only moves forward.
//
//    b) Fallback path: If too many runs span this block (exceeds MAX_CACHED_RUNS),
//       fall back to binary search in global memory for each element.
//
// TODO(0ax1): Investigate whether there are faster solutions.
template <typename ValueT, typename EndsT>
__device__ void runend_decode_kernel(const EndsT *const __restrict ends,
                                     uint64_t num_runs,
                                     const ValueT *const __restrict values,
                                     uint64_t offset,
                                     uint64_t output_len,
                                     ValueT *const __restrict output) {
    __shared__ EndsT shared_ends[MAX_CACHED_RUNS];
    __shared__ ValueT shared_values[MAX_CACHED_RUNS];
    __shared__ uint64_t block_first_run;
    __shared__ uint32_t block_num_runs;

    const uint32_t elements_per_block = blockDim.x * ELEMENTS_PER_THREAD;
    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * elements_per_block;
    const uint64_t block_end = min(block_start + elements_per_block, output_len);

    if (block_start >= output_len)
        return;

    // Thread 0 finds the run range for this block.
    if (threadIdx.x == 0) {
        uint64_t first_pos = block_start + offset;
        uint64_t last_pos = (block_end - 1) + offset;

        uint64_t first_run = upper_bound(ends, num_runs, first_pos);
        uint64_t last_run = upper_bound(ends, num_runs, last_pos);

        block_first_run = first_run;
        block_num_runs =
            static_cast<uint32_t>(min(last_run - first_run + 1, static_cast<uint64_t>(MAX_CACHED_RUNS)));
    }
    __syncthreads();

    // Cooperatively load ends and values into shared memory.
    if (block_num_runs < MAX_CACHED_RUNS) {
        for (uint32_t i = threadIdx.x; i < block_num_runs; i += blockDim.x) {
            shared_ends[i] = ends[block_first_run + i];
            shared_values[i] = values[block_first_run + i];
        }
    }
    __syncthreads();

    if (block_num_runs < MAX_CACHED_RUNS) {
        uint32_t current_run = 0;
        for (uint64_t idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
            uint64_t pos = idx + offset;

            // Scan forward to find the run containing this position
            while (current_run < block_num_runs && static_cast<uint64_t>(shared_ends[current_run]) <= pos) {
                current_run++;
            }

            output[idx] = shared_values[current_run < block_num_runs ? current_run : block_num_runs - 1];
        }
    } else {
        // Fallback for blocks with very short runs. Search the full `num_runs`
        // array. `block_num_runs` is clamped to `MAX_CACHED_RUNS`.
        for (uint64_t idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
            uint64_t pos = idx + offset;
            uint64_t run_idx = upper_bound(ends, num_runs, pos);
            if (run_idx >= num_runs)
                run_idx = num_runs - 1;
            output[idx] = values[run_idx];
        }
    }
}

#define GENERATE_RUNEND_KERNEL(value_suffix, ValueType, ends_suffix, EndsType)                               \
    extern "C" __global__ void runend_##value_suffix##_##ends_suffix(                                        \
        const EndsType *const __restrict ends,                                                               \
        uint64_t num_runs,                                                                                   \
        const ValueType *const __restrict values,                                                            \
        uint64_t offset,                                                                                     \
        uint64_t output_len,                                                                                 \
        ValueType *const __restrict output) {                                                                \
        runend_decode_kernel<ValueType, EndsType>(ends, num_runs, values, offset, output_len, output);       \
    }

#define GENERATE_RUNEND_KERNELS_FOR_VALUE(value_suffix, ValueType)                                           \
    GENERATE_RUNEND_KERNEL(value_suffix, ValueType, u8, uint8_t)                                             \
    GENERATE_RUNEND_KERNEL(value_suffix, ValueType, u16, uint16_t)                                           \
    GENERATE_RUNEND_KERNEL(value_suffix, ValueType, u32, uint32_t)                                           \
    GENERATE_RUNEND_KERNEL(value_suffix, ValueType, u64, uint64_t)

GENERATE_RUNEND_KERNELS_FOR_VALUE(u8, uint8_t)
GENERATE_RUNEND_KERNELS_FOR_VALUE(i8, int8_t)
GENERATE_RUNEND_KERNELS_FOR_VALUE(u16, uint16_t)
GENERATE_RUNEND_KERNELS_FOR_VALUE(i16, int16_t)
GENERATE_RUNEND_KERNELS_FOR_VALUE(u32, uint32_t)
GENERATE_RUNEND_KERNELS_FOR_VALUE(i32, int32_t)
GENERATE_RUNEND_KERNELS_FOR_VALUE(u64, uint64_t)
GENERATE_RUNEND_KERNELS_FOR_VALUE(i64, int64_t)
GENERATE_RUNEND_KERNELS_FOR_VALUE(f16, __half)
GENERATE_RUNEND_KERNELS_FOR_VALUE(f32, float)
GENERATE_RUNEND_KERNELS_FOR_VALUE(f64, double)
