// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Dynamic dispatch kernel: decodes an array by applying a sequence of operations
/// in a single kernel launch. The source op fills shared memory (e.g. bitunpack),
/// then scalar ops are applied element-wise in registers (e.g. FoR, zigzag, ALP).

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "bit_unpack.cuh"
#include "dynamic_dispatch.h"
#include "types.cuh"

constexpr uint32_t ELEMENTS_PER_BLOCK = 2048;

/// Executes the source operation (e.g., bitunpack) to fill shared memory with unpacked data.
///
/// This function handles the first phase of the dynamic dispatch pipeline. It reads compressed
/// data from global memory and decompresses it into shared memory, preparing it for subsequent
/// scalar operations.
///
/// # Parameters
///
/// * `input` - Pointer to the compressed input data in global memory
/// * `smem` - Pointer to shared memory buffer where unpacked data is written (size: ELEMENTS_PER_BLOCK)
/// * `chunk_start` - Starting index of the data chunk to process in the input array
/// * `chunk_len` - Number of elements in this chunk (may be less than ELEMENTS_PER_BLOCK for tail blocks)
/// * `source_op` - The source operation descriptor containing the operation type and parameters
template <typename T>
__device__ inline void dynamic_source_op(const T *__restrict input, T *__restrict smem,
                                               uint64_t chunk_start, uint32_t chunk_len,
                                               const struct SourceOp &source_op) {
    constexpr uint32_t T_BITS = sizeof(T) * 8;
    constexpr uint32_t FL_LANES = ELEMENTS_PER_BLOCK / T_BITS;

    switch (source_op.op_code) {
    case SourceOp::BITUNPACK: {
        constexpr uint32_t FL_CHUNK_SIZE = 1024;
        constexpr uint32_t LANES_PER_FL_BLOCK = FL_CHUNK_SIZE / T_BITS;
        const uint32_t bit_width = source_op.params.bitunpack.bit_width;
        const uint32_t packed_words_per_fl_block = LANES_PER_FL_BLOCK * bit_width;
        const uint64_t first_fl_block = chunk_start / FL_CHUNK_SIZE;

        // FL blocks must divide evenly. Otherwise, the last unpack would overflow `smem`.
        static_assert((ELEMENTS_PER_BLOCK % FL_CHUNK_SIZE) == 0);

        const auto div_ceil = [](auto a, auto b) { return (a + b - 1) / b; };
        const uint32_t num_fl_chunks = div_ceil(chunk_len, FL_CHUNK_SIZE);

        for (uint32_t chunk_idx = 0; chunk_idx < num_fl_chunks; ++chunk_idx) {
            const T *packed_chunk = input + (first_fl_block + chunk_idx) * packed_words_per_fl_block;
            T *smem_lane = smem + chunk_idx * FL_CHUNK_SIZE;
            // Distribute unpacking across threads via lane-wise decomposition.
            for (uint32_t lane = threadIdx.x; lane < LANES_PER_FL_BLOCK; lane += blockDim.x) {
                bit_unpack_lane<T>(packed_chunk, smem_lane, lane, bit_width);
            }
        }
        break;
    }
    default: __builtin_unreachable();
    }
}

/// Applies a single scalar operation to a value.
///
/// Scalar operations are applied element-wise after unpacking, operating on
/// values in registers.
///
/// # Parameters
///
/// * `value` - The input value to be transformed
/// * `op` - The scalar operation descriptor containing the operation and parameters
///
/// # Returns
///
/// The transformed value after applying the scalar operation.
template <typename T>
__device__ inline T dynamic_scalar_op(T value, const struct ScalarOp &op) {
    switch (op.op_code) {
    case ScalarOp::FOR: {
        return value + static_cast<T>(op.params.frame_of_ref.reference);
    }
    case ScalarOp::ZIGZAG: {
        return (value >> 1) ^ static_cast<T>(-(value & 1));
    }
    case ScalarOp::ALP: {
        float result = static_cast<float>(static_cast<int32_t>(value)) * op.params.alp.f * op.params.alp.e;
        return static_cast<T>(__float_as_uint(result));
    }
    default: __builtin_unreachable();
    }
}

/// Entry point of the dynamic dispatch kernel.
///
/// Unpacks compressed data from global memory into shared memory, applies a
/// sequence of scalar operations (e.g., FoR, zigzag, ALP) to each element while
/// holding values in registers, and writes decoded results back to global memory.
///
/// # Parameters
///
/// * `input` - Compressed input data
/// * `output` - Output buffer
/// * `array_len` - Total number of elements
/// * `plan` - Operation sequence to apply
template <typename T>
__device__ void dynamic_dispatch_impl(const T *__restrict input, T *__restrict output, uint64_t array_len,
                                      const struct DynamicDispatchPlan *__restrict plan) {
    constexpr uint32_t VALUES_PER_LOOP = 32 / sizeof(T);

    __shared__ struct DynamicDispatchPlan smem_plan;
    __shared__ T smem_values[ELEMENTS_PER_BLOCK];

    // Cache the plan in shared memory.
    if (threadIdx.x == 0) smem_plan = *plan;
    __syncthreads();

    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * ELEMENTS_PER_BLOCK;
    const uint64_t block_end = min(block_start + ELEMENTS_PER_BLOCK, array_len);
    const uint32_t block_len = static_cast<uint32_t>(block_end - block_start);

    dynamic_source_op<T>(input, smem_values, block_start, block_len, smem_plan.source);
    __syncthreads();

    const uint32_t tile_size = blockDim.x * VALUES_PER_LOOP;
    const uint32_t num_full_tiles = block_len / tile_size;

    for (uint32_t tile = 0; tile < num_full_tiles; ++tile) {
        const uint32_t tile_base = tile * tile_size;

        // Operate on values in registers. This is faster than a coalesced
        // one-element-per-thread loop as it enables better instruction-level
        // parallelism.
        T values[VALUES_PER_LOOP];

        #pragma unroll
        for (uint32_t idx = 0; idx < VALUES_PER_LOOP; ++idx) {
            values[idx] = smem_values[tile_base + idx * blockDim.x + threadIdx.x];
        }

        for (uint8_t op_idx = 0; op_idx < smem_plan.num_scalar_ops; ++op_idx) {
            const struct ScalarOp &scalar_op = smem_plan.scalar_ops[op_idx];

            #pragma unroll
            for (uint32_t idx = 0; idx < VALUES_PER_LOOP; ++idx) {
                values[idx] = dynamic_scalar_op(values[idx], scalar_op);
            }
        }

        #pragma unroll
        for (uint32_t idx = 0; idx < VALUES_PER_LOOP; ++idx) {
            output[block_start + tile_base + idx * blockDim.x + threadIdx.x] = values[idx];
        }
    }

    // Handle remaining elements that were not part of a full tile.
    const uint32_t rem_start = num_full_tiles * tile_size;
    for (uint32_t elem_idx = rem_start + threadIdx.x; elem_idx < block_len; elem_idx += blockDim.x) {
        T val = smem_values[elem_idx];
        for (uint8_t op_idx = 0; op_idx < smem_plan.num_scalar_ops; ++op_idx) {
            val = dynamic_scalar_op(val, smem_plan.scalar_ops[op_idx]);
        }
        output[block_start + elem_idx] = val;
    }
}

/// Generates a dynamic dispatch kernel for the specific type.
///
/// Creates a CUDA kernel entry point by instantiating `dynamic_dispatch_impl` for the given type.
#define GENERATE_DYNAMIC_DISPATCH_KERNEL(suffix, Type)                                                       \
    extern "C" __global__ void dynamic_dispatch_##suffix(const Type *__restrict input,                       \
                                                         Type *__restrict output, uint64_t array_len,        \
                                                         const struct DynamicDispatchPlan *__restrict plan) { \
        dynamic_dispatch_impl<Type>(input, output, array_len, plan);                                         \
    }

FOR_EACH_UNSIGNED_INT(GENERATE_DYNAMIC_DISPATCH_KERNEL)
