// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// GPU kernel that decompresses a Vortex encoding tree in a single launch via dynamic dispatch.
///
/// Stages communicate through shared memory: early input stages populate
/// persistent smem regions (e.g., dictionary values, run-end endpoints) that
/// later stages reference via smem offsets.
///
/// The final output stage writes directly to global memory instead of back
/// to shared memory. Shared memory is dynamically sized at launch time to
/// fit all intermediate buffers that must coexist simultaneously.

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include "bit_unpack.cuh"
#include "dynamic_dispatch.h"
#include "types.cuh"

/// Binary search for first element strictly greater than value.
template <typename T>
__device__ inline uint64_t upper_bound(const T *data, uint64_t len, uint64_t value) {
    auto it = thrust::upper_bound(thrust::seq, data, data + len, value);
    return it - data;
}

/// Executes a source operation to fill a shared memory region with decoded data.
///
/// This function handles the first phase of each stage's pipeline. It reads
/// compressed or raw data from global memory and writes decoded elements into
/// the stage's shared memory region.
///
/// @param input        Global memory pointer to the stage's encoded input data
/// @param smem_output  Shared memory pointer where decoded elements are written
/// @param chunk_start  Starting index of the chunk to process (block-relative for output stage)
/// @param chunk_len    Number of elements to produce (may be < ELEMENTS_PER_BLOCK for tail blocks)
/// @param source_op    Source operation descriptor (BITUNPACK, LOAD, or RUNEND)
/// @param smem_base    Base of the entire dynamic shared memory pool, used by RUNEND
///                     to resolve offsets to ends/values decoded by earlier stages
template <typename T>
__device__ inline void dynamic_source_op(const T *__restrict input,
                                         T *__restrict &smem_output,
                                         uint64_t chunk_start,
                                         uint32_t chunk_len,
                                         const struct SourceOp &source_op,
                                         T *__restrict smem_base) {
    constexpr uint32_t T_BITS = sizeof(T) * 8;

    switch (source_op.op_code) {
    case SourceOp::BITUNPACK: {
        constexpr uint32_t FL_CHUNK_SIZE = 1024;
        constexpr uint32_t LANES_PER_FL_BLOCK = FL_CHUNK_SIZE / T_BITS;
        const uint32_t bit_width = source_op.params.bitunpack.bit_width;
        const uint32_t packed_words_per_fl_block = LANES_PER_FL_BLOCK * bit_width;

        const uint32_t element_offset = source_op.params.bitunpack.element_offset;
        const uint32_t smem_within_offset = (chunk_start + element_offset) % FL_CHUNK_SIZE;
        const uint64_t first_fl_block = (chunk_start + element_offset) / FL_CHUNK_SIZE;

        // FL blocks must divide evenly. Otherwise, the last unpack would overflow smem.
        static_assert((ELEMENTS_PER_BLOCK % FL_CHUNK_SIZE) == 0);

        const auto div_ceil = [](auto a, auto b) {
            return (a + b - 1) / b;
        };
        const uint32_t num_fl_chunks = div_ceil(chunk_len + smem_within_offset, FL_CHUNK_SIZE);

        for (uint32_t chunk_idx = 0; chunk_idx < num_fl_chunks; ++chunk_idx) {
            const T *packed_chunk = input + (first_fl_block + chunk_idx) * packed_words_per_fl_block;
            T *smem_lane = smem_output + chunk_idx * FL_CHUNK_SIZE;
            // Distribute unpacking across threads via lane-wise decomposition.
            for (uint32_t lane = threadIdx.x; lane < LANES_PER_FL_BLOCK; lane += blockDim.x) {
                bit_unpack_lane<T>(packed_chunk, smem_lane, 0, lane, bit_width);
            }
        }
        smem_output += smem_within_offset;
        return;
    }

    case SourceOp::LOAD: {
        // Copy elements verbatim from global memory into shared memory.
        for (uint32_t i = threadIdx.x; i < chunk_len; i += blockDim.x) {
            smem_output[i] = input[chunk_start + i];
        }
        return;
    }

    case SourceOp::RUNEND: {
        // Ends and values were decoded into shared memory by earlier stages.
        const T *ends = &smem_base[source_op.params.runend.ends_smem_offset];
        const T *values = &smem_base[source_op.params.runend.values_smem_offset];
        const uint64_t num_runs = source_op.params.runend.num_runs;
        const uint64_t offset = source_op.params.runend.offset;

        // Each thread binary-searches for its first position's run, then
        // forward-scans for subsequent positions. Strided positions are
        // monotonically increasing per thread, so current_run only advances.
        uint64_t current_run = upper_bound(ends, num_runs, chunk_start + threadIdx.x + offset);

        for (uint32_t i = threadIdx.x; i < chunk_len; i += blockDim.x) {
            uint64_t pos = chunk_start + i + offset;

            while (current_run < num_runs && static_cast<uint64_t>(ends[current_run]) <= pos) {
                current_run++;
            }

            smem_output[i] = values[min(current_run, num_runs - 1)];
        }
        return;
    }

    default:
        __builtin_unreachable();
    }
}

/// Applies a single scalar operation to N values in registers.
///
/// Scalar operations are applied element-wise after the source op fills shared
/// memory. All ops compose fluently in any order: FoR adds a constant, ZigZag
/// decodes signed integers, ALP decodes floats, and DICT gathers from a
/// dictionary in shared memory.
///
/// @param values    Array of N values to transform in-place
/// @param op        The scalar operation descriptor
/// @param smem_base Base of dynamic shared memory pool (used by DICT to resolve offsets)
template <typename T, uint32_t N>
__device__ inline void apply_scalar_op(T *values, const struct ScalarOp &op, T *__restrict smem_base) {
    switch (op.op_code) {
    case ScalarOp::FOR: {
        const T ref = static_cast<T>(op.params.frame_of_ref.reference);
        // clang-format off
        #pragma unroll
        // clang-format on
        for (uint32_t i = 0; i < N; ++i) {
            values[i] += ref;
        }
        break;
    }
    case ScalarOp::ZIGZAG: {
        // clang-format off
        #pragma unroll
        // clang-format on
        for (uint32_t i = 0; i < N; ++i) {
            values[i] = (values[i] >> 1) ^ static_cast<T>(-(values[i] & 1));
        }
        break;
    }
    case ScalarOp::ALP: {
        const float f = op.params.alp.f;
        const float e = op.params.alp.e;
        // clang-format off
        #pragma unroll
        // clang-format on
        for (uint32_t i = 0; i < N; ++i) {
            float result = static_cast<float>(static_cast<int32_t>(values[i])) * f * e;
            values[i] = static_cast<T>(__float_as_uint(result));
        }
        break;
    }
    case ScalarOp::DICT: {
        const T *dict_values = &smem_base[op.params.dict.values_smem_offset];
        // clang-format off
        #pragma unroll
        // clang-format on
        for (uint32_t i = 0; i < N; ++i) {
            values[i] = dict_values[static_cast<uint32_t>(values[i])];
        }
        break;
    }
    default:
        __builtin_unreachable();
    }
}

/// Store policy for global memory writes.
enum class StorePolicy {
    /// Default write-back stores — data stays in L2 cache.
    WRITEBACK,
    /// Streaming stores (`__stcs` / `st.cs`) — hint L2 to evict early.
    /// Use for write-only output data that this kernel will not read again.
    /// `__stcs` is a regular synchronous store (not async like `cp.async`),
    /// so the existing `__syncthreads()` barrier after each tile is
    /// sufficient for ordering.
    STREAMING,
};

/// Reads values from `smem_input`, applies scalar ops in registers, and
/// writes results to `write_dest` at `write_offset`.
template <typename T, StorePolicy S>
__device__ void apply_scalar_ops(const T *__restrict smem_input,
                                 T *__restrict write_dest,
                                 uint64_t write_offset,
                                 uint32_t chunk_len,
                                 uint8_t num_scalar_ops,
                                 const struct ScalarOp *scalar_ops,
                                 T *__restrict smem_base) {
    constexpr uint32_t VALUES_PER_LOOP = 64 / sizeof(T);
    const uint32_t tile_size = blockDim.x * VALUES_PER_LOOP;
    const uint32_t num_full_tiles = chunk_len / tile_size;

    // Each thread holds multiple values in registers for instruction-level
    // parallelism, hiding pipeline latency between independent operations.
    for (uint32_t tile = 0; tile < num_full_tiles; ++tile) {
        const uint32_t tile_base = tile * tile_size;
        T values[VALUES_PER_LOOP];

        // clang-format off
        #pragma unroll
        // clang-format on
        for (uint32_t idx = 0; idx < VALUES_PER_LOOP; ++idx) {
            values[idx] = smem_input[tile_base + idx * blockDim.x + threadIdx.x];
        }

        for (uint8_t op_idx = 0; op_idx < num_scalar_ops; ++op_idx) {
            apply_scalar_op<T, VALUES_PER_LOOP>(values, scalar_ops[op_idx], smem_base);
        }

        // clang-format off
        #pragma unroll
        // clang-format on
        for (uint32_t idx = 0; idx < VALUES_PER_LOOP; ++idx) {
            if constexpr (S == StorePolicy::STREAMING) {
                __stcs(&write_dest[write_offset + tile_base + idx * blockDim.x + threadIdx.x], values[idx]);
            } else {
                write_dest[write_offset + tile_base + idx * blockDim.x + threadIdx.x] = values[idx];
            }
        }
    }

    const uint32_t rem_start = num_full_tiles * tile_size;
    for (uint32_t elem_idx = rem_start + threadIdx.x; elem_idx < chunk_len; elem_idx += blockDim.x) {
        T val = smem_input[elem_idx];
        for (uint8_t op_idx = 0; op_idx < num_scalar_ops; ++op_idx) {
            apply_scalar_op<T, 1>(&val, scalar_ops[op_idx], smem_base);
        }
        if constexpr (S == StorePolicy::STREAMING) {
            __stcs(&write_dest[write_offset + elem_idx], val);
        } else {
            write_dest[write_offset + elem_idx] = val;
        }
    }
}

/// Decodes and transforms a stage's data through shared memory, writing
/// final results to `write_dest` at `write_offset`. Input stages write
/// back to smem; the output stage writes to global memory.
template <typename T, StorePolicy S>
__device__ void execute_stage(const struct Stage &stage,
                              T *__restrict smem_base,
                              uint64_t chunk_start,
                              uint32_t chunk_len,
                              T *__restrict write_dest,
                              uint64_t write_offset) {
    T *smem_output = &smem_base[stage.smem_offset];

    dynamic_source_op<T>(reinterpret_cast<const T *>(stage.input_ptr),
                         smem_output,
                         chunk_start,
                         chunk_len,
                         stage.source,
                         smem_base);
    __syncthreads();

    apply_scalar_ops<T, S>(smem_output,
                           write_dest,
                           write_offset,
                           chunk_len,
                           stage.num_scalar_ops,
                           stage.scalar_ops,
                           smem_base);
    __syncthreads();
}

/// Computes the number of elements to process in an output tile.
///
/// Each tile decodes exactly one FL block == SMEM_TILE_SIZE elements into
/// shared memory. In case BITUNPACK is sliced, we need to account for the
/// sub-byte element offset.
__device__ inline uint32_t output_tile_len(const struct Stage &stage, uint32_t block_len, uint32_t tile_off) {
    const uint32_t element_offset = (tile_off == 0 && stage.source.op_code == SourceOp::BITUNPACK)
                                        ? stage.source.params.bitunpack.element_offset
                                        : 0;
    return min(SMEM_TILE_SIZE - element_offset, block_len - tile_off);
}

/// Entry point of the dynamic dispatch kernel.
///
/// Executes the plan's stages in order:
///   1. Input stages populate shared memory with intermediate data
///      for the output stage to reference.
///   2. The output stage decodes the root array and writes directly to
///      global memory.
///
/// @param output    Global memory output buffer
/// @param array_len Total number of elements to produce
/// @param plan      Device pointer to the dispatch plan
template <typename T>
__device__ void dynamic_dispatch(T *__restrict output,
                                 uint64_t array_len,
                                 const struct DynamicDispatchPlan *__restrict plan) {

    // Dynamically-sized shared memory: The host computes the exact byte count
    // needed to hold all stage outputs that must coexist simultaneously, and
    // passes the count at kernel launch (see DynamicDispatchPlan::shared_mem_bytes).
    extern __shared__ char smem_bytes[];
    T *smem_base = reinterpret_cast<T *>(smem_bytes);

    __shared__ struct DynamicDispatchPlan smem_plan;
    if (threadIdx.x == 0) {
        smem_plan = *plan;
    }
    __syncthreads();

    const uint8_t last = smem_plan.num_stages - 1;

    // Input stages: Decode inputs into smem regions.
    for (uint8_t i = 0; i < last; ++i) {
        const struct Stage &stage = smem_plan.stages[i];
        T *smem_output = &smem_base[stage.smem_offset];
        execute_stage<T, StorePolicy::WRITEBACK>(stage, smem_base, 0, stage.len, smem_output, 0);
    }

    const struct Stage &output_stage = smem_plan.stages[last];
    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * ELEMENTS_PER_BLOCK;
    const uint64_t block_end = min(block_start + ELEMENTS_PER_BLOCK, array_len);
    const uint32_t block_len = static_cast<uint32_t>(block_end - block_start);

    for (uint32_t tile_off = 0; tile_off < block_len;) {
        const uint32_t tile_len = output_tile_len(output_stage, block_len, tile_off);
        execute_stage<T, StorePolicy::STREAMING>(output_stage,
                                                 smem_base,
                                                 block_start + tile_off,
                                                 tile_len,
                                                 output,
                                                 block_start + tile_off);
        tile_off += tile_len;
    }
}

/// Generates a dynamic dispatch kernel entry point for each unsigned integer type.
#define GENERATE_DYNAMIC_DISPATCH_KERNEL(suffix, Type)                                                       \
    extern "C" __global__ void dynamic_dispatch_##suffix(                                                    \
        Type *__restrict output,                                                                             \
        uint64_t array_len,                                                                                  \
        const struct DynamicDispatchPlan *__restrict plan) {                                                 \
        dynamic_dispatch<Type>(output, array_len, plan);                                                     \
    }

FOR_EACH_UNSIGNED_INT(GENERATE_DYNAMIC_DISPATCH_KERNEL)
