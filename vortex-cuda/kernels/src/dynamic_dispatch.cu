// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Dynamic dispatch kernel: decodes an array by applying a sequence of operations
// in a single kernel launch. The source op fills shared memory (e.g. bitunpack),
// then scalar ops are applied element-wise in registers (e.g. FoR, zigzag, ALP).

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "bit_unpack_8.cu"
#include "bit_unpack_16.cu"
#include "bit_unpack_32.cu"
#include "bit_unpack_64.cu"
#include "dynamic_dispatch.h"
#include "types.cuh"

constexpr uint32_t ELEMENTS_PER_BLOCK = 2048;

template <typename T>
__device__ __forceinline__ void bitunpack_lane_to_smem(const T *__restrict packed_chunk, T *__restrict smem,
                                                       unsigned int lane, uint32_t bit_width);

#define BITUNPACK_LANE(bits, UType, Type)                                                                    \
    template <>                                                                                              \
    __device__ __forceinline__ void bitunpack_lane_to_smem<Type>(const Type *in, Type *out,                  \
                                                                 unsigned int lane, uint32_t bw) {           \
        bit_unpack_##bits##_lane(reinterpret_cast<const UType *>(in), reinterpret_cast<UType *>(out), lane,  \
                                 bw);                                                                        \
    }

BITUNPACK_LANE(8, uint8_t, uint8_t)
BITUNPACK_LANE(8, uint8_t, int8_t)
BITUNPACK_LANE(16, uint16_t, uint16_t)
BITUNPACK_LANE(16, uint16_t, int16_t)
BITUNPACK_LANE(32, uint32_t, uint32_t)
BITUNPACK_LANE(32, uint32_t, int32_t)
BITUNPACK_LANE(64, uint64_t, uint64_t)
BITUNPACK_LANE(64, uint64_t, int64_t)

template <typename T>
__device__ __forceinline__ void dynamic_source_op(const T *__restrict input, T *__restrict smem,
                                               uint64_t chunk_start, uint32_t chunk_len,
                                               const struct SourceOp &source_op) {
    constexpr uint32_t T_BITS = sizeof(T) * 8;
    constexpr uint32_t FL_LANES = ELEMENTS_PER_BLOCK / T_BITS;

    switch (source_op.op_code) {
    case SourceOp::BITUNPACK: {
        constexpr uint32_t ELEMENTS_PER_FL_BLOCK = 1024;
        constexpr uint32_t LANES_PER_FL_BLOCK = ELEMENTS_PER_FL_BLOCK / T_BITS;
        const uint32_t bit_width = source_op.params.bitunpack.bit_width;
        const uint32_t packed_words_per_fl_block = LANES_PER_FL_BLOCK * bit_width;
        const uint64_t first_fl_block = chunk_start / ELEMENTS_PER_FL_BLOCK;

        #pragma unroll
        for (uint32_t blk = 0; blk < ELEMENTS_PER_BLOCK / ELEMENTS_PER_FL_BLOCK; ++blk) {
            const T *packed_fl = input + (first_fl_block + blk) * packed_words_per_fl_block;
            T *smem_fl = smem + blk * ELEMENTS_PER_FL_BLOCK;
            for (uint32_t lane = threadIdx.x; lane < LANES_PER_FL_BLOCK; lane += blockDim.x) {
                bitunpack_lane_to_smem<T>(packed_fl, smem_fl, lane, bit_width);
            }
        }
        break;
    }
    default: __builtin_unreachable();
    }
}

template <typename T>
__device__ __forceinline__ T dynamic_scalar_op(T value, const struct ScalarOp &op) {
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

#define GENERATE_DYNAMIC_DISPATCH_KERNEL(suffix, Type)                                                       \
    extern "C" __global__ void dynamic_dispatch_##suffix(const Type *__restrict input,                       \
                                                         Type *__restrict output, uint64_t array_len,        \
                                                         const struct DynamicDispatchPlan *__restrict plan) { \
        dynamic_dispatch_impl<Type>(input, output, array_len, plan);                                         \
    }

FOR_EACH_INTEGER(GENERATE_DYNAMIC_DISPATCH_KERNEL)
