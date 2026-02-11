// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Dynamic dispatch kernel: decodes an array by applying a sequence of operations
// in a single kernel launch. The first op may optionally be a "source" op, e.g. bitunpack.
// Subsequent transform ops are applied element-wise in registers.

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

constexpr uint8_t MAX_DECODE_OPS = 8;
constexpr uint32_t FL_CHUNK_SIZE = 1024;

__device__ __forceinline__ bool is_source_op(enum DynamicOpCode op) {
    return op == BITUNPACK;
}

template <typename T>
__device__ __forceinline__ T apply_scalar_op(T value, const DynamicOp &op) {
    switch (op.op) {
    case FOR: {
        return value + static_cast<T>(op.param);
    }
    case ZIGZAG: {
        return (value >> 1) ^ static_cast<T>(-(value & 1));
    }
    default:
        return value;
    }
}

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
__device__ __forceinline__ void source_fill_op(const T *__restrict input, T *__restrict smem,
                                               uint64_t chunk_start, uint32_t chunk_len,
                                               const DynamicOp &source_op) {
    constexpr uint32_t T_BITS = sizeof(T) * 8;
    constexpr uint32_t FL_LANES = FL_CHUNK_SIZE / T_BITS;

    switch (source_op.op) {
    case BITUNPACK: {
        const uint32_t bit_width = static_cast<uint32_t>(source_op.param);
        const uint32_t packed_words_per_chunk = FL_LANES * bit_width;
        const uint64_t chunk_idx = chunk_start / FL_CHUNK_SIZE;
        const T *packed_chunk = input + chunk_idx * packed_words_per_chunk;
        for (uint32_t lane = threadIdx.x; lane < FL_LANES; lane += blockDim.x) {
            bitunpack_lane_to_smem<T>(packed_chunk, smem, lane, bit_width);
        }
        break;
    }
    default:
        for (uint32_t elem_idx = threadIdx.x; elem_idx < chunk_len; elem_idx += blockDim.x) {
            smem[elem_idx] = input[chunk_start + elem_idx];
        }
        break;
    }
}

template <typename T>
__device__ void dynamic_dispatch_impl(const T *__restrict input, T *__restrict output, uint64_t array_len,
                                      const DynamicOp *__restrict ops, uint8_t num_ops) {
    assert(num_ops <= MAX_DECODE_OPS);

    constexpr uint32_t ELEMENTS_PER_BLOCK = 2048;
    constexpr uint32_t VALUES_PER_LOOP = 32 / sizeof(T);

    __shared__ DynamicOp smem_ops[MAX_DECODE_OPS];
    __shared__ T smem_values[FL_CHUNK_SIZE];

    // Cache ops in shared memory.
    if (threadIdx.x < num_ops) {
        smem_ops[threadIdx.x] = ops[threadIdx.x];
    }
    __syncthreads();

    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * ELEMENTS_PER_BLOCK;
    const uint64_t block_end = min(block_start + ELEMENTS_PER_BLOCK, array_len);

    for (uint64_t chunk_start = block_start; chunk_start < block_end; chunk_start += FL_CHUNK_SIZE) {
        const uint32_t chunk_len =
            static_cast<uint32_t>(min(static_cast<uint64_t>(FL_CHUNK_SIZE), block_end - chunk_start));

        source_fill_op<T>(input, smem_values, chunk_start, chunk_len, smem_ops[0]);
        __syncthreads();

        const uint32_t tile_size = blockDim.x * VALUES_PER_LOOP;
        const uint32_t num_full_tiles = chunk_len / tile_size;
        const uint8_t scalar_op_start_idx = is_source_op(smem_ops[0].op);

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

            for (uint8_t op_idx = scalar_op_start_idx; op_idx < num_ops; ++op_idx) {
                const DynamicOp &decode_op = smem_ops[op_idx];

                #pragma unroll
                for (uint32_t idx = 0; idx < VALUES_PER_LOOP; ++idx) {
                    values[idx] = apply_scalar_op(values[idx], decode_op);
                }
            }

            #pragma unroll
            for (uint32_t idx = 0; idx < VALUES_PER_LOOP; ++idx) {
                output[chunk_start + tile_base + idx * blockDim.x + threadIdx.x] = values[idx];
            }
        }

        // Handle remaining elements that were not part of a full tile.
        const uint32_t rem_start = num_full_tiles * tile_size;
        for (uint32_t elem_idx = rem_start + threadIdx.x; elem_idx < chunk_len; elem_idx += blockDim.x) {
            T val = smem_values[elem_idx];
            for (uint8_t op_idx = scalar_op_start_idx; op_idx < num_ops; ++op_idx) {
                val = apply_scalar_op(val, smem_ops[op_idx]);
            }
            output[chunk_start + elem_idx] = val;
        }

        __syncthreads();
    }
}

#define GENERATE_DYNAMIC_DISPATCH_KERNEL(suffix, Type)                                                       \
    extern "C" __global__ void dynamic_dispatch_##suffix(const Type *__restrict input,                       \
                                                         Type *__restrict output, uint64_t array_len,        \
                                                         const DynamicOp *__restrict ops, uint8_t num_ops) {  \
        dynamic_dispatch_impl<Type>(input, output, array_len, ops, num_ops);                                 \
    }

FOR_EACH_INTEGER(GENERATE_DYNAMIC_DISPATCH_KERNEL)
