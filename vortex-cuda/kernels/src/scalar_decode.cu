// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Scalar decode kernel: Applies a sequence of element-wise operations 
// in a single kernel launch. Instead of launching N separate kernels, 
// this kernel loads each element once from global memory, applies all 
// operations in sequence.

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "types.cuh"

// ---------------------------------------------------------------------------
// Decode program types
// ---------------------------------------------------------------------------

/// Element-wise operation identifier.
enum ScalarOp : uint32_t {
    /// Frame-of-Reference: value += param (param interpreted as T)
    SCALAR_OP_FOR_ADD = 0,

    /// Zigzag decode: (value >> 1) ^ -(value & 1)
    SCALAR_OP_ZIGZAG = 1,
};

/// A decode operation.
struct DecodeOp {
    ScalarOp op;
    uint32_t _pad;
    uint64_t param;
};

constexpr uint32_t MAX_SMEM_OPS = 32;

/// Operation dispatch
template<typename T>
__device__ __forceinline__ T apply_op(T value, const DecodeOp& op) {
    switch (op.op) {
        case SCALAR_OP_FOR_ADD:
            return value + static_cast<T>(op.param);
        case SCALAR_OP_ZIGZAG:
            return (value >> 1) ^ static_cast<T>(-(value & 1));
        default:
            return value;
    }
}

template<typename T>
__device__ void scalar_decode_impl(
    T *__restrict values,
    uint64_t array_len,
    const DecodeOp *__restrict ops,
    uint32_t num_ops
) {
    assert(num_ops <= MAX_SMEM_OPS);

    // Cache ops in shared memory.
    __shared__ DecodeOp smem_ops[MAX_SMEM_OPS];
    if (threadIdx.x < num_ops) {
        smem_ops[threadIdx.x] = ops[threadIdx.x];
    }
    __syncthreads();

    constexpr uint32_t ELEMENTS_PER_BLOCK = 2048;
    constexpr uint32_t VALUES_PER_LOOP = 16 / sizeof(T);

    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * ELEMENTS_PER_BLOCK;
    const uint64_t block_end = min(block_start + ELEMENTS_PER_BLOCK, array_len);

    const uint64_t tile_size = blockDim.x * VALUES_PER_LOOP;
    const uint64_t num_full_tiles = (block_end - block_start) / tile_size;

    for (uint64_t tile = 0; tile < num_full_tiles; ++tile) {
        const uint64_t tile_base = block_start + tile * tile_size;
        T regs[VALUES_PER_LOOP];
        #pragma unroll
        for (uint32_t idx = 0; idx < VALUES_PER_LOOP; ++idx) {
            regs[idx] = values[tile_base + idx * blockDim.x + threadIdx.x];
        }

        // Operate on values in registers. This is faster than a coalesced
        // one-element-per-thread loop because the tiled load puts independent
        // values in flight simultaneously, enabling instruction-level parallelism.
        for (uint32_t op_idx = 0; op_idx < num_ops; op_idx++) {
            const DecodeOp& decode_op = smem_ops[op_idx];
            #pragma unroll
            for (uint32_t i = 0; i < VALUES_PER_LOOP; ++i) {
                regs[i] = apply_op(regs[i], decode_op);
            }
        }

        #pragma unroll
        for (uint32_t i = 0; i < VALUES_PER_LOOP; ++i) {
            values[tile_base + i * blockDim.x + threadIdx.x] = regs[i];
        }
    }

    // Handle remaining elements that were not part of a full tile.
    const uint64_t remaining_start = block_start + num_full_tiles * tile_size;
    for (uint64_t idx = remaining_start + threadIdx.x;
         idx < block_end;
         idx += blockDim.x) {
        T val = values[idx];
        for (uint32_t op_idx = 0; op_idx < num_ops; op_idx++) {
            val = apply_op(val, smem_ops[op_idx]);
        }
        values[idx] = val;
    }
}

#define GENERATE_SCALAR_DECODE_KERNEL(suffix, Type)     \
extern "C" __global__ void scalar_decode_##suffix(      \
    Type *__restrict values,                            \
    uint64_t array_len,                                 \
    const DecodeOp *__restrict ops,                     \
    uint32_t num_ops                                    \
) {                                                     \
    scalar_decode_impl<Type>(values, array_len, ops, num_ops); \
}

FOR_EACH_INTEGER(GENERATE_SCALAR_DECODE_KERNEL)
