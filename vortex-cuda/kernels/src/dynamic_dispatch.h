// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Dynamic dispatch plan shared between the host plan builder and the GPU kernel.
///
/// The plan builder walks an encoding tree and emits a linear sequence of
/// stages. The kernel executes stages in order within a single launch.
///
/// ## Stage plan
///
/// The plan is packed as a variable-length byte buffer.
///
/// Layout (contiguous bytes):
///   [PlanHeader]
///   [PackedStage 0][ScalarOp × N0]
///   [PackedStage 1][ScalarOp × N1]
///   ...

#pragma once

#include <stdint.h>

/// Elements processed per CUDA block.
#define ELEMENTS_PER_BLOCK 2048

/// Each tile is flushed to global before the next is decoded.
#define SMEM_TILE_SIZE 1024

#ifdef __cplusplus
extern "C" {
#endif

/// Parameters for source ops, which decode data into a stage's shared memory region.
union SourceParams {
    /// Unpack FastLanes bit-packed data.
    struct BitunpackParams {
        uint8_t bit_width;
        uint32_t element_offset; // Sub-byte offset
    } bitunpack;

    /// Copy from global to shared memory.
    struct LoadParams {
        uint8_t _placeholder;
    } load;

    /// Decode run-end encoding using ends and values already in shared memory.
    struct RunEndParams {
        uint32_t ends_smem_offset;   // element offset to decoded ends in smem
        uint32_t values_smem_offset; // element offset to decoded values in smem
        uint64_t num_runs;
        uint64_t offset; // slice offset into the run-end encoded array
    } runend;

    /// Generate a linear sequence: `value[i] = base + i * multiplier`.
    struct SequenceParams {
        int64_t base;
        int64_t multiplier;
    } sequence;
};

struct SourceOp {
    enum SourceOpCode { BITUNPACK, LOAD, RUNEND, SEQUENCE } op_code;
    union SourceParams params;
};

/// Scalar ops: element-wise transforms in registers.
/// All ops compose fluently in any order.
union ScalarParams {
    struct FoRParams {
        uint64_t reference;
    } frame_of_ref;

    struct AlpParams {
        float f;
        float e;
    } alp;

    /// Dictionary gather: use current value as index into decoded values in smem.
    struct DictParams {
        uint32_t values_smem_offset; // element offset to decoded dict values in smem
    } dict;
};

struct ScalarOp {
    enum ScalarOpCode { FOR, ZIGZAG, ALP, DICT } op_code;
    union ScalarParams params;
};

/// Packed stage header, followed by `num_scalar_ops` inline ScalarOps.
struct PackedStage {
    uint64_t input_ptr;   // global memory pointer to this stage's encoded input
    uint32_t smem_offset; // element offset within dynamic shared memory for output
    uint32_t len;         // number of elements this stage produces

    struct SourceOp source;
    uint8_t num_scalar_ops;
};

/// Header for the packed plan byte buffer.
struct __attribute__((aligned(8))) PlanHeader {
    uint8_t num_stages;
    uint16_t plan_size_bytes; // total size of the packed plan including this header
};

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

/// Stage parsed from the packed plan byte buffer.
///
/// Input stages decode data (e.g. dict values, run-end endpoints) into a
/// shared memory region for the output stage to reference. The output stage
/// decodes the root encoding and writes to global memory.
struct Stage {
    uint64_t input_ptr;                // encoded input in global memory
    uint32_t smem_offset;              // output offset in shared memory (elements)
    uint32_t len;                      // elements produced
    struct SourceOp source;            // source decode op
    uint8_t num_scalar_ops;            // number of scalar ops
    const struct ScalarOp *scalar_ops; // scalar deoode ops
};

/// Parse a single stage from the packed plan byte buffer and advance the cursor.
///
/// @param cursor  Pointer into the packed plan buffer, pointing at a PackedStage.
///                On return, advanced past this stage's ScalarOps.
/// @return        A Stage referencing data within the packed plan buffer.
__device__ inline Stage parse_stage(const uint8_t *&cursor) {
    const auto *packed_stage = reinterpret_cast<const struct PackedStage *>(cursor);
    cursor += sizeof(struct PackedStage);

    const auto *ops = reinterpret_cast<const struct ScalarOp *>(cursor);
    cursor += packed_stage->num_scalar_ops * sizeof(struct ScalarOp);

    return Stage {
        .input_ptr = packed_stage->input_ptr,
        .smem_offset = packed_stage->smem_offset,
        .len = packed_stage->len,
        .source = packed_stage->source,
        .num_scalar_ops = packed_stage->num_scalar_ops,
        .scalar_ops = ops,
    };
}

#endif
