// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Dynamic dispatch plan shared between the host plan builder and the GPU kernel.
///
/// The plan builder walks an encoding tree and emits a linear sequence of
/// stages. The kernel executes stages in order within a single launch.
///
/// Shared memory: The plan builder bump-allocates shared memory regions for
/// each input stage's output. The output stage (last) is placed after all
/// input stages. Since all regions must coexist for the output stage to
/// reference, the total shared memory is the end of whichever region extends
/// furthest, in elements, times `sizeof(T)`.
///
/// Example: RunEnd(ends=FoR(BitPacked), values=FoR(BitPacked)) with 100 runs
///
///   Stage 0 (input):  BITUNPACK(7) → FoR(0)            → smem[0..100)      // run ends
///   Stage 1 (input):  BITUNPACK(10) → FoR(50)          → smem[100..200)    // run values
///   Stage 2 (output): RUNEND(ends=0, values=100)        → smem[200..1224)  // resolved
///
///   shared_mem_bytes = (200 + 1024) * sizeof(T)

#pragma once

#include <stdint.h>

/// Elements processed per CUDA block.
#define ELEMENTS_PER_BLOCK 2048

/// Shared memory tile size for the output stage. Each block decompresses
/// ELEMENTS_PER_BLOCK elements but only holds SMEM_TILE_SIZE in smem at a
/// time — each tile is written to global memory before the next is decoded
/// into the same region. Input stages cannot tile because their outputs must
/// remain accessible for random access (e.g., dictionary lookup, run-end
/// binary search). Smaller tiles reduce smem per block, improving occupancy.
#define SMEM_TILE_SIZE 1024

#ifdef __cplusplus
extern "C" {
#endif

/// Parameters for source ops, which decode data into a stage's shared memory region.
union SourceParams {
    /// Unpack bit-packed data using FastLanes layout.
    struct BitunpackParams {
        uint8_t bit_width;
        uint32_t element_offset; // Sub-byte offset
    } bitunpack;

    /// Copy elements verbatim from global memory to shared memory.
    /// The input pointer is pre-adjusted on the host to account for slicing.
    struct LoadParams {
        uint8_t _placeholder;
    } load;

    /// Decode run-end encoding using ends and values already in shared memory.
    struct RunEndParams {
        uint32_t ends_smem_offset;   // element offset to decoded ends in smem
        uint32_t values_smem_offset; // element offset to decoded values in smem
        uint64_t num_runs;
        uint64_t offset;
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

#define MAX_SCALAR_OPS 4

/// A single stage in the dispatch plan.
///
/// Each stage is a pipeline (source + scalar ops) that writes decoded data
/// into a shared memory region at `smem_offset`. Input stage outputs persist
/// in smem so the output stage can reference them (via DICT or RUNEND offsets).
struct Stage {
    uint64_t input_ptr;   // global memory pointer to this stage's encoded input
    uint32_t smem_offset; // element offset within dynamic shared memory for output
    uint32_t len;         // number of elements this stage produces

    struct SourceOp source;
    uint8_t num_scalar_ops;
    struct ScalarOp scalar_ops[MAX_SCALAR_OPS];
};

#define MAX_STAGES 4

/// Dispatch plan: a sequence of stages.
///
/// The plan builder walks the encoding tree recursively, emitting an input
/// stage each time it encounters a child array that needs to live in shared
/// memory (e.g., dictionary values, run-end endpoints). Shared memory
/// offsets are assigned with a simple bump allocator.
///
/// The last stage is the output pipeline which directly writes to global memory.
struct DynamicDispatchPlan {
    uint8_t num_stages;
    struct Stage stages[MAX_STAGES];
};

#ifdef __cplusplus
}
#endif
