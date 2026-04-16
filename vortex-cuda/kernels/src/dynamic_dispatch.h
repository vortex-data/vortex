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
///
/// ## Per-op type tracking
///
/// Each source op and scalar op may produce a different PType than its input.
/// For example, DICT transforms codes (e.g. u8) into values (e.g. f32), and
/// ALP transforms encoded integers (i32) into floats (f32).
///
/// `PTypeTag` is a compact enum that identifies the primitive type at each
/// point in the pipeline. The kernel uses it to dispatch typed memory
/// operations (LOAD, BITUNPACK) and cross-stage references (DICT gather,
/// RUNEND lookup) at the correct element width and signedness.

#pragma once

#include <stdint.h>

/// Compact tag identifying a Vortex PType for GPU dispatch.
///
/// NOTE: These values intentionally skip F16 (which Rust PType includes),
/// so numeric values do NOT match Rust PType directly. The Rust
/// `ptype_to_tag()` function handles the mapping at plan-build time.
///
/// The kernel uses this to:
/// - Select the correct element width for LOAD / BITUNPACK source ops.
/// - Index shared memory at the correct stride for DICT / RUNEND cross-stage
///   references.
/// - Distinguish int vs float for ALP decode.
enum PTypeTag : uint8_t {
    PTYPE_U8 = 0,
    PTYPE_U16 = 1,
    PTYPE_U32 = 2,
    PTYPE_U64 = 3,
    PTYPE_I8 = 4,
    PTYPE_I16 = 5,
    PTYPE_I32 = 6,
    PTYPE_I64 = 7,
    PTYPE_F32 = 8,
    PTYPE_F64 = 9,
};

/// Return the unsigned equivalent of a PTypeTag (same width).
#ifdef __cplusplus
#ifdef __CUDACC__
#define PTYPE_HOST_DEVICE __host__ __device__
#else
#define PTYPE_HOST_DEVICE
#endif
PTYPE_HOST_DEVICE constexpr PTypeTag ptype_to_unsigned(PTypeTag tag) {
    switch (tag) {
    case PTYPE_I8:
        return PTYPE_U8;
    case PTYPE_I16:
        return PTYPE_U16;
    case PTYPE_I32:
    case PTYPE_F32:
        return PTYPE_U32;
    case PTYPE_I64:
    case PTYPE_F64:
        return PTYPE_U64;
    default:
        return tag;
    }
}
#endif

/// Number of threads per CUDA block.
#define BLOCK_SIZE 64

/// Elements processed per CUDA block.
#define ELEMENTS_PER_BLOCK 2048

/// Each tile is flushed to global before the next is decoded.
#define SMEM_TILE_SIZE 1024

/// Fixed shared memory declared in the kernel (bytes), excluded from
/// the dynamic shared memory budget. Accounts for
/// `runend_cursors[BLOCK_SIZE]` — one uint64_t cursor per thread.
///
/// Uses a literal (64 * 8 = 512) instead of `BLOCK_SIZE * sizeof(uint64_t)`
/// so that bindgen can export it as a Rust constant (bindgen cannot evaluate
/// expressions involving other macros or sizeof).
#define KERNEL_FIXED_SHARED_BYTES 512

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
    ///
    /// The smem offsets are byte offsets so that ends and values can have
    /// different element widths.
    struct RunEndParams {
        uint32_t ends_smem_byte_offset;   // byte offset to decoded ends in smem
        uint32_t values_smem_byte_offset; // byte offset to decoded values in smem
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
///
/// Each scalar op declares its `output_ptype` — the PType of the values it
/// produces. Most ops preserve the input type (FOR, ZIGZAG), but some
/// change it:
///   - ALP: encoded int → float (e.g. i32 → f32)
///   - DICT: codes type → values type (e.g. u8 → u32)
///
/// The plan builder uses `output_ptype` to determine the element width
/// for shared memory allocation and to propagate type information
/// through the pipeline.
union ScalarParams {
    struct FoRParams {
        uint64_t reference;
    } frame_of_ref;

    struct AlpParams {
        float f;
        float e;
    } alp;

    /// Dictionary gather: use current value as index into decoded values in smem.
    ///
    /// `values_smem_byte_offset` is a byte offset so that values can have
    /// a different element width than the codes. The plan builder uses
    /// `output_ptype` (on the enclosing ScalarOp) to determine the values'
    /// element type.
    struct DictParams {
        uint32_t values_smem_byte_offset; // byte offset to decoded dict values in smem
    } dict;
};

struct ScalarOp {
    enum ScalarOpCode { FOR, ZIGZAG, ALP, DICT } op_code;
    /// The PType this op produces. For type-preserving ops (FOR, ZIGZAG)
    /// this equals the input PType. For type-changing ops (ALP, DICT) this
    /// is the new output PType.
    enum PTypeTag output_ptype;
    union ScalarParams params;
};

/// Packed stage header, followed by `num_scalar_ops` inline ScalarOps.
///
/// `source_ptype` identifies the PType that the source op (BITUNPACK, LOAD,
/// etc.) produces. This may differ from the output PType when scalar ops
/// change the type (e.g. DICT transforms u8 codes into u32 values).
///
/// `smem_byte_offset` is a byte offset into the dynamic shared memory
/// pool so that stages with different element widths can coexist.
struct PackedStage {
    uint64_t input_ptr;        // global memory pointer to this stage's encoded input
    uint32_t smem_byte_offset; // byte offset within dynamic shared memory for output
    uint32_t len;              // number of elements this stage produces

    struct SourceOp source;
    uint8_t num_scalar_ops;
    enum PTypeTag source_ptype; // PType produced by the source op
};

/// Header for the packed plan byte buffer.
struct __attribute__((aligned(8))) PlanHeader {
    uint8_t num_stages;
    enum PTypeTag output_ptype; // PType of the final output array
    uint16_t plan_size_bytes;   // total size of the packed plan including this header
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
///
/// `source_ptype` is the PType produced by the source op. Scalar ops may
/// change the type; the final output PType is given by the last scalar op's
/// `output_ptype` (or `source_ptype` if there are no scalar ops).
struct Stage {
    uint64_t input_ptr;                // encoded input in global memory
    uint32_t smem_byte_offset;         // byte offset within dynamic shared memory
    uint32_t len;                      // elements produced
    enum PTypeTag source_ptype;        // PType produced by the source op
    struct SourceOp source;            // source decode op
    uint8_t num_scalar_ops;            // number of scalar ops
    const struct ScalarOp *scalar_ops; // scalar decode ops
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
        .smem_byte_offset = packed_stage->smem_byte_offset,
        .len = packed_stage->len,
        .source_ptype = packed_stage->source_ptype,
        .source = packed_stage->source,
        .num_scalar_ops = packed_stage->num_scalar_ops,
        .scalar_ops = ops,
    };
}

#endif
