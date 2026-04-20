// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// ═══════════════════════════════════════════════════════════════════════════
// Dynamic dispatch kernel
// ═══════════════════════════════════════════════════════════════════════════
//
// Vortex arrays are stored as nested encodings — e.g. ALP(FoR(BitPacked))
// or Dict(codes=BitPacked, values=FoR(BitPacked)). This kernel walks
// such a tree in a single launch by decomposing it into a linear sequence
// of stages described by a packed plan buffer on the device.
//
// Each block produces ELEMENTS_PER_BLOCK output elements. Input stages
// are fully decoded per block (every block independently decodes the
// complete dict values, run-end endpoints, etc. into its own shared
// memory).
//
// ## Pipeline
//
// Input stages run first: each decodes a dependency (dict values, run-end
// endpoints) into shared memory that the output stage later references via
// byte offsets for DICT gathers and RUNEND binary searches.
//
// The output stage then processes the full block through:
//
//   source_op → scalar_op (FoR/ZigZag/ALP/DICT) → streaming store
//
// in register batches of VALUES_PER_TILE (8 for u32) per thread.
//
// ## Source ops
//
// BITUNPACK  Cooperative FastLanes unpack into smem scratch, sync,
//            then batch-read from smem. Tiles at 1024 elements.
// LOAD       Read from global memory, widening to T if narrower.
// SEQUENCE   Compute base + i * multiplier in registers.
// RUNEND     Forward-scan through ends/values arrays that input stages
//            decoded into shared memory. Per-thread cursor in
//            runend_cursors[] avoids re-searching across tile iterations.
//
// ## Mixed-width support
//
// LOAD sources from pending subtrees may have a narrower type than the
// output (e.g. u8 dict codes in a u32 plan). load_element() widens
// to T via static_cast — no separate widen kernel or smem intermediate.

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include "bit_unpack.cuh"
#include "dynamic_dispatch.h"
#include "types.cuh"

// ═══════════════════════════════════════════════════════════════════════════
// Primitives
// ═══════════════════════════════════════════════════════════════════════════

/// Binary search for the first element in `data[0..len)` strictly greater
/// than `value`. Returns `len` if all elements are ≤ value.
template <typename T>
__device__ inline uint64_t upper_bound(const T *data, uint64_t len, uint64_t value) {
    auto it = thrust::upper_bound(thrust::seq, data, data + len, value);
    return it - data;
}

/// Read one element from global memory at `ptype` width, widen to T.
/// Signed types are sign-extended; unsigned types are zero-extended.
template <typename T>
__device__ inline T load_element(const void *__restrict ptr, PTypeTag ptype, uint64_t idx) {
    switch (ptype) {
    case PTYPE_U8:
        return static_cast<T>(static_cast<const uint8_t *>(ptr)[idx]);
    case PTYPE_I8:
        return static_cast<T>(static_cast<const int8_t *>(ptr)[idx]);
    case PTYPE_U16:
        return static_cast<T>(static_cast<const uint16_t *>(ptr)[idx]);
    case PTYPE_I16:
        return static_cast<T>(static_cast<const int16_t *>(ptr)[idx]);
    case PTYPE_U32:
    case PTYPE_F32:
        return static_cast<T>(static_cast<const uint32_t *>(ptr)[idx]);
    case PTYPE_I32:
        return static_cast<T>(static_cast<const int32_t *>(ptr)[idx]);
    case PTYPE_U64:
    case PTYPE_F64:
        return static_cast<T>(static_cast<const uint64_t *>(ptr)[idx]);
    case PTYPE_I64:
        return static_cast<T>(static_cast<const int64_t *>(ptr)[idx]);
    default:
        __builtin_unreachable();
    }
}

/// Per-thread run cursor for RUNEND forward-scan, one entry per thread.
///
/// Stored in shared memory so the cursor persists across successive
/// source_op calls in the tile loop. Each thread's positions are
/// monotonically increasing across tiles, so the cursor only advances
/// forward — the next tile picks up exactly where the previous one
/// stopped, avoiding a binary search per tile. The only binary search
/// is the initial upper_bound seed before the tile loop begins.
__shared__ uint64_t runend_cursors[BLOCK_SIZE];

// ═══════════════════════════════════════════════════════════════════════════
// Scalar ops
// ═══════════════════════════════════════════════════════════════════════════

/// Apply one scalar operation to N values in registers.
template <typename T, uint32_t N>
__device__ inline void scalar_op(T *values, const struct ScalarOp &op, char *__restrict smem) {
    switch (op.op_code) {
    case ScalarOp::FOR: {
        const T ref = static_cast<T>(op.params.frame_of_ref.reference);
#pragma unroll
        for (uint32_t i = 0; i < N; ++i) {
            values[i] += ref;
        }
        break;
    }
    case ScalarOp::ZIGZAG: {
#pragma unroll
        for (uint32_t i = 0; i < N; ++i) {
            values[i] = (values[i] >> 1) ^ static_cast<T>(-(values[i] & 1));
        }
        break;
    }
    case ScalarOp::ALP: {
        const float f = op.params.alp.f, e = op.params.alp.e;
#pragma unroll
        for (uint32_t i = 0; i < N; ++i) {
            float r = static_cast<float>(static_cast<int32_t>(values[i])) * f * e;
            values[i] = static_cast<T>(__float_as_uint(r));
        }
        break;
    }
    case ScalarOp::DICT: {
        const T *dict = reinterpret_cast<const T *>(smem + op.params.dict.values_smem_byte_offset);
#pragma unroll
        for (uint32_t i = 0; i < N; ++i) {
            values[i] = dict[static_cast<uint32_t>(values[i])];
        }
        break;
    }
    default:
        __builtin_unreachable();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Source ops
// ═══════════════════════════════════════════════════════════════════════════

/// FastLanes cooperative unpack — all threads in the block scatter-write
/// decoded elements into `dst`. Caller must issue __syncthreads() before
/// any thread reads from `dst`.
template <typename T>
__device__ inline void bitunpack(const T *__restrict packed,
                                 T *__restrict dst,
                                 uint64_t chunk_start,
                                 uint32_t chunk_len,
                                 const struct SourceOp &src) {
    constexpr uint32_t T_BITS = sizeof(T) * 8;
    constexpr uint32_t FL_CHUNK = 1024;
    constexpr uint32_t LANES = FL_CHUNK / T_BITS;
    const uint32_t bw = src.params.bitunpack.bit_width;
    const uint32_t words_per_block = LANES * bw;
    const uint32_t elem_off = src.params.bitunpack.element_offset;
    const uint32_t dst_off = (chunk_start + elem_off) % FL_CHUNK;
    const uint64_t first_block = (chunk_start + elem_off) / FL_CHUNK;

    static_assert((ELEMENTS_PER_BLOCK % FL_CHUNK) == 0);
    const uint32_t n_chunks = (chunk_len + dst_off + FL_CHUNK - 1) / FL_CHUNK;

    for (uint32_t c = 0; c < n_chunks; ++c) {
        const T *src_chunk = packed + (first_block + c) * words_per_block;
        T *chunk_dst = dst + c * FL_CHUNK;
        for (uint32_t lane = threadIdx.x; lane < LANES; lane += blockDim.x) {
            bit_unpack_lane<T>(src_chunk, chunk_dst, 0, lane, bw);
        }
    }
}

/// Read N values from a source op into `out`.
///
/// Dispatches on `src.op_code` to handle each encoding:
///   BITUNPACK — read from `smem_src` at `smem_base` offset.
///   LOAD      — read from `raw_input` via load_element (type-widening).
///   SEQUENCE  — compute base + pos × multiplier in registers.
///   RUNEND    — forward-scan ends/values in smem using runend_cursors.
///
/// Position calculation (via THREAD_POS macro):
///   N > 1 (batched): pos = base + j·blockDim.x + threadIdx.x.
///                    Caller passes the tile base WITHOUT threadIdx.x.
///   N = 1 (single):  base is the exact position. No stride added.
template <typename T, uint32_t N>
__device__ inline void source_op(T *out,
                                 const struct SourceOp &src,
                                 const void *raw_input,
                                 PTypeTag ptype,
                                 const T *smem_src,
                                 uint32_t smem_base,
                                 uint64_t global_base,
                                 char *__restrict smem) {
    // Wrapped in a macro, rather than a lambda, to avoid allocating additional GPU registers.
#define THREAD_POS(base, j) ((N == 1) ? (base) : ((base) + (j) * blockDim.x + threadIdx.x))

    switch (src.op_code) {
    case SourceOp::BITUNPACK: {
#pragma unroll
        for (uint32_t j = 0; j < N; ++j) {
            out[j] = smem_src[THREAD_POS(smem_base, j)];
        }
        return;
    }
    case SourceOp::LOAD: {
#pragma unroll
        for (uint32_t j = 0; j < N; ++j) {
            out[j] = load_element<T>(raw_input, ptype, THREAD_POS(global_base, j));
        }
        return;
    }
    case SourceOp::SEQUENCE: {
        const T base = static_cast<T>(src.params.sequence.base);
        const T mul = static_cast<T>(src.params.sequence.multiplier);
#pragma unroll
        for (uint32_t j = 0; j < N; ++j) {
            out[j] = base + static_cast<T>(THREAD_POS(global_base, j)) * mul;
        }
        return;
    }
    case SourceOp::RUNEND: {
        const T *ends = reinterpret_cast<const T *>(smem + src.params.runend.ends_smem_byte_offset);
        const T *values = reinterpret_cast<const T *>(smem + src.params.runend.values_smem_byte_offset);
        const uint64_t num_runs = src.params.runend.num_runs;
        const uint64_t offset = src.params.runend.offset;
        uint64_t &run = runend_cursors[threadIdx.x];
#pragma unroll
        for (uint32_t j = 0; j < N; ++j) {
            const uint64_t pos = THREAD_POS(global_base, j) + offset;
            while (run < num_runs && static_cast<uint64_t>(ends[run]) <= pos) {
                run++;
            }
            out[j] = values[min(run, num_runs - 1)];
        }
        return;
    }
    default:
        __builtin_unreachable();
    }

#undef THREAD_POS
}

// ═══════════════════════════════════════════════════════════════════════════
// Output stage — source_op → scalar_op → streaming store
// ═══════════════════════════════════════════════════════════════════════════
//
// BITUNPACK tiles at SMEM_TILE_SIZE: cooperative unpack → smem → sync →
// batched read.  LOAD, SEQUENCE, and RUNEND need no smem scratch and
// process the full block in a single outer iteration, tiled by tile_idx.

/// How many elements to process in this BITUNPACK tile iteration.
/// The first tile may be shorter due to `element_offset` alignment;
/// the last tile may be shorter because we've reached `block_len`.
__device__ inline uint32_t bitunpack_tile_len(const Stage &stage, uint32_t block_len, uint32_t tile_off) {
    const uint32_t off = (tile_off == 0) ? stage.source.params.bitunpack.element_offset : 0;
    return min(SMEM_TILE_SIZE - off, block_len - tile_off);
}

/// Process the final / output stage: decode source → apply scalar ops →
/// streaming-store to global memory. Handles the full block, tiling through
/// smem scratch for BITUNPACK.
template <typename T>
__device__ void execute_output_stage(T *__restrict output,
                                     const Stage &stage,
                                     char *__restrict smem,
                                     uint64_t block_start,
                                     uint32_t block_len) {
    // Cap at 4 values per thread per tile to minimise register pressure.
    constexpr uint32_t VALUES_PER_TILE = (32 / sizeof(T)) < 4 ? (32 / sizeof(T)) : 4;
    const uint32_t tile_size = blockDim.x * VALUES_PER_TILE;
    const auto &src = stage.source;
    const void *raw_input = reinterpret_cast<const void *>(stage.input_ptr);
    const PTypeTag ptype = stage.source_ptype;

    if (src.op_code == SourceOp::RUNEND) {
        // Seed each thread's cursor with the run containing its first
        // strided position. The RUNEND arm in source_op advances the
        // cursor monotonically, so this avoids a full binary search on
        // every element.
        const T *ends = reinterpret_cast<const T *>(smem + src.params.runend.ends_smem_byte_offset);
        runend_cursors[threadIdx.x] = upper_bound(ends,
                                                  src.params.runend.num_runs,
                                                  block_start + threadIdx.x + src.params.runend.offset);
    }

    for (uint32_t elem_idx = 0; elem_idx < block_len;) {
        uint32_t chunk_len;
        const T *smem_src = nullptr;

        // BITUNPACK uses smem scratch, so the outer loop advances one
        // chunk at a time. LOAD, SEQUENCE, and RUNEND need no smem
        // scratch, so chunk_len = block_len (single outer iteration);
        // tiling happens in the inner tile_idx loop.
        if (src.op_code == SourceOp::BITUNPACK) {
            chunk_len = bitunpack_tile_len(stage, block_len, elem_idx);
            T *scratch = reinterpret_cast<T *>(smem + stage.smem_byte_offset);
            bitunpack<T>(reinterpret_cast<const T *>(stage.input_ptr),
                         scratch,
                         block_start + elem_idx,
                         chunk_len,
                         src);
            constexpr uint32_t FL_CHUNK = 1024; // FastLanes chunk size
            const uint32_t align = (block_start + elem_idx + src.params.bitunpack.element_offset) % FL_CHUNK;
            smem_src = scratch + align;
            // Write barrier: all threads finished bitunpack, safe to read from scratch.
            __syncthreads();
        } else {
            chunk_len = block_len;
        }

        const uint32_t tile_count = chunk_len / tile_size;
        for (uint32_t tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
            const uint64_t tile_start = block_start + elem_idx + static_cast<uint64_t>(tile_idx) * tile_size;
            T values[VALUES_PER_TILE];

            source_op<T, VALUES_PER_TILE>(values,
                                          src,
                                          raw_input,
                                          ptype,
                                          smem_src,
                                          tile_idx * tile_size,
                                          tile_start,
                                          smem);

            for (uint8_t op = 0; op < stage.num_scalar_ops; ++op) {
                scalar_op<T, VALUES_PER_TILE>(values, stage.scalar_ops[op], smem);
            }

#pragma unroll
            for (uint32_t j = 0; j < VALUES_PER_TILE; ++j) {
                // st.cs (cache streaming): marks this line for earliest
                // eviction in L1 and L2. Output data is written once and
                // never read again by this kernel, so keeping it cached
                // would only compete with the packed input buffers and
                // smem-resident dict/runend data that the next tiles still
                // need to read. Evict-first lets those stay resident.
                __stcs(&output[tile_start + j * blockDim.x + threadIdx.x], values[j]);
            }
        }

        const uint32_t rem = tile_count * tile_size;
        for (uint32_t i = rem + threadIdx.x; i < chunk_len; i += blockDim.x) {
            const uint64_t gpos = block_start + elem_idx + i;
            T val;
            source_op<T, 1>(&val, src, raw_input, ptype, smem_src, i, gpos, smem);

            for (uint8_t op = 0; op < stage.num_scalar_ops; ++op) {
                scalar_op<T, 1>(&val, stage.scalar_ops[op], smem);
            }
            __stcs(&output[gpos], val);
        }

        if (src.op_code == SourceOp::BITUNPACK) {
            // Read barrier: all threads finished reading scratch, safe to
            // overwrite it with the next chunk's bitunpack.
            __syncthreads();
        }
        elem_idx += chunk_len;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Input stages — decode dependencies into shared memory for the output stage
// ═══════════════════════════════════════════════════════════════════════════

/// Decode one input stage (dict values, run-end endpoints, etc.) into its
/// shared memory region so the output stage can reference it later.
/// Applies any scalar ops in-place before returning.
///
/// Unlike execute_output_stage, this does not tile — the entire stage is
/// decoded in one pass. The output stage needs random access into these
/// smem regions (e.g. DICT gathers by arbitrary code value), so the data
/// must be fully resident. The smem limit check in the Rust plan builder
/// ensures the stage fits; if it doesn't, the plan falls back to Unfused.
template <typename T>
__device__ void execute_input_stage(const Stage &stage, char *__restrict smem) {
    T *smem_out = reinterpret_cast<T *>(smem + stage.smem_byte_offset);
    const auto &src = stage.source;

    if (src.op_code == SourceOp::BITUNPACK) {
        bitunpack<T>(reinterpret_cast<const T *>(stage.input_ptr), smem_out, 0, stage.len, src);
        smem_out += src.params.bitunpack.element_offset % SMEM_TILE_SIZE;
        // Write barrier: cooperative bitunpack finished, safe to read
        // decoded elements in the scalar-op loop below.
        __syncthreads();

        if (stage.num_scalar_ops > 0) {
            for (uint32_t i = threadIdx.x; i < stage.len; i += blockDim.x) {
                T val = smem_out[i];
                for (uint8_t op = 0; op < stage.num_scalar_ops; ++op) {
                    scalar_op<T, 1>(&val, stage.scalar_ops[op], smem);
                }
                smem_out[i] = val;
            }
            // Write barrier: scalar ops applied in-place, smem region is
            // now fully populated for subsequent stages to read.
            __syncthreads();
        }
    } else {
        if (src.op_code == SourceOp::RUNEND) {
            // Seed each thread's cursor with the run containing its first
            // strided position. The RUNEND arm in source_op advances the
            // cursor monotonically, so this avoids a full binary search on
            // every element.
            const T *ends = reinterpret_cast<const T *>(smem + src.params.runend.ends_smem_byte_offset);
            runend_cursors[threadIdx.x] =
                upper_bound(ends, src.params.runend.num_runs, threadIdx.x + src.params.runend.offset);
        }
        const void *raw_input = reinterpret_cast<const void *>(stage.input_ptr);
        for (uint32_t i = threadIdx.x; i < stage.len; i += blockDim.x) {
            T val;
            source_op<T, 1>(&val, src, raw_input, stage.source_ptype, nullptr, 0, i, smem);
            for (uint8_t op = 0; op < stage.num_scalar_ops; ++op) {
                scalar_op<T, 1>(&val, stage.scalar_ops[op], smem);
            }
            smem_out[i] = val;
        }
        // Write barrier: smem region is fully populated for subsequent
        // stages to read.
        __syncthreads();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel entry
// ═══════════════════════════════════════════════════════════════════════════

/// Kernel entry point. Parses the packed plan, runs all input stages to
/// populate shared memory, then runs the output stage to produce results.
template <typename T>
__device__ void
dynamic_dispatch(T *__restrict output, uint64_t array_len, const uint8_t *__restrict packed_plan) {
    extern __shared__ char smem[];

    const auto *hdr = reinterpret_cast<const struct PlanHeader *>(packed_plan);
    const uint8_t *cursor = packed_plan + sizeof(struct PlanHeader);
    const uint8_t last = hdr->num_stages - 1;

    for (uint8_t i = 0; i < last; ++i) {
        Stage input_stage = parse_stage(cursor);
        execute_input_stage<T>(input_stage, smem);
    }

    Stage output_stage = parse_stage(cursor);
    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * ELEMENTS_PER_BLOCK;
    const uint64_t block_end = min(block_start + ELEMENTS_PER_BLOCK, array_len);
    execute_output_stage<T>(output,
                            output_stage,
                            smem,
                            block_start,
                            static_cast<uint32_t>(block_end - block_start));
}

// Kernels are instantiated only for unsigned integer types. Signed and
// floating-point arrays reuse the unsigned kernel of the same width —
// the data is bit-identical under reinterpretation, and all arithmetic
// in the pipeline (FoR add, ZigZag decode, ALP decode, DICT gather) is
// correct on the unsigned representation. The one place where signedness
// matters is load_element(), which dispatches on the per-op PTypeTag to
// sign-extend or zero-extend when widening a narrow source to T.
#define GENERATE_KERNEL(suffix, Type)                                                                        \
    extern "C" __global__ void __launch_bounds__(BLOCK_SIZE, 32)                                             \
        dynamic_dispatch_##suffix(Type *__restrict output,                                                   \
                                  uint64_t array_len,                                                        \
                                  const uint8_t *__restrict packed_plan) {                                   \
        dynamic_dispatch<Type>(output, array_len, packed_plan);                                              \
    }

FOR_EACH_UNSIGNED_INT(GENERATE_KERNEL)
