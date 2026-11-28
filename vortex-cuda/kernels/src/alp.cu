// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "patches.cuh"

// ALP (Adaptive Lossless floating-Point) decode: out[i] = (FloatT)in[i] * f * e.
//
// Each block processes one 1024-element chunk cooperatively and applies patches
// into shared memory before writing to global memory, mirroring the strategy
// used by bit_unpack. f = F10[exponents.f], e = IF10[exponents.e].
//
// Patch positions count from 1024-row chunks of the unsliced array, but `in` is already sliced, so
// blocks decode chunk-aligned rows shifted by `slice_shift`, the slice start modulo 1024. The
// caller trims the shift off the output.
//
// The cast from EncT to FloatT must preserve ALP's lossless contract: f32 is
// only encoded as i32, and f64 is only encoded as i64. The i64 → double cast
// is lossless for all values ALP can produce.
template <typename EncT, typename FloatT>
__device__ void alp_device(const EncT *__restrict in,
                           FloatT *__restrict out,
                           FloatT f,
                           FloatT e,
                           uint64_t array_len,
                           uint32_t slice_shift,
                           int thread_idx,
                           GPUPatches &patches) {
    constexpr int ThreadCount = 32;
    // ThreadCount == 32 (one warp) is baked into this kernel:
    //   - __syncwarp() below is only sufficient because all threads live in one warp.
    //   - per_thread must evenly divide 1024 so the unrolled loops cover the chunk.
    static_assert(ThreadCount == 32, "alp kernel requires exactly one warp per block");
    static_assert(1024 % ThreadCount == 0, "ThreadCount must evenly divide 1024");
    __shared__ FloatT shared_out[1024];

    constexpr int per_thread = 1024 / ThreadCount;
    uint64_t chunk_base = static_cast<uint64_t>(blockIdx.x) * 1024;

    // The whole block can read without per-row checks when it lies inside the sliced input.
    if (chunk_base >= slice_shift && chunk_base - slice_shift + 1024 <= array_len) {
        const EncT *chunk_in = in + (chunk_base - slice_shift);
#pragma unroll
        for (int i = 0; i < per_thread; i++) {
            int idx = i * ThreadCount + thread_idx;
            shared_out[idx] = static_cast<FloatT>(chunk_in[idx]) * f * e;
        }
    } else {
#pragma unroll
        for (int i = 0; i < per_thread; i++) {
            int idx = i * ThreadCount + thread_idx;
            uint64_t aligned_row = chunk_base + static_cast<uint64_t>(idx);
            if (aligned_row >= slice_shift && aligned_row - slice_shift < array_len) {
                shared_out[idx] = static_cast<FloatT>(in[aligned_row - slice_shift]) * f * e;
            } else {
                shared_out[idx] = FloatT {};
            }
        }
    }
    __syncwarp();

    // Step 2: apply patches in parallel across the warp.
    PatchesCursor<FloatT> cursor(patches, blockIdx.x, thread_idx, static_cast<uint32_t>(ThreadCount));
    auto patch = cursor.next();
    while (patch.index != 1024) {
        shared_out[patch.index] = patch.value;
        patch = cursor.next();
    }
    __syncwarp();

// Step 3: coalesced write-out of the full 1024-element chunk. The caller
// allocates `out` for every aligned row of every block, so each block writes
// entirely within bounds. Rows outside the slice hold don't-care values; the
// caller slices them off.
#pragma unroll
    for (int i = 0; i < per_thread; i++) {
        int idx = i * ThreadCount + thread_idx;
        out[chunk_base + idx] = shared_out[idx];
    }
}

#define GENERATE_ALP_KERNEL(enc_suffix, float_suffix, EncT, FloatT)                                          \
    extern "C" __global__ void alp_##enc_suffix##_##float_suffix##_32t(const EncT *__restrict full_in,       \
                                                                       FloatT *__restrict full_out,          \
                                                                       FloatT f,                             \
                                                                       FloatT e,                             \
                                                                       uint64_t array_len,                   \
                                                                       uint32_t slice_shift,                 \
                                                                       GPUPatches patches) {                 \
        alp_device<EncT, FloatT>(full_in, full_out, f, e, array_len, slice_shift, threadIdx.x, patches);     \
    }

// The only ALPInt bindings produced by the encoder are (f32, i32) and (f64, i64).
// i64 → double is lossless; i32 → float is lossless for all values ALP emits.
GENERATE_ALP_KERNEL(i32, f32, int32_t, float)
GENERATE_ALP_KERNEL(i64, f64, int64_t, double)
