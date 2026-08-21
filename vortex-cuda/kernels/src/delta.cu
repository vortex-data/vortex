// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "fastlanes_common.cuh"
#include "types.cuh"

// FastLanes delta decode.
//
// Delta is stored in the FastLanes transposed layout, where a 1024-element chunk is
// FL_LANES<T> independent columns of (1024 / FL_LANES<T>) rows. Each column carries its own
// running total seeded from that lane's base, so the chunk decodes as FL_LANES<T> independent
// sequential scans rather than one scan over 1024 elements. One thread owns one lane.
//
// This mirrors `fastlanes::Delta::undelta` followed by `Transpose::untranspose`, which is what
// the CPU decoder (`delta_decompress`) runs. Both steps must stay in step with that crate.

/// Maps a position in the transposed layout to its position in natural order.
///
/// Mirrors `fastlanes::transpose`; `untranspose` is `output[transpose(i)] = input[i]`.
__device__ inline uint32_t fl_transpose_index(uint32_t idx) {
    const uint32_t lane = idx % 16;
    const uint32_t order = (idx / 16) % 8;
    const uint32_t row = idx / 128;
    return (lane * 64) + (FL_ORDER[order] * 8) + row;
}

/// Decodes `num_chunks` full 1024-element delta chunks.
///
/// `deltas` and `output` hold `num_chunks * FL_CHUNK` elements; `bases` holds
/// `num_chunks * FL_LANES<T>`. Only unsigned types are instantiated: the CPU decoder
/// reinterprets signed input through its unsigned counterpart so that the wrapping add here
/// inverts the wrapping subtract done at compress time.
template <typename T>
__device__ void delta_decode_kernel(const T *const __restrict deltas,
                                    const T *const __restrict bases,
                                    T *const __restrict output,
                                    uint64_t num_chunks) {
    constexpr uint32_t LANES = FL_LANES<T>;
    constexpr uint32_t ROWS = FL_CHUNK / LANES;

    // The undelta pass writes the chunk in transposed order; the untranspose pass then reads it
    // back in a different order, so the whole chunk is staged in shared memory between them.
    __shared__ T transposed[FL_CHUNK];

    for (uint64_t chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
        const T *const in = deltas + chunk * FL_CHUNK;
        const T *const base = bases + chunk * LANES;

        // Each lane accumulates down its own column, seeded by that lane's base.
        for (uint32_t lane = threadIdx.x; lane < LANES; lane += blockDim.x) {
            T running = base[lane];
            for (uint32_t row = 0; row < ROWS; ++row) {
                const uint32_t idx = INDEX(row, lane);
                running = static_cast<T>(running + in[idx]);
                transposed[idx] = running;
            }
        }
        __syncthreads();

        T *const out = output + chunk * FL_CHUNK;
        for (uint32_t i = threadIdx.x; i < FL_CHUNK; i += blockDim.x) {
            out[fl_transpose_index(i)] = transposed[i];
        }
        // Guard the staging buffer before the next chunk overwrites it.
        __syncthreads();
    }
}

#define GENERATE_DELTA_KERNEL(suffix, Type)                                                                  \
    extern "C" __global__ void delta_##suffix(const Type *const __restrict deltas,                           \
                                              const Type *const __restrict bases,                            \
                                              Type *const __restrict output,                                 \
                                              uint64_t num_chunks) {                                         \
        delta_decode_kernel<Type>(deltas, bases, output, num_chunks);                                        \
    }

// Signed input is reinterpreted to its unsigned counterpart before the launch.
FOR_EACH_UNSIGNED_INT(GENERATE_DELTA_KERNEL)
