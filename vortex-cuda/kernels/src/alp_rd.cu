// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "patches.cuh"

// ALP-RD dictionaries hold at most 8 left parts (`MAX_DICT_SIZE` in the `alp` crate), so the
// dictionary is a kernel parameter, not a device buffer.
struct AlpRdDictionary {
    uint16_t entries[8];
};
static_assert(sizeof(AlpRdDictionary) == 16, "AlpRdDictionary must match [u16; 8] on the host");

// ALP-RD (ALP for "real doubles") decode: each value is stored as a left part, looked up from a
// dictionary of at most 8 entries through a code, and a right part. The decoded bits are
// `(left << right_bit_width) | right`.
//
// Left parts that are not in the dictionary are stored as patches. Like ALP, each block decodes
// one 1024-element chunk cooperatively: it maps the codes to left parts in shared memory, applies
// that chunk's patches there, and then combines the left and right parts on the way out.
//
// Patch positions are relative to 1024-row chunks of the unsliced array, while the children are
// already sliced. So blocks decode chunk-aligned rows: aligned row `r` is sliced row
// `r - slice_shift`, where `slice_shift` is the slice start modulo 1024. The caller trims the
// first `slice_shift` rows of the output.
template <typename UIntT>
__device__ void alp_rd_device(const uint16_t *__restrict left_codes,
                              const UIntT *__restrict right_parts,
                              UIntT *__restrict out,
                              const AlpRdDictionary &dictionary,
                              uint32_t dictionary_len,
                              uint32_t right_bit_width,
                              uint64_t array_len,
                              uint32_t slice_shift,
                              const uint8_t *__restrict validity,
                              uint64_t validity_offset,
                              uint32_t *__restrict invalid_code,
                              int thread_idx,
                              GPUPatches &patches) {
    constexpr int ThreadCount = 32;
    // As in alp.cu, one warp per block is baked in: __syncwarp() is only enough because every
    // thread lives in the same warp.
    static_assert(ThreadCount == 32, "alp_rd kernel requires exactly one warp per block");
    static_assert(1024 % ThreadCount == 0, "ThreadCount must evenly divide 1024");
    __shared__ uint16_t shared_left[1024];

    constexpr int per_thread = 1024 / ThreadCount;
    uint64_t chunk_base = static_cast<uint64_t>(blockIdx.x) * 1024;

    // Step 1: map codes to left parts. A code outside the dictionary maps to 0 rather than reading
    // past it. Null rows may hold such codes, as on the CPU, but a valid row raises `invalid_code`.
    // `validity` is null when every row is valid.
    bool saw_invalid_code = false;
#pragma unroll
    for (int i = 0; i < per_thread; i++) {
        int idx = i * ThreadCount + thread_idx;
        uint64_t aligned_row = chunk_base + static_cast<uint64_t>(idx);
        uint16_t left = 0;
        if (aligned_row >= slice_shift && aligned_row - slice_shift < array_len) {
            uint64_t row = aligned_row - slice_shift;
            uint16_t code = left_codes[row];
            if (code < dictionary_len) {
                left = dictionary.entries[code];
            } else {
                uint64_t bit = validity_offset + row;
                saw_invalid_code |= validity == nullptr || ((validity[bit >> 3] >> (bit & 7)) & 1);
            }
        }
        shared_left[idx] = left;
    }
    // One atomic write per warp, so threads and blocks never race on the flag.
    if (__any_sync(0xffffffffu, saw_invalid_code) && thread_idx == 0) {
        atomicOr(invalid_code, 1u);
    }
    __syncwarp();

    // Step 2: overwrite the left parts that could not be dictionary encoded.
    PatchesCursor<uint16_t> cursor(patches, blockIdx.x, thread_idx, static_cast<uint32_t>(ThreadCount));
    auto patch = cursor.next();
    while (patch.index != 1024) {
        shared_left[patch.index] = patch.value;
        patch = cursor.next();
    }
    __syncwarp();

    // Step 3: combine with the right parts. The caller allocates `out` for every aligned row of
    // every block, so each block writes a whole chunk.
#pragma unroll
    for (int i = 0; i < per_thread; i++) {
        int idx = i * ThreadCount + thread_idx;
        uint64_t aligned_row = chunk_base + static_cast<uint64_t>(idx);
        UIntT right = 0;
        if (aligned_row >= slice_shift && aligned_row - slice_shift < array_len) {
            right = right_parts[aligned_row - slice_shift];
        }
        out[aligned_row] = (static_cast<UIntT>(shared_left[idx]) << right_bit_width) | right;
    }
}

#define GENERATE_ALP_RD_KERNEL(uint_suffix, UIntT)                                                           \
    extern "C" __global__ void alp_rd_##uint_suffix##_32t(                                                   \
        const uint16_t *__restrict left_codes,                                                               \
        const UIntT *__restrict right_parts,                                                                 \
        UIntT *__restrict out,                                                                               \
        const __grid_constant__ AlpRdDictionary dictionary,                                                  \
        uint32_t dictionary_len,                                                                             \
        uint32_t right_bit_width,                                                                            \
        uint64_t array_len,                                                                                  \
        uint32_t slice_shift,                                                                                \
        const uint8_t *__restrict validity,                                                                  \
        uint64_t validity_offset,                                                                            \
        uint32_t *__restrict invalid_code,                                                                   \
        GPUPatches patches) {                                                                                \
        alp_rd_device<UIntT>(left_codes,                                                                     \
                             right_parts,                                                                    \
                             out,                                                                            \
                             dictionary,                                                                     \
                             dictionary_len,                                                                 \
                             right_bit_width,                                                                \
                             array_len,                                                                      \
                             slice_shift,                                                                    \
                             validity,                                                                       \
                             validity_offset,                                                                \
                             invalid_code,                                                                   \
                             threadIdx.x,                                                                    \
                             patches);                                                                       \
    }

// f32 values are stored as u32 bits and f64 values as u64 bits.
GENERATE_ALP_RD_KERNEL(u32, uint32_t)
GENERATE_ALP_RD_KERNEL(u64, uint64_t)
