// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompress — constant-2-byte dictionary specialization.
//
// Sibling of `onpair_shmem_const1` for columns where every dict entry
// has length 2 (e.g. language/country/state-code columns). Each lane
// reads 8 consecutive codes, looks up two-byte dict entries, packs into
// a uint4 (8 × 2 B = 16 B) and emits one aligned `st.global.cs.v4.u32`.
// 32 lanes × 16 B = 512 B = 256 tokens per warp.

extern "C" __global__ __launch_bounds__(512, 4) void onpair_shmem_const2(
    const uint16_t *__restrict codes,
    const uint16_t *__restrict dict_const2, // dict[code] = 2-byte entry; 2 B stride
    uint8_t *__restrict output_bytes, uint64_t total_tokens) {
    const int lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint64_t chunk =
        (uint64_t)blockIdx.x * (uint64_t)(blockDim.x >> 5) + (uint64_t)warp_id;
    // Each warp produces 256 tokens × 2 B = 512 B at output[chunk * 512].
    const uint64_t block_start_tok = chunk * 256u;
    if (block_start_tok >= total_tokens) {
        return;
    }

    const uint64_t lane_start = block_start_tok + (uint64_t)(lane * 8);
    if (lane_start >= total_tokens) {
        return;
    }

    // Common-case fast path: 8 tokens × 2 B = uint4 store.
    if (lane_start + 8u <= total_tokens) {
        uint16_t w[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t c = (uint32_t)codes[lane_start + j];
            w[j] = dict_const2[c];
        }
        const uint4 v = *reinterpret_cast<const uint4 *>(w);
        const uint64_t out = (uint64_t)(lane_start * 2u);
        __stcs(reinterpret_cast<uint4 *>(output_bytes + out), v);
        return;
    }

    // Tail: this lane straddles `total_tokens`. Write 2 bytes per token.
    const uint32_t n = (uint32_t)(total_tokens - lane_start);
    for (uint32_t j = 0; j < n; ++j) {
        const uint32_t c = (uint32_t)codes[lane_start + j];
        const uint16_t entry = dict_const2[c];
        output_bytes[(lane_start + j) * 2u]     = (uint8_t)(entry & 0xffu);
        output_bytes[(lane_start + j) * 2u + 1] = (uint8_t)(entry >> 8);
    }
}
