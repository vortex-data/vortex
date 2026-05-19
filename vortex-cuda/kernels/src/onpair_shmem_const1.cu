// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// OnPair decompress — constant-1-byte dictionary specialization.
//
// When every dict entry has length 1 (e.g. TPC-H l_returnflag/l_linestatus,
// country codes, status codes, single-character categorical columns), the
// whole "variable-length compaction" machinery — warp scan, byte ladder,
// shared-mem staging — collapses to a no-op. Each lane:
//
//   * reads 16 consecutive codes (codes[chunk*512 + lane*16 .. + 16])
//   * looks up the 1-byte dict entry for each (dict served from L1; the
//     dict is at most 256 entries so it stays cache-resident after warm-up)
//   * packs the 16 bytes into a uint4 register
//   * issues one aligned `st.global.cs.v4.u32` to output[chunk*512 + lane*16]
//
// 32 lanes × 16 B = 512 B per warp, fully coalesced. No scan, no sync, no
// shared mem. Eliminates ~30 cycles of warp-scan + ~20 cycles of byte
// ladder + 1 __syncwarp per warp iter relative to the generic kernels.

extern "C" __global__ __launch_bounds__(512, 4) void onpair_shmem_const1(
    const uint16_t *__restrict codes,
    const uint8_t *__restrict dict_const1, // dict[code] = 1-byte entry; 1 B stride
    uint8_t *__restrict output_bytes, uint64_t total_tokens) {
    const int lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    const uint64_t chunk =
        (uint64_t)blockIdx.x * (uint64_t)(blockDim.x >> 5) + (uint64_t)warp_id;
    // Each warp emits a 512-token block = 512 bytes at output[chunk * 512].
    const uint64_t block_start_tok = chunk * 512u;
    if (block_start_tok >= total_tokens) {
        return;
    }

    const uint64_t lane_start = block_start_tok + (uint64_t)(lane * 16);
    if (lane_start >= total_tokens) {
        return;
    }

    // Common-case fast path: full 16-byte aligned uint4 store.
    if (lane_start + 16u <= total_tokens) {
        uint8_t b[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint32_t c = (uint32_t)codes[lane_start + j];
            b[j] = dict_const1[c];
        }
        const uint4 v = *reinterpret_cast<const uint4 *>(b);
        __stcs(reinterpret_cast<uint4 *>(output_bytes + lane_start), v);
        return;
    }

    // Tail: this lane straddles `total_tokens`. Write bytes individually.
    const uint32_t n = (uint32_t)(total_tokens - lane_start);
    for (uint32_t j = 0; j < n; ++j) {
        const uint32_t c = (uint32_t)codes[lane_start + j];
        output_bytes[lane_start + j] = dict_const1[c];
    }
}
