// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace cg = cooperative_groups;

// OnPair decompress — 4 tokens/thread, dictionary staged in CLUSTER-distributed
// shared memory (DSMEM).
//
// Targets the bits16 wall: the ~1 MB padded dict (65 536 × 16 B) does not fit
// one block's shared memory and overflows the ~256 KB L1, so the per-token
// random dict read misses L1 and pays an L2 round-trip — a gather-latency wall
// that L2-persist, narrower reads, and the length-bucket layout all failed to
// move. Here a thread-block cluster of ONPAIR_CLUSTER_N co-resident blocks
// shards the dict across their shared memory (each block holds 1/N of the
// entries); a token's dict entry is read from the owning block's shared memory
// over the on-chip SM-to-SM network via `map_shared_rank`, replacing the L2
// round-trip with a (lower-latency) DSMEM access. Decode bytes are identical to
// `onpair_shmem_4tpt` — this only changes where the dict bytes are read from.
//
// Cost / risk: the dict slice forces ~1 block/SM (low occupancy), and remote
// DSMEM reads traverse the GPC fabric (higher latency than local shared, and a
// random all-to-all gather can saturate it). Whether DSMEM latency beats the
// L2 round-trip net of lost occupancy is the open, wall-clock-only question.

#ifndef WARPS_PER_BLOCK_MAX
#define WARPS_PER_BLOCK_MAX 8u
#endif
#ifndef ONPAIR_CLUSTER_N
#define ONPAIR_CLUSTER_N 8
#endif
#define WARP_BUF_BYTES 2080u

__device__ inline uint32_t warp_inclusive_scan_u32_cdsm(uint32_t x, int lane) {
    constexpr unsigned mask = 0xffffffffu;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        uint32_t y = __shfl_up_sync(mask, x, offset);
        if (lane >= offset) {
            x += y;
        }
    }
    return x;
}

__device__ inline void emit_token_cdsm(uint8_t *s_buf, uint32_t base,
                                       const uint4 &tok, uint32_t len) {
    const uint8_t *tb = reinterpret_cast<const uint8_t *>(&tok);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        if (j < (int)len) {
            s_buf[base + j] = tb[j];
        }
    }
}

extern "C" __global__ void __cluster_dims__(ONPAIR_CLUSTER_N, 1, 1)
    onpair_shmem_4tpt_cluster_dsmem(
        const uint16_t *__restrict codes,
        const uint64_t *__restrict chunk_offsets,
        const uint8_t *__restrict dict_padded, const uint8_t *__restrict lens,
        uint8_t *__restrict output_bytes, uint64_t total_tokens,
        uint32_t dict_entries) {
    constexpr unsigned mask = 0xffffffffu;

    // Dynamic shared layout: [dict slice: entries_per_block*16][warp staging].
    // The slice is at offset 0 in every block, so `map_shared_rank(dict_slice,
    // r)` returns block r's slice base.
    extern __shared__ __align__(16) uint8_t smem[];

    cg::cluster_group cluster = cg::this_cluster();
    const uint32_t crank = cluster.block_rank();
    const uint32_t entries_per_block =
        (dict_entries + ONPAIR_CLUSTER_N - 1u) / ONPAIR_CLUSTER_N;
    const uint32_t slice_bytes = entries_per_block * 16u;

    uint8_t *dict_slice = smem;
    uint8_t *s_buf_all = smem + slice_bytes;

    // Cooperative load: this block fills its slice from global; out-of-range
    // tail entries (last block) are zeroed but never addressed (code < entries).
    const uint32_t my_first = crank * entries_per_block;
    for (uint32_t e = threadIdx.x; e < entries_per_block; e += blockDim.x) {
        const uint32_t g = my_first + e;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (g < dict_entries) {
            v = *reinterpret_cast<const uint4 *>(dict_padded + (size_t)g * 16u);
        }
        *reinterpret_cast<uint4 *>(dict_slice + (size_t)e * 16u) = v;
    }
    // All slices must be populated before any remote read.
    cluster.sync();

    const int lane = threadIdx.x & 31;
    const uint32_t warp_in_block = threadIdx.x >> 5;
    const uint32_t warps_per_block = blockDim.x >> 5;
    uint8_t *s_buf_base = &s_buf_all[warp_in_block * WARP_BUF_BYTES];

    const uint64_t total_chunks = (total_tokens + 127u) / 128u;
    const uint64_t gwarp0 =
        (uint64_t)blockIdx.x * (uint64_t)warps_per_block + (uint64_t)warp_in_block;
    const uint64_t gstride = (uint64_t)gridDim.x * (uint64_t)warps_per_block;

    for (uint64_t chunk = gwarp0; chunk < total_chunks; chunk += gstride) {
        const uint64_t base_i = chunk * 128u + (uint64_t)lane;
        uint4 t[4];
        uint32_t l[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint64_t i = base_i + (uint64_t)(k * 32);
            if (i < total_tokens) {
                const uint32_t c = (uint32_t)codes[i];
                const uint32_t rank = c / entries_per_block;
                const uint32_t local = c - rank * entries_per_block;
                const uint8_t *remote = reinterpret_cast<const uint8_t *>(
                    cluster.map_shared_rank(dict_slice, rank));
                t[k] = *reinterpret_cast<const uint4 *>(remote + (size_t)local * 16u);
                l[k] = (uint32_t)lens[c];
            } else {
                t[k] = make_uint4(0u, 0u, 0u, 0u);
                l[k] = 0u;
            }
        }

        uint32_t excl[4];
        uint32_t acc_base = 0u;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t incl = warp_inclusive_scan_u32_cdsm(l[k], lane);
            excl[k] = acc_base + (incl - l[k]);
            acc_base += __shfl_sync(mask, incl, 31);
        }
        const uint32_t warp_total = acc_base;

        const uint64_t out_start = chunk_offsets[chunk];
        const uint32_t head_pre = (16u - (uint32_t)(out_start & 15u)) & 15u;
        uint8_t *s_buf = s_buf_base + ((16u - head_pre) & 15u);

#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (l[k] > 0u) {
                emit_token_cdsm(s_buf, excl[k], t[k], l[k]);
            }
        }
        __syncwarp();

        const uint32_t head = head_pre < warp_total ? head_pre : warp_total;
        if ((uint32_t)lane < head) {
            output_bytes[out_start + (uint64_t)lane] = s_buf[lane];
        }
        if (head < warp_total) {
            const uint32_t body_chunks = (warp_total - head) >> 4;
            for (uint32_t k = lane; k < body_chunks; k += 32u) {
                const uint32_t off = head + k * 16u;
                const uint4 v = *reinterpret_cast<const uint4 *>(s_buf + off);
                __stcs(reinterpret_cast<uint4 *>(output_bytes + out_start + off), v);
            }
            const uint32_t tail_start = head + (body_chunks << 4);
            if ((uint32_t)lane < warp_total - tail_start) {
                output_bytes[out_start + (uint64_t)tail_start + (uint64_t)lane] =
                    s_buf[tail_start + lane];
            }
        }
        __syncwarp();
    }

    // No block may free its shared slice while another block is still reading
    // it: park every block here until the whole cluster has finished decoding.
    cluster.sync();
}
