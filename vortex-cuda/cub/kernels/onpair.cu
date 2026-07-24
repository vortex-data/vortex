// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// OnPair per-batch output-offsets regeneration, fused into a single sweep
// kernel: every warp reduces one 128-token batch's decoded size from the codes
// and the length LUT, the first warp scans the block's batch sums, and CUB's
// decoupled look-back (`ScanTileState` + `TilePrefixCallbackOp` — the same
// machinery `cub::DeviceScan` is built on) resolves the running prefix of all
// preceding tiles in-kernel. The per-batch sizes live only in registers and
// shared memory; only the offsets are written to global memory.
//
// `onpair_batch_offsets` is the sole entry point. The tile-state init launch
// it performs first is CUB's own requirement — `DeviceScan` runs the identical
// init internally before its sweep.

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <limits.h>
#include <stdint.h>

namespace {

// One warp reduces one 128-token batch (4 tokens per lane); a 512-thread block
// covers a tile of 16 batches. The decode kernel shares this geometry.
constexpr uint32_t ONPAIR_TOKENS_PER_BATCH = 128;
constexpr uint32_t ONPAIR_BLOCK_THREADS = 512;
constexpr uint32_t ONPAIR_WARPS_PER_BLOCK = ONPAIR_BLOCK_THREADS / 32;

using OnPairTileState = cub::ScanTileState<uint64_t>;

// Version-stable u64 addition functor (avoids deprecated cub thread operators).
struct SumU64 {
    __host__ __device__ inline uint64_t operator()(uint64_t a, uint64_t b) const {
        return a + b;
    }
};

using OnPairPrefixOp = cub::TilePrefixCallbackOp<uint64_t, SumU64, OnPairTileState>;

__global__ void onpair_batch_offsets_init(OnPairTileState tile_state, int num_tiles) {
    tile_state.InitializeStatus(num_tiles);
}

// The fused sweep, instantiated for the two code widths OnPair stores (u16
// natively, u8 when the compressor narrowed the codes). A code outside the
// dictionary raises `status` to 1 and contributes zero bytes: the host must
// check the flag before trusting the offsets and before launching the decode
// kernel, whose dictionary gathers are unchecked.
template <typename CodeT>
__global__
__launch_bounds__(ONPAIR_BLOCK_THREADS) void onpair_batch_offsets_sweep(const CodeT *__restrict codes,
                                                                        const uint8_t *__restrict lens,
                                                                        uint32_t dict_size,
                                                                        uint64_t total_tokens,
                                                                        uint64_t *__restrict chunk_offsets,
                                                                        uint32_t *__restrict status,
                                                                        OnPairTileState tile_state,
                                                                        int64_t num_batches) {
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp = threadIdx.x >> 5;
    const int tile_idx = static_cast<int>(blockIdx.x);
    const int64_t batch = (int64_t)tile_idx * ONPAIR_WARPS_PER_BLOCK + (int64_t)warp;

    // Warp-parallel reduction of this batch's decoded size. Code reads are
    // lane-consecutive (coalesced); the length LUT is small and
    // cache-resident. Batches past the end (last tile) contribute zero.
    uint32_t batch_bytes = 0;
    if (batch < num_batches) {
        const uint64_t base = (uint64_t)batch * (uint64_t)ONPAIR_TOKENS_PER_BATCH;
#pragma unroll
        for (uint8_t token = 0; token < 4; ++token) {
            const uint64_t i = base + (uint64_t)lane + (uint64_t)(token * 32);
            if (i < total_tokens) {
                const uint32_t code = (uint32_t)codes[i];
                if (code < dict_size) {
                    batch_bytes += (uint32_t)lens[code];
                } else {
                    atomicMax(status, 1u);
                }
            }
        }
    }
#pragma unroll
    for (uint8_t offset = 16; offset > 0; offset >>= 1) {
        batch_bytes += __shfl_down_sync(0xffffffffu, batch_bytes, offset);
    }

    __shared__ uint64_t warp_sums[ONPAIR_WARPS_PER_BLOCK];
    __shared__ uint64_t warp_excl[ONPAIR_WARPS_PER_BLOCK];
    __shared__ typename OnPairPrefixOp::TempStorage prefix_storage;
    if (lane == 0) {
        warp_sums[warp] = (uint64_t)batch_bytes;
    }
    __syncthreads();

    // The first warp scans the tile's batch sums and resolves the running
    // prefix of all preceding tiles via decoupled look-back.
    if (warp == 0) {
        const uint64_t lane_sum = (lane < ONPAIR_WARPS_PER_BLOCK) ? warp_sums[lane] : 0;
        uint64_t inclusive = lane_sum;
#pragma unroll
        for (uint8_t offset = 1; offset < 32; offset <<= 1) {
            const uint64_t shifted = __shfl_up_sync(0xffffffffu, inclusive, offset);
            if (lane >= offset) {
                inclusive += shifted;
            }
        }
        const uint64_t aggregate = __shfl_sync(0xffffffffu, inclusive, 31);

        uint64_t prefix = 0;
        if (tile_idx == 0) {
            if (lane == 0) {
                tile_state.SetInclusive(0, aggregate);
            }
        } else {
            // Collective over the first warp, as BlockScan would invoke it.
            // The callback's return value is only defined in lane 0 (its
            // look-back window reduction is a WarpReduce); broadcast it, the
            // same way BlockScan shares the prefix before applying it.
            OnPairPrefixOp prefix_op(tile_state, prefix_storage, SumU64(), tile_idx);
            const uint64_t lane0_prefix = prefix_op(aggregate);
            prefix = __shfl_sync(0xffffffffu, lane0_prefix, 0);
        }
        if (lane < ONPAIR_WARPS_PER_BLOCK) {
            warp_excl[lane] = prefix + inclusive - lane_sum;
        }
    }
    __syncthreads();

    if (lane == 0 && batch < num_batches) {
        chunk_offsets[batch] = warp_excl[warp];
        // The trailing slot holds the total decoded byte count.
        if (batch == num_batches - 1) {
            chunk_offsets[num_batches] = warp_excl[warp] + warp_sums[warp];
        }
    }
}

int onpair_num_tiles(int64_t num_batches) {
    return static_cast<int>((num_batches + ONPAIR_WARPS_PER_BLOCK - 1) / ONPAIR_WARPS_PER_BLOCK);
}

} // namespace

// Query the look-back tile-state storage for `num_batches` batches.
extern "C" cudaError_t onpair_batch_offsets_temp_size(size_t *temp_bytes, int64_t num_batches) {
    if (num_batches < 0 || num_batches / ONPAIR_WARPS_PER_BLOCK >= INT_MAX) {
        return cudaErrorInvalidValue;
    }
    return OnPairTileState::AllocationSize(onpair_num_tiles(num_batches), *temp_bytes);
}

// Regenerate the OnPair decode kernel's per-batch output offsets on `stream`
// in one fused sweep, writing `chunk_offsets[0..num_batches]` where
// `chunk_offsets[b]` is the decoded byte count preceding batch `b` and
// `chunk_offsets[num_batches]` is the total. `code_width` selects the code
// stream's element size in bytes (1 or 2), so narrowed codes never need a
// widening pass. A code outside the dictionary raises `*status` to 1 and
// contributes zero bytes; the caller must check the flag before trusting the
// offsets.
extern "C" cudaError_t onpair_batch_offsets(void *d_temp,
                                            size_t temp_bytes,
                                            const void *codes,
                                            uint32_t code_width,
                                            const uint8_t *lens,
                                            uint32_t dict_size,
                                            uint64_t total_tokens,
                                            uint64_t *chunk_offsets,
                                            uint32_t *status,
                                            int64_t num_batches,
                                            cudaStream_t stream) {
    if (num_batches <= 0 || num_batches / ONPAIR_WARPS_PER_BLOCK >= INT_MAX) {
        return cudaErrorInvalidValue;
    }
    const int num_tiles = onpair_num_tiles(num_batches);

    OnPairTileState tile_state;
    cudaError_t err = tile_state.Init(num_tiles, d_temp, temp_bytes);
    if (err != cudaSuccess) {
        return err;
    }

    constexpr int INIT_THREADS = 128;
    const int init_blocks = (num_tiles + INIT_THREADS - 1) / INIT_THREADS;
    onpair_batch_offsets_init<<<init_blocks, INIT_THREADS, 0, stream>>>(tile_state, num_tiles);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }

    switch (code_width) {
    case 1:
        onpair_batch_offsets_sweep<uint8_t>
            <<<num_tiles, ONPAIR_BLOCK_THREADS, 0, stream>>>(static_cast<const uint8_t *>(codes),
                                                             lens,
                                                             dict_size,
                                                             total_tokens,
                                                             chunk_offsets,
                                                             status,
                                                             tile_state,
                                                             num_batches);
        break;
    case 2:
        onpair_batch_offsets_sweep<uint16_t>
            <<<num_tiles, ONPAIR_BLOCK_THREADS, 0, stream>>>(static_cast<const uint16_t *>(codes),
                                                             lens,
                                                             dict_size,
                                                             total_tokens,
                                                             chunk_offsets,
                                                             status,
                                                             tile_state,
                                                             num_batches);
        break;
    default:
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
