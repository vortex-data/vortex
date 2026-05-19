// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// ============================================================================
// ANNOTATED COPY of `onpair_shmem_2tpt.cu`.
//
// This file is a line-for-line annotated version of the production kernel
// next door. Variable names are expanded and every block has commentary
// explaining *why* the code is shaped the way it is.
//
// The kernel symbol here is renamed (`onpair_shmem_2tpt_annotated`) so this
// file can coexist with the original in the same translation unit list
// without a duplicate-symbol error. Everything else — control flow,
// operation count, unroll factors, launch bounds, memory access widths,
// cache hints — is identical, so the SASS produced should match the
// original modulo register allocation noise from identifier renaming.
// Treat the renaming as syntactic only; do not "improve" anything here
// without first updating the production kernel.
//
// ----------------------------------------------------------------------------
// WHAT THE KERNEL DOES (high level)
//
// OnPair is a dictionary-based variable-length codec. The compressed stream
// is an array of `uint16_t codes`. To decode, you look each code up in a
// dictionary that stores the original bytes, and concatenate the results
// into the output buffer.
//
// Inputs:
//   codes[i]              — the dictionary code for token i (uint16).
//   dict_padded[code*16]  — token payload, padded to 16 bytes so we can
//                           load it as a single `uint4` vector load.
//   lens[code]            — the real (unpadded) length of that token,
//                           guaranteed ≤ 16. The upper (16 - lens[code])
//                           bytes of the padded entry are garbage.
//   chunk_offsets[chunk]  — prefix-summed *byte* offset into output where
//                           this chunk's 64-token group begins. The host
//                           computes this so each warp knows independently
//                           where its decoded bytes go.
//   total_tokens          — bound check for the tail chunk.
//
// Output:
//   output_bytes — fully decoded byte stream, contiguous.
//
// Parallelism shape:
//   * One warp (32 lanes) processes one 64-token chunk.
//   * Within the warp, each lane handles TWO tokens (hence "2 tokens per
//     thread" — "2tpt"). The pair is stride-32, not consecutive: lane L
//     owns tokens L and L+32 within the chunk.
//
// Why 2 tokens per thread? See the header on the original file. Short
// columns (mean token length ≈ 3-4 B) produce too few output bytes per
// 32-token chunk (~100 B), so only a third of the warp participates in
// the aligned-vector drain. Doubling the chunk roughly doubles the byte
// yield and reduces per-chunk fixed costs amortised over those bytes.
//
// ----------------------------------------------------------------------------
// FOUR PHASES PER WARP
//
//   Phase 1 — Per-lane load:
//     Each lane reads its two codes, fetches the corresponding padded
//     dictionary entries as `uint4` (16 B) vectors into registers, and
//     reads the real lengths.
//
//   Phase 2 — Warp scan of lengths:
//     Two inclusive scans across the 32 lanes (one for the low-half
//     tokens, one for the high-half) compute the exclusive byte offset
//     for each token within the chunk. The chunk's total byte count
//     falls out of the scans for free.
//
//   Phase 3 — Stage to shared memory:
//     Each lane writes its (up to 16) token bytes into per-warp shared
//     scratch at its exclusive offset. We use a byte ladder rather than
//     a `memcpy` so the dictionary payload stays in registers (`uint4`
//     pulled apart per byte via `reinterpret_cast` on a stack-local).
//
//   Phase 4 — Aligned drain to global:
//     The warp drains shared → global as one head segment (≤15 byte
//     stores to reach the next 16-B aligned address in output), then a
//     run of aligned `uint4` stores for the body, then a tail segment
//     of ≤15 byte stores for the remainder. Using `__stcs` (streaming
//     cache hint) tells the L1/L2 not to keep the lines hot, since the
//     producer never reads them back.
// ============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// Maximum number of warps per CTA we'll ever launch with. The shared-memory
// allocation below sizes itself to this so we can support any launch shape
// up to 16 warps = 512 threads per block. The launch is `<<<.., 512>>>` in
// practice (see `__launch_bounds__` below).
#define WARPS_PER_BLOCK_MAX 16u

// Per-warp shared-memory scratch budget, in bytes.
//   - 64 tokens × 16 B (padded payload) = 1024 B max useful payload.
//   - We also need up to 15 bytes of slack at the front so the warp can
//     "shift" the start of its data inside the scratch to align the
//     eventual drain with global output (see the head_padding calculation).
// Rounded up to a 16-byte multiple (1024 + 32 = 1056) so each warp's
// scratch base is 16-B aligned, which matters for the `uint4` body drain.
#define WARP_BUF_BYTES 1056u

// ---------------------------------------------------------------------------
// Warp-wide inclusive scan over uint32.
//
// Implements the classic Kogge-Stone (a.k.a. Hillis-Steele) pattern using
// warp shuffles. After 5 doubling steps, every lane holds the inclusive
// prefix-sum of `x` across lanes [0..lane].
//
// `mask = 0xffffffff` means "all 32 lanes participate" — required by the
// `_sync` intrinsics on Volta+ for correctness even when the warp is
// fully active (the compiler/driver use this as a barrier predicate).
//
// Note: lanes with len = 0 (inactive tail tokens) contribute nothing and
// the scan still produces correct results, so we don't need a separate
// "is this lane active" mask. The scan output at lane 31 is therefore
// always the total warp byte count.
// ---------------------------------------------------------------------------
__device__ inline uint32_t warp_inclusive_scan_u32_annotated(uint32_t x, int lane_id) {
    constexpr unsigned ALL_LANES_MASK = 0xffffffffu;
#pragma unroll
    for (int shuffle_offset = 1; shuffle_offset < 32; shuffle_offset <<= 1) {
        // Pull the value from `shuffle_offset` lanes below us. Lanes
        // [0..offset) receive their own value back from `__shfl_up_sync`
        // (undefined per-spec but in practice their own `x`), which is
        // why the predicate gates the accumulation.
        uint32_t neighbor_below = __shfl_up_sync(ALL_LANES_MASK, x, shuffle_offset);
        if (lane_id >= shuffle_offset) {
            x += neighbor_below;
        }
    }
    return x;
}

// ---------------------------------------------------------------------------
// Kernel entry.
//
// `__launch_bounds__(512, 4)` tells the compiler:
//   - Max threads per block = 512 (16 warps).
//   - Min resident CTAs per SM = 4 (so 4 × 512 = 2048 threads = full SM
//     occupancy on Volta/Ampere/Hopper).
// This caps the per-thread register count so we hit the occupancy target.
// Don't change these without re-tuning — the kernel was profiled with this
// budget in mind.
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(512, 4) void onpair_shmem_2tpt_annotated(
    const uint16_t *__restrict codes,
    const uint64_t *__restrict chunk_offsets,
    const uint8_t *__restrict dict_padded,
    const uint8_t *__restrict lens,
    uint8_t *__restrict output_bytes,
    uint64_t total_tokens) {

    constexpr unsigned ALL_LANES_MASK = 0xffffffffu;

    // ----- Indexing: who am I in the warp / block / grid? -----
    const int lane_id = threadIdx.x & 31;          // 0..31 within warp
    const uint32_t warp_id_in_block = threadIdx.x >> 5;
    const uint32_t warps_per_block = blockDim.x >> 5;
    // Global chunk id: one warp = one 64-token chunk.
    const uint64_t chunk_id =
        (uint64_t)blockIdx.x * (uint64_t)warps_per_block
        + (uint64_t)warp_id_in_block;

    // Tail bound: if this warp's whole chunk starts past the end, bail.
    // We do *not* early-out individual lanes here — we still need the
    // warp to participate in scans and shuffles. Bound checks are
    // applied per-lane on the actual token indices below.
    if (chunk_id * 64u >= total_tokens) {
        return;
    }

    // ----- Shared-memory scratch (one slab per warp) -----
    // All warps in the block share one declaration; each warp indexes
    // into its own slab via `warp_id_in_block`.
    __shared__ __align__(16) uint8_t shared_scratch_all_warps[WARPS_PER_BLOCK_MAX * WARP_BUF_BYTES];
    uint8_t *warp_scratch_base = &shared_scratch_all_warps[warp_id_in_block * WARP_BUF_BYTES];

    // ----- Phase 1: per-lane load -----
    // Each lane handles two tokens at indices `low_token_idx` and
    // `high_token_idx`. The stride-32 layout (rather than consecutive
    // pairs like 2L and 2L+1) is deliberate: it makes the per-lane
    // exclusive offsets coming out of the two warp scans monotonic in
    // lane id, which matches the linear shared-memory layout we write
    // into below. With consecutive pairs the writes would interleave
    // and you'd need a more complex scan over interleaved lengths.
    const uint64_t low_token_idx = chunk_id * 64u + (uint64_t)lane_id;
    const uint64_t high_token_idx = low_token_idx + 32u;
    const bool low_in_range = (low_token_idx < total_tokens);
    const bool high_in_range = (high_token_idx < total_tokens);

    // Initialise both payload registers to zero so out-of-range lanes
    // contribute len = 0 (and therefore no bytes) to the scan/drain.
    uint4 low_payload = make_uint4(0u, 0u, 0u, 0u);
    uint4 high_payload = make_uint4(0u, 0u, 0u, 0u);
    uint32_t low_len = 0u;
    uint32_t high_len = 0u;

    if (low_in_range) {
        const uint32_t low_code = (uint32_t)codes[low_token_idx];
        // 16-B aligned vector load of the padded dictionary entry.
        low_payload = *reinterpret_cast<const uint4 *>(dict_padded + (size_t)low_code * 16u);
        low_len = (uint32_t)lens[low_code];
    }
    if (high_in_range) {
        const uint32_t high_code = (uint32_t)codes[high_token_idx];
        high_payload = *reinterpret_cast<const uint4 *>(dict_padded + (size_t)high_code * 16u);
        high_len = (uint32_t)lens[high_code];
    }

    // ----- Phase 2: two inclusive scans over lengths -----
    // Scan #1 covers the low half (tokens 0..31 within the chunk).
    //   inclusive_low[L]  = sum of low_len[0..L]    (inclusive of L)
    //   exclusive_low[L]  = sum of low_len[0..L)    (exclusive of L)
    //                     = inclusive_low[L] - low_len[L]
    //   warp_low_total    = inclusive_low[31] = total bytes contributed
    //                       by the low half of the chunk.
    const uint32_t inclusive_sum_low =
        warp_inclusive_scan_u32_annotated(low_len, lane_id);
    const uint32_t exclusive_sum_low = inclusive_sum_low - low_len;
    const uint32_t warp_low_total =
        __shfl_sync(ALL_LANES_MASK, inclusive_sum_low, 31);

    // Scan #2 over the high half (tokens 32..63). It's independent of
    // the low scan; we just add `warp_low_total` to each high token's
    // offset when staging so the high half lands contiguously after
    // the low half in shared memory.
    const uint32_t inclusive_sum_high =
        warp_inclusive_scan_u32_annotated(high_len, lane_id);
    const uint32_t exclusive_sum_high = inclusive_sum_high - high_len;
    const uint32_t warp_high_total =
        __shfl_sync(ALL_LANES_MASK, inclusive_sum_high, 31);

    // Total bytes this warp will emit for its 64-token chunk.
    const uint32_t warp_total_bytes = warp_low_total + warp_high_total;

    // ----- Phase 3a: figure out the head padding for aligned drain -----
    // `output_start_byte` is the byte address in global memory where
    // this warp's data begins. It is NOT in general 16-B aligned, so a
    // straight uint4 drain wouldn't work — we'd misalign the stores.
    //
    // Trick: we pad the front of our shared scratch by `head_padding`
    // bytes, so that the byte at `warp_scratch[head_padding]` corresponds
    // to `output_bytes[output_start_byte]`. Then the byte at
    // `warp_scratch[16]` corresponds to the next 16-B-aligned global
    // address, and the body drain can be aligned uint4 stores.
    //
    // head_padding = ((-output_start_byte) mod 16). Computed via:
    //   (16 - (output_start_byte & 15)) & 15
    // which is 0 when already aligned, else 16 - misalignment.
    const uint64_t output_start_byte = chunk_offsets[chunk_id];
    const uint32_t head_padding_bytes =
        (16u - (uint32_t)(output_start_byte & 15u)) & 15u;

    // Shift `warp_scratch` backwards from the slab base by enough that
    // index `head_padding_bytes` lands on the aligned slot. Equivalent
    // to `warp_scratch_base + ((-head_padding_bytes) mod 16)`. Since
    // `warp_scratch_base` itself is 16-B aligned (the slab size 1056 is
    // a multiple of 16 and shared_scratch_all_warps is __align__(16)),
    // this gives us a "logical zero" that starts `head_padding_bytes`
    // before the first 16-B boundary in scratch.
    uint8_t *warp_scratch = warp_scratch_base + ((16u - head_padding_bytes) & 15u);

    // ----- Phase 3b: byte-write tokens to per-warp shared scratch -----
    //
    // The two halves are written contiguously:
    //   token low  (idx L)  → warp_scratch[exclusive_sum_low + j]            for j in [0, low_len)
    //   token high (idx L+32) → warp_scratch[warp_low_total + exclusive_sum_high + j]
    //
    // We use an unrolled byte ladder rather than memcpy(_, _, runtime_len)
    // because NVCC's lowering of memcpy with a runtime length spills the
    // dictionary entry to local memory (stack) and then issues per-byte
    // loads back. Keeping `low_payload` / `high_payload` in registers
    // and reinterpret-casting to a byte pointer-into-stack-slot makes
    // the compiler keep the bytes in registers/imm — much cheaper.
    //
    // The branch `if (j < (int)len)` is a register-resident predicate
    // and the loop is fully unrolled, so the conditional store costs
    // one predicated `STS` per iteration with no real branching.
    if (low_in_range) {
        const uint8_t *low_payload_bytes =
            reinterpret_cast<const uint8_t *>(&low_payload);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (j < (int)low_len) {
                warp_scratch[exclusive_sum_low + j] = low_payload_bytes[j];
            }
        }
    }
    if (high_in_range) {
        const uint8_t *high_payload_bytes =
            reinterpret_cast<const uint8_t *>(&high_payload);
        // High-half tokens start after the entire low-half block ends.
        const uint32_t high_dest_base = warp_low_total + exclusive_sum_high;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (j < (int)high_len) {
                warp_scratch[high_dest_base + j] = high_payload_bytes[j];
            }
        }
    }
    // Synchronise just the warp — Phase 4 reads back what Phase 3 wrote.
    // We don't need a __syncthreads since no inter-warp data flows here.
    __syncwarp();

    // ----- Phase 4: drain shared → global with head/body/tail -----
    //
    // The drain has three segments because `output_start_byte` is
    // unaligned by `head_padding_bytes`:
    //
    //   1. HEAD — up to 15 byte stores to fill the gap between
    //      `output_start_byte` and the next 16-B aligned global address.
    //      If the total bytes are shorter than `head_padding_bytes`, we
    //      cap the head segment at `warp_total_bytes` instead.
    //
    //   2. BODY — aligned 16-byte (uint4) stores. Each lane in turn
    //      grabs one 16-byte block from scratch and stores it. With 32
    //      lanes per warp and one __stcs per block, the warp processes
    //      512 bytes per iteration of the body loop.
    //
    //   3. TAIL — up to 15 byte stores for the leftover bytes after the
    //      last whole 16-byte body block.

    // Effective head length: min(head_padding_bytes, warp_total_bytes).
    // Done as a `select` rather than `min(...)` to keep the lowered
    // code identical to the original.
    const uint32_t head_bytes =
        head_padding_bytes < warp_total_bytes
            ? head_padding_bytes
            : warp_total_bytes;

    // First `head_bytes` lanes do one byte store each. Note we read
    // `warp_scratch[lane_id]` here: this works because `warp_scratch`
    // was shifted so its index 0 corresponds to `output_start_byte`.
    if ((uint32_t)lane_id < head_bytes) {
        output_bytes[output_start_byte + (uint64_t)lane_id] =
            warp_scratch[lane_id];
    }
    // If the head consumed everything (very small chunk), we're done.
    if (head_bytes >= warp_total_bytes) {
        return;
    }

    // Number of 16-byte body blocks we can do aligned. After the head,
    // we're at a 16-B aligned global address; we have
    // `warp_total_bytes - head_bytes` bytes left, of which the lower
    // `(... >> 4) << 4` form whole body blocks.
    const uint32_t body_vec_count = (warp_total_bytes - head_bytes) >> 4;

    // Body loop: lanes round-robin through 16-byte blocks. `__stcs`
    // is the "streaming" store cache hint — tells the cache hierarchy
    // that we won't read these lines back, so they can be evicted
    // aggressively without polluting L1/L2.
    for (uint32_t k = (uint32_t)lane_id; k < body_vec_count; k += 32u) {
        const uint32_t block_byte_offset = head_bytes + k * 16u;
        const uint4 block_vector =
            *reinterpret_cast<const uint4 *>(warp_scratch + block_byte_offset);
        __stcs(
            reinterpret_cast<uint4 *>(output_bytes + output_start_byte + block_byte_offset),
            block_vector);
    }

    // Tail: the remaining (warp_total_bytes - head_bytes) mod 16 bytes,
    // each handled by one lane.
    const uint32_t tail_start_offset = head_bytes + (body_vec_count << 4);
    if ((uint32_t)lane_id < warp_total_bytes - tail_start_offset) {
        output_bytes[output_start_byte
                     + (uint64_t)tail_start_offset
                     + (uint64_t)lane_id] =
            warp_scratch[tail_start_offset + lane_id];
    }
}
