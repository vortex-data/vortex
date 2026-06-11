// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"

#include <limits.h>
#include <stdint.h>

namespace {

constexpr uint32_t MAX_INLINED_SIZE = 12;

struct BinaryView {
    uint32_t size;
    uint8_t inline_data[MAX_INLINED_SIZE];
};

struct BinaryViewRef {
    uint32_t size;
    uint8_t prefix[4];
    uint32_t buffer_index;
    uint32_t offset;
};

// Return whether a row is valid in a little-endian Arrow/Vortex bitmap, treating a missing (null)
// validity bitmap as all-valid.
__device__ bool is_valid(const uint8_t *const validity, uint64_t idx) {
    return validity == nullptr || ((validity[idx / 8] >> (idx % 8)) & 1);
}

// Initialize scan input from BinaryView sizes. Null rows contribute zero bytes so the gather kernel
// never needs to read their view payload.
//
// Threads stride by blockDim within the block's element range so warp accesses to views and scan
// stay coalesced.
__device__ void init_scan_device(const BinaryView *const __restrict views,
                                 const uint8_t *const __restrict validity,
                                 const uint64_t *const __restrict data_buffer_lens,
                                 int32_t *const __restrict scan,
                                 uint32_t *const status,
                                 uint64_t data_buffer_count,
                                 uint64_t len) {
    const uint64_t scan_len = len + 1;
    const uint64_t elements_per_block = blockDim.x * ELEMENTS_PER_THREAD;
    const uint64_t block_start = blockIdx.x * elements_per_block;
    const uint64_t block_stop = min(block_start + elements_per_block, scan_len);

    for (uint64_t idx = block_start + threadIdx.x; idx < block_stop; idx += blockDim.x) {
        if (idx >= len || !is_valid(validity, idx)) {
            scan[idx] = 0;
            continue;
        }

        const BinaryView view = views[idx];
        const uint32_t size = view.size;
        if (size > static_cast<uint32_t>(INT32_MAX)) {
            scan[idx] = 0;
            atomicMax(status, 2u);
            continue;
        }

        if (size > MAX_INLINED_SIZE) {
            const BinaryViewRef *const view_ref = reinterpret_cast<const BinaryViewRef *>(&view);
            const uint64_t buffer_index = static_cast<uint64_t>(view_ref->buffer_index);
            const uint64_t offset = static_cast<uint64_t>(view_ref->offset);
            // Both addends are u32 widened to u64, so the end position cannot wrap.
            const uint64_t end = offset + static_cast<uint64_t>(size);
            if (buffer_index >= data_buffer_count || end > data_buffer_lens[buffer_index]) {
                scan[idx] = 0;
                atomicMax(status, 1u);
                continue;
            }
        }

        scan[idx] = static_cast<int32_t>(size);
    }
}

// Detect i32 overflow of the CUB exclusive-sum offsets by checking signs. init_scan rejects
// per-row sizes above i32::MAX, so consecutive true prefix sums differ by less than 2^31 and the
// first overflowing prefix lands in [2^31, 2^32), which wraps to a negative offset in the scan
// output. No negative offset therefore proves no prefix overflowed.
__device__ void
validate_offsets_device(const int32_t *const __restrict offsets, uint32_t *const status, uint64_t scan_len) {
    const uint64_t elements_per_block = blockDim.x * ELEMENTS_PER_THREAD;
    const uint64_t block_start = blockIdx.x * elements_per_block;
    const uint64_t block_stop = min(block_start + elements_per_block, scan_len);

    for (uint64_t idx = block_start + threadIdx.x; idx < block_stop; idx += blockDim.x) {
        if (offsets[idx] < 0) {
            atomicMax(status, 2u);
        }
    }
}

__device__ uint64_t upper_bound_offsets(const int32_t *const offsets, uint64_t len, uint64_t value) {
    uint64_t first = 0;
    while (len > 0) {
        const uint64_t half = len / 2;
        const uint64_t mid = first + half;
        if (static_cast<uint64_t>(offsets[mid]) <= value) {
            first = mid + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }
    return first;
}

// Resolve the payload pointer for one view, pointing inline views at their global view bytes.
__device__ const uint8_t *
input_ptr(const BinaryView *const views, uint64_t row, const uint64_t *const data_buffer_ptrs) {
    const BinaryView *const view = views + row;
    if (view->size <= MAX_INLINED_SIZE) {
        return view->inline_data;
    }

    const BinaryViewRef *const view_ref = reinterpret_cast<const BinaryViewRef *>(view);
    return reinterpret_cast<const uint8_t *>(data_buffer_ptrs[view_ref->buffer_index]) + view_ref->offset;
}

// Copy BinaryView payload bytes into one contiguous Arrow Binary values buffer.
//
// Each thread owns a contiguous ELEMENTS_PER_THREAD output byte range, so the row lookup runs once
// per range and then advances sequentially. Byte stores from such strided ranges waste almost the
// whole memory transaction, so threads stage 16 output bytes in registers and write them with one
// vector store. Full chunks start at a multiple of ELEMENTS_PER_THREAD, keeping the stores aligned.
__device__ void gather_device(const BinaryView *const __restrict views,
                              const uint64_t *const __restrict data_buffer_ptrs,
                              const int32_t *const __restrict offsets,
                              uint8_t *const __restrict output,
                              uint64_t len,
                              uint64_t total_bytes) {
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t start = start_elem(worker, total_bytes);
    const uint64_t stop = stop_elem(worker, total_bytes);
    if (start == stop) {
        return;
    }

    uint64_t row = upper_bound_offsets(offsets, len + 1, start) - 1;
    uint64_t row_start = static_cast<uint64_t>(offsets[row]);
    uint64_t row_end = static_cast<uint64_t>(offsets[row + 1]);
    const uint8_t *input = input_ptr(views, row, data_buffer_ptrs);

    const auto next_byte = [&](uint64_t byte_idx) -> uint8_t {
        while (byte_idx >= row_end) {
            row++;
            row_start = static_cast<uint64_t>(offsets[row]);
            row_end = static_cast<uint64_t>(offsets[row + 1]);
            input = input_ptr(views, row, data_buffer_ptrs);
        }
        return input[byte_idx - row_start];
    };

    uint64_t byte_idx = start;
    for (; byte_idx + 16 <= stop; byte_idx += 16) {
        while (byte_idx >= row_end) {
            row++;
            row_start = static_cast<uint64_t>(offsets[row]);
            row_end = static_cast<uint64_t>(offsets[row + 1]);
            input = input_ptr(views, row, data_buffer_ptrs);
        }

        // Fast path: the group sits inside one row and the source allows word loads.
        const uint8_t *const src = input + (byte_idx - row_start);
        if (byte_idx + 16 <= row_end && (reinterpret_cast<uintptr_t>(src) & 3) == 0) {
            const uint32_t *const src_words = reinterpret_cast<const uint32_t *>(src);
            *reinterpret_cast<uint4 *>(output + byte_idx) =
                make_uint4(src_words[0], src_words[1], src_words[2], src_words[3]);
            continue;
        }

        uint32_t words[4];
#pragma unroll
        for (uint32_t word = 0; word < 4; word++) {
            uint32_t value = 0;
#pragma unroll
            for (uint32_t byte = 0; byte < 4; byte++) {
                const uint64_t lane = byte_idx + word * 4 + byte;
                value |= static_cast<uint32_t>(next_byte(lane)) << (byte * 8);
            }
            words[word] = value;
        }
        *reinterpret_cast<uint4 *>(output + byte_idx) = make_uint4(words[0], words[1], words[2], words[3]);
    }

    for (; byte_idx < stop; byte_idx++) {
        output[byte_idx] = next_byte(byte_idx);
    }
}

} // namespace

// Fill the CUB scan input with per-row binary lengths plus a final zero sentinel. A null validity
// pointer marks an all-valid array.
extern "C" __global__ void arrow_binary_init_scan(const BinaryView *const views,
                                                  const uint8_t *const validity,
                                                  const uint64_t *const data_buffer_lens,
                                                  int32_t *const scan,
                                                  uint32_t *const status,
                                                  uint64_t data_buffer_count,
                                                  uint64_t len) {
    init_scan_device(views, validity, data_buffer_lens, scan, status, data_buffer_count, len);
}

// Check that the scanned Arrow Binary offsets never overflowed the i32 range.
extern "C" __global__ void
arrow_binary_validate_offsets(const int32_t *const offsets, uint32_t *const status, uint64_t scan_len) {
    validate_offsets_device(offsets, status, scan_len);
}

// Gather inline and referenced BinaryView payloads into Arrow Binary's contiguous values buffer.
extern "C" __global__ void arrow_binary_gather(const BinaryView *const views,
                                               const uint64_t *const data_buffer_ptrs,
                                               const int32_t *const offsets,
                                               uint8_t *const output,
                                               uint64_t len,
                                               uint64_t total_bytes) {
    gather_device(views, data_buffer_ptrs, offsets, output, len, total_bytes);
}
