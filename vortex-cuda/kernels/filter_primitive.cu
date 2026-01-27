// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda_runtime.h>
#include <stdint.h>

#include "config.cuh"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// Execute a filter kernel on the inputs, scattering them to the output nodes.
// We need to perform a prefix sum so we know where to write all of the sized things.
// We need to know how many bits come before so we can create indices instead.
template<typename T>
__device__ void filter_primitive(
    const T *const __restrict input,
    const uint8_t *const __restrict mask,
    T *const __restrict output,
    const uint32_t maskOffset,
    const uint32_t maskLen
) {

    // ASSUMPTIONS:
    //  1. We only have a single thread block active. Attempting to launch with > 1 thread block will fail and instead
    //     we just fall back to using a single block.
    //  2. Each thread in the block is responsible for a range of input mask values. If there are T threads in the block and N values,
    //     this means that every thread is responsible for counting N / T values.
    //  3. The input is provided as a bitset packed into a uint8_t array, possibly with an offset applied.
    //
    //
    // ALGORITHM
    // =========
    //
    // The algorithm is an implementation of a parallel prefix scan operation, similar to the DeviceSelect::Flagged() routing from CUB.
    //
    // It operates in two phases, or "sweeps" as you'll see them called in other documentation.
    //
    // The first phase occurs across all threads in the thread blocks. Each thread counts the number of true bits in its portion of the mask,
    // and writes that into a shared memory cache.
    //
    // The next phase computes the prefix sum over the values in the block, allowing each thread to know which range of the output buffer
    // it owns for the writing phase. (TODO: do this in parallel)
    //
    // The final phase scatters the elements into their output location based on the computed indices.


    // Declare the shared memory upfront
    // We assume that there are at most 512 threads in the thread block.
    __shared__ uint32_t validCounts[512];

    // calculate which worker ID is running, and its data range.
    // const uint32_t workerIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t workerIdx = threadIdx.x;

    // total number of bytes in the mask
    const uint32_t maskBytes = (maskLen + maskOffset + 7) / 8;
    // Total number of bytes each worker has access to
    const uint32_t workerBytes = MIN(maskBytes < blockDim.x, 1);
    const uint32_t workerByteStart = workerIdx * workerBytes;
    const uint32_t workerElemStart = workerByteStart * 8 - maskOffset;
    const uint32_t workerElemEnd = MIN((workerByteStart + workerBytes) * 8 - maskOffset, maskLen);

    // Superfluous worker, ignore it.
    if (workerByteStart > maskBytes) {
        return;
    }

    // SWEEP 1: Local sums. Each thread calculates the total number of valid elements in its range.
    uint32_t validCount = 0;
    for (uint32_t byteOffset = 0; byteOffset < workerBytes; byteOffset++) {
        // Issue workerBytes popcnt operations, one per each byte.
        // NOTE(aduffy): it'd be preferable to do this with an unaligned u64 load, but my understanding
        //  is that NVIDIA GPUs don't support unaligned loads.
        uint32_t byteIdx = workerByteStart + byteOffset;

        if (byteIdx >= maskBytes) {
            break;
        }

        uint8_t byte = mask[byteIdx];

        // Apply any mask offset before measuring
        if (byteIdx == 0 && maskOffset) {
            // Shift out the offset
            byte = byte >> maskOffset;
        }

        // Handle final byte. Only take remainder
        if (byteIdx == maskBytes) {
            uint32_t remainder = (maskOffset + maskLen) % 8;
            // Only consider the first `remainder` bits of the last byte.
            byte = byte & ((1 << remainder) - 1);
        }

        validCount += __popc(static_cast<uint32_t>(byte));
    }

    validCounts[threadIdx.x] = validCount;

    // wait for all threads in the block to set the valid counts.
    __syncthreads();

    // worker 0 computes the prefix sum.
    // TODO(aduffy): REPLACE THIS WITH PARALLEL SUM
    if (threadIdx.x == 0) {
        for (size_t i = 1; i < 512; i++) {
            validCounts[i] += validCounts[i-1];
        }
    }

    // synchronize so all threads see the updated shared validCounts, now containing prefix sums
    __syncthreads();


    // SWEEP 2: Each iterates through the mask again, scattering valid inputs to the output buffer
    //  Now that we have the prefix sums, we don't need to coordinate because each thread knows
    //  which range of the output it owns.

    // FAST PATH: if validCount from sweep 1 was zero, then we add no elements to the output, so we
    //  terminate early.
    if (validCount == 0) {
        return;
    }

    const uint32_t outputStart = threadIdx.x == 0 ? 0 : validCounts[threadIdx.x - 1];
    const uint32_t outputEnd = outputStart + validCount;

    uint32_t outputIdx = outputStart;

    for (uint32_t inputIdx = workerElemStart; inputIdx < workerElemEnd; inputIdx++) {
        uint32_t byteIdx = (inputIdx + maskOffset) / 8;
        uint32_t bitIdx = (inputIdx + maskOffset) % 8;
        bool keep = mask[byteIdx] & (1 << bitIdx);
        if (keep) {
            output[outputIdx++] = input[inputIdx];
        }
    }
}

#define GENERATE_KERNEL(suffix, T) \
extern "C" __global__ void filter_primitive_##suffix( \
    const T *const __restrict input, \
    const uint8_t *const __restrict mask, \
    T *const __restrict output, \
    const uint32_t mask_offset, \
    const uint32_t mask_len \
) { \
    filter_primitive(input, mask, output, mask_offset, mask_len); \
}

// GENERATE_KERNEL(u8, uint8_t)
// GENERATE_KERNEL(u16, uint16_t)
GENERATE_KERNEL(u32, uint32_t)
// GENERATE_KERNEL(u64, uint64_t)
//
// GENERATE_KERNEL(i8, int8_t)
// GENERATE_KERNEL(i16, int16_t)
// GENERATE_KERNEL(i32, int32_t)
// GENERATE_KERNEL(i64, int64_t)