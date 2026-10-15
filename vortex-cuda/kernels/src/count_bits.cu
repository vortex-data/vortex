// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Adds the number of set bits in `words[0..num_words)` to `*count`, which starts at zero.
//
// Launch config: grid_dim = (num_words / 2048, 1, 1), block_dim = (64, 1, 1).
extern "C" __global__ void
count_bits(const uint32_t *__restrict words, uint64_t num_words, unsigned long long *__restrict count) {
    constexpr uint32_t words_per_block = 2048;
    const uint64_t block_start = static_cast<uint64_t>(blockIdx.x) * words_per_block;
    uint32_t bits = 0;
    for (uint32_t offset = threadIdx.x; offset < words_per_block; offset += blockDim.x) {
        const uint64_t idx = block_start + offset;
        if (idx < num_words) {
            bits += __popc(words[idx]);
        }
    }
    bits = __reduce_add_sync(0xffffffff, bits);
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(count, static_cast<unsigned long long>(bits));
    }
}
