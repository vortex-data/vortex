// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cstdint>

// Copies host memory that the device reads through the host page tables (for example a mapped
// file on a coherent CPU-GPU system) into device memory. Callers align source and target to
// the same offset within 16 bytes, so the body moves 16-byte words.
extern "C" __global__ void
mapped_copy(const uint8_t *__restrict__ source, uint8_t *__restrict__ target, uint64_t len) {
    const uint64_t thread = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;

    const uint64_t misalignment = reinterpret_cast<uintptr_t>(source) % 16;
    const uint64_t head = misalignment == 0 ? 0 : (16 - misalignment < len ? 16 - misalignment : len);
    for (uint64_t i = thread; i < head; i += stride) {
        target[i] = source[i];
    }

    const uint64_t words = (len - head) / 16;
    const uint4 *const source_words = reinterpret_cast<const uint4 *>(source + head);
    uint4 *const target_words = reinterpret_cast<uint4 *>(target + head);
    for (uint64_t i = thread; i < words; i += stride) {
        target_words[i] = source_words[i];
    }

    for (uint64_t i = head + words * 16 + thread; i < len; i += stride) {
        target[i] = source[i];
    }
}
