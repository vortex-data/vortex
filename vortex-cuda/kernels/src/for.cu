// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "scalar_kernel.cuh"
#include "types.cuh"

#include <type_traits>

#define FL_CHUNK_SIZE 1024 // Chunk size for Blocked FoR algorithm as per fastlanes.

// Frame-of-Reference operation: adds a reference value to each element.
template <typename T>
struct ForOp {
    T reference;
    static constexpr bool uses_index = false;

    __device__ inline T operator()(T value) const {
        return value + reference;
    }
};

// Each block has it's own set of FoR operations to apply to the chunk of data it processes.
template <typename T>
struct BlockedForOp{
    T* references;   // ceil(array-size/FL_CHUNK_SIZE) reference values, one for each block + tail
    uint64_t offset;
    static constexpr bool uses_index = true;

    __device__ inline T operator()(T value, uint64_t i) const {
        return static_cast<std::make_signed_t<T>>(
            static_cast<std::make_unsigned_t<T>>(value) + 
            static_cast<std::make_unsigned_t<T>>(references[(offset + i) / FL_CHUNK_SIZE])
        );
    }
};

// Macro to generate in-place FoR kernel for each type.
#define GENERATE_FOR_KERNEL(suffix, Type)                                                                    \
    extern "C" __global__ void for_##suffix(Type *__restrict values, Type reference, uint64_t array_len) {   \
        scalar_kernel_inplace(values, array_len, ForOp<Type> {reference});                                   \
    }

// Macro to generate FoR kernel with separate input/output buffers.
#define GENERATE_FOR_IN_OUT_KERNEL(suffix, Type)                                                             \
    extern "C" __global__ void for_in_out_##suffix(const Type *__restrict input,                             \
                                                   Type *__restrict output,                                  \
                                                   Type reference,                                           \
                                                   uint64_t array_len) {                                     \
        scalar_kernel(input, output, array_len, ForOp<Type> {reference});                                    \
    }

#define GENERATE_BLOCKED_FOR_KERNEL(suffix, Type)                                                            \
    extern "C" __global__ void for_blocked_##suffix(Type *__restrict values, Type *references,               \
        uint64_t offset,  uint64_t array_len) {                                                              \
        scalar_kernel_inplace(values, array_len, BlockedForOp<Type> {references, offset});                   \
    }

#define GENERATE_BLOCKED_FOR_IN_OUT_KERNEL(suffix, Type)                                                     \
    extern "C" __global__ void for_blocked_in_out_##suffix(const Type *__restrict input,                             \
                                                   Type *__restrict output,                                  \
                                                   Type *references,                                         \
                                                   uint64_t offset,                                          \
                                                   uint64_t array_len) {                                     \
        scalar_kernel(input, output, array_len, BlockedForOp<Type> {references, offset});                    \
    }

// In-place variants (modifies input buffer) - FoR is only used for integers
FOR_EACH_INTEGER(GENERATE_FOR_KERNEL)
FOR_EACH_INTEGER(GENERATE_BLOCKED_FOR_KERNEL)

// Separate input/output variants (preserves input buffer)
FOR_EACH_INTEGER(GENERATE_FOR_IN_OUT_KERNEL)
FOR_EACH_INTEGER(GENERATE_BLOCKED_FOR_IN_OUT_KERNEL)
