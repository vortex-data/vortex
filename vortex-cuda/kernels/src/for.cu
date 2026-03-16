// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "scalar_kernel.cuh"
#include "types.cuh"

// Frame-of-Reference operation: adds a reference value to each element.
template <typename T>
struct ForOp {
    T reference;

    __device__ inline T operator()(T value) const {
        return value + reference;
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

// In-place variants (modifies input buffer) - FoR is only used for integers
FOR_EACH_INTEGER(GENERATE_FOR_KERNEL)

// Separate input/output variants (preserves input buffer)
FOR_EACH_INTEGER(GENERATE_FOR_IN_OUT_KERNEL)
