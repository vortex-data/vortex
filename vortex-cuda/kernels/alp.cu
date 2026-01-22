// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "scalar_kernel.cuh"

// ALP (Adaptive Lossless floating-Point) decode operation.
// Converts integers to floats by multiplying by precomputed exponent factors.
// Formula: decoded = (float)encoded * f * e
// Where f = F10[exponents.f] and e = IF10[exponents.e] are passed directly.
template<typename EncodedT, typename FloatT>
struct AlpOp {
    FloatT f;  // F10[exponents.f] - power of 10
    FloatT e;  // IF10[exponents.e] - inverse power of 10

    __device__ __forceinline__ FloatT operator()(EncodedT value) const {
        return static_cast<FloatT>(value) * f * e;
    }
};

// Macro to generate ALP kernel for each type combination.
// Input is integer (encoded), output is float (decoded).
#define GENERATE_ALP_KERNEL(enc_suffix, float_suffix, EncType, FloatType) \
extern "C" __global__ void alp_##enc_suffix##_##float_suffix( \
    const EncType *__restrict encoded, \
    FloatType *__restrict decoded, \
    FloatType f, \
    FloatType e, \
    uint64_t array_len \
) { \
    scalar_kernel(encoded, decoded, array_len, AlpOp<EncType, FloatType>{f, e}); \
}

// f32 variants (ALP for f32 encodes as i32 or i64)
GENERATE_ALP_KERNEL(i32, f32, int32_t, float)
GENERATE_ALP_KERNEL(i64, f32, int64_t, float)

// f64 variants (ALP for f64 encodes as i64)
GENERATE_ALP_KERNEL(i64, f64, int64_t, double)
