// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Shared type definitions between CUDA and Rust for the dynamic dispatch kernel.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Source ops: Fills shared memory from input (exactly one, required)
union SourceParams {
    struct BitunpackParams {
        uint8_t bit_width;
    } bitunpack;
};

struct SourceOp {
    enum SourceOpCode { BITUNPACK } op_code;
    union SourceParams params;
};

/// Scalar ops: Element-wise transforms in registers (0 or more)
union ScalarParams {
    struct FoRParams {
        uint64_t reference;
    } frame_of_ref;
    struct AlpParams {
        float f;
        float e;
    } alp;
};

struct ScalarOp {
    enum ScalarOpCode { FOR, ZIGZAG, ALP } op_code;
    union ScalarParams params;
};

/// Dispatch plan: Complete pipeline passed to the kernel
#define MAX_SCALAR_OPS 8

struct DynamicDispatchPlan {
    struct SourceOp source;
    uint8_t num_scalar_ops;
    struct ScalarOp scalar_ops[MAX_SCALAR_OPS];
};

#ifdef __cplusplus
}
#endif
