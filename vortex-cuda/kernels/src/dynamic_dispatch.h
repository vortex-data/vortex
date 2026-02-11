// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Shared type definitions betwee CUDA and Rust for the dynamic dispatch kernel.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Op codes for the dynamic dispatch.
enum DynamicOpCode {
    FOR,
    ZIGZAG,
    BITUNPACK,
};

// Operation to pass to the dynamic dispatch kernel.
struct DynamicOp {
    enum DynamicOpCode op;
    uint64_t           param;
};

#ifdef __cplusplus
}
#endif
