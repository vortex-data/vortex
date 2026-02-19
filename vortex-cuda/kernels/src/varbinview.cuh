// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <stdint.h>

// Maximum length of inlined string.
constexpr int32_t MAX_INLINED_SIZE = 12;

// a byte buffer holding string data
typedef uint8_t *Buffer;

// an i32 offsets buffer
typedef int32_t *Offsets;

struct InlinedBinaryView {
    int32_t size;
    uint8_t data[12];
};

struct RefBinaryView {
    int32_t size;
    uint8_t prefix[4];
    int32_t index;
    int32_t offset;
};

// The BinaryView type is how we access values.
union alignas(int64_t) BinaryView {
    InlinedBinaryView inlined;
    RefBinaryView ref;
};
