// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

// Maximum length of inlined string.
constexpr int32_t MAX_INLINED_SIZE = 12;

// a byte buffer holding string data
using Buffer = uint8_t*;

// an i32 offsets buffer
using Offsets = int32_t*;

// The BinaryView type is how we access values.
union alignas(int64_t) BinaryView {
    InlinedBinaryView inlined;
    RefBinaryView ref;
}

struct InlinedBinaryView {
    int32_t size;
    uint8_t bytes[12];
}

struct RefBinaryView {
    int32_t size;
    uint8_t prefix[4];
    int32_t index;
    int32_t offset;
}
