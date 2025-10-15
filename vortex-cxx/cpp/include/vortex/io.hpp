// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <vector>
#include <cstdint>
#include "rust/cxx.h"

namespace vortex {

namespace io {

class VortexReadAt {
public:
    virtual ~VortexReadAt() = default;

    virtual std::vector<uint8_t> ReadAt(uint64_t pos, size_t len) const = 0;

    virtual uint64_t GetSize() const = 0;
};

/// TODO: Is there any better way to do this?
inline rust::Vec<uint8_t> read_at(const VortexReadAt &reader, uint64_t pos, size_t len) {
    auto data = reader.ReadAt(pos, len);
    rust::Vec<uint8_t> result;
    result.reserve(data.size());
    std::copy(data.begin(), data.end(), std::back_inserter(result));
    return result;
}

inline uint64_t get_size(const VortexReadAt &reader) {
    return reader.GetSize();
}

} /// namespace io
} /// namespace vortex
