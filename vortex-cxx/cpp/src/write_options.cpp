// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/write_options.hpp"
#include "vortex/exception.hpp"

#include "rust/cxx.h"

namespace vortex {
void VortexWriteOptions::WriteArrayStream(ArrowArrayStream &stream, const std::string &path) {
    try {
        ffi::write_array_stream(std::move(impl_), reinterpret_cast<uint8_t *>(&stream), path);
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

} // namespace vortex