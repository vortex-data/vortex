// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/writer.hpp"
#include "vortex/exception.hpp"
#include "rust/cxx.h"

namespace vortex {

void VortexWriter::PushArrayStream(ArrowArrayStream &stream) {
    try {
        ffi::writer_push_array_stream(*impl_, reinterpret_cast<uint8_t *>(&stream));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

uint64_t VortexWriter::BytesWritten() const {
    return ffi::writer_bytes_written(*impl_);
}

uint64_t VortexWriter::BufferedBytes() const {
    return ffi::writer_buffered_bytes(*impl_);
}

void VortexWriter::Finish() {
    try {
        ffi::writer_finish(std::move(impl_));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

} // namespace vortex
