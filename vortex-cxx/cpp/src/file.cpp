// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/file.hpp"
#include "vortex/io.hpp"
#include "vortex/scan.hpp"
#include "vortex/exception.hpp"
#include "rust/cxx.h"

namespace vortex {

VortexFile VortexFile::Open(const uint8_t *data, size_t length) {
    try {
        rust::Slice<const uint8_t> slice(data, length);
        return VortexFile(ffi::open_file_from_buffer(slice));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

VortexFile VortexFile::Open(const std::string &path) {
    try {
        return VortexFile(ffi::open_file(path));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

VortexFile VortexFile::OpenSeekable(std::unique_ptr<io::VortexReadAt> reader) {
    try {
        return VortexFile(ffi::open_with_read_at(std::move(reader)));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

uint64_t VortexFile::RowCount() const {
    return impl_->row_count();
}

ScanBuilder VortexFile::CreateScanBuilder() const {
    return ScanBuilder(impl_->scan_builder());
}

} // namespace vortex