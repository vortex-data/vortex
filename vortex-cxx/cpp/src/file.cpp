// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/file.hpp"
#include "vortex/scan.hpp"
#include "vortex/exception.hpp"
#include "rust/cxx.h"

namespace vortex {
VortexFile::VortexFile(rust::Box<ffi::VortexFile> impl) : impl_(std::move(impl)) {
}

VortexFile::VortexFile(VortexFile &&other) noexcept : impl_(std::move(other.impl_)) {
}

VortexFile &VortexFile::operator=(VortexFile &&other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

VortexFile::~VortexFile() = default;

VortexFile VortexFile::Open(const std::string &path) {
    try {
        return VortexFile(ffi::open_file(path));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

uint64_t VortexFile::RowCount() const {
    return ffi::file_row_count(*impl_);
}

ScanBuilder VortexFile::CreateScanBuilder() const {
    return ScanBuilder(ffi::file_scan_builder(*impl_));
}

} // namespace vortex