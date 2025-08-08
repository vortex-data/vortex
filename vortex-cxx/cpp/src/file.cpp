// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/file.hpp"
#include "vortex/scan.hpp"
#include "vortex/exception.hpp"
#include "scan_impl.hpp"
#include "rust/cxx.h"

namespace vortex {
// VortexFile implementation
struct VortexFile::Impl {
    rust::Box<ffi::VortexFile> rust_impl;

    explicit Impl(rust::Box<ffi::VortexFile> impl) : rust_impl(std::move(impl)) {
    }
};
VortexFile::VortexFile(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
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
        return VortexFile(std::make_unique<Impl>(ffi::open_file(path)));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

uint64_t VortexFile::RowCount() const {
    return ffi::file_row_count(*impl_->rust_impl);
}

ScanBuilder VortexFile::CreateScanBuilder() const {
    auto rust_builder = ffi::file_scan_builder(*impl_->rust_impl);
    return ScanBuilder(std::make_unique<ScanBuilder::Impl>(std::move(rust_builder)));
}

} // namespace vortex