// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/file.hpp"
#include "vortex/exception.hpp"

#include "rust/cxx.h"
#include "vortex-cxx/src/lib.rs.h"

namespace vortex {

struct ScanBuilder::Impl {
    rust::Box<ffi::VortexScanBuilder> rust_impl;

    explicit Impl(rust::Box<ffi::VortexScanBuilder> impl) : rust_impl(std::move(impl)) {
    }
};

struct VortexFile::Impl {
    rust::Box<ffi::VortexFile> rust_impl;

    explicit Impl(rust::Box<ffi::VortexFile> impl) : rust_impl(std::move(impl)) {
    }
};

// ScanBuilder implementation
ScanBuilder::ScanBuilder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}

ScanBuilder::ScanBuilder(ScanBuilder &&other) noexcept : impl_(std::move(other.impl_)) {
}

ScanBuilder &ScanBuilder::operator=(ScanBuilder &&other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

ScanBuilder::~ScanBuilder() = default;

ScanBuilder &ScanBuilder::SetFilter(std::string_view filter) {
    try {
        ffi::scan_builder_set_filter(
            *impl_->rust_impl,
            rust::Slice<const uint8_t>(reinterpret_cast<const uint8_t *>(filter.data()), filter.size()));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
    return *this;
}

ScanBuilder &ScanBuilder::SetLimit(uint64_t limit) {
    ffi::scan_builder_set_limit(*impl_->rust_impl, limit);
    return *this;
}

arrow::Result<std::shared_ptr<arrow::RecordBatchReader>> ScanBuilder::IntoStream() {
    try {
        ArrowArrayStream stream;
        ffi::scan_builder_into_stream(std::move(impl_->rust_impl), reinterpret_cast<uint8_t *>(&stream));
        return arrow::ImportRecordBatchReader(&stream);
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

std::pair<ArrowArray, ArrowSchema> ScanBuilder::IntoArray() {
    try {
        ArrowArray array;
        ArrowSchema schema;
        ffi::scan_builder_into_arrow(std::move(impl_->rust_impl), reinterpret_cast<uint8_t *>(&array),
                                     reinterpret_cast<uint8_t *>(&schema));
        return {array, schema};
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

// VortexFile implementation
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