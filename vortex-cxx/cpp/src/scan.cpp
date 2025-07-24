// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/scan.hpp"
#include "vortex/exception.hpp"
#include "scan_impl.hpp"

namespace vortex {

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

ScanBuilder &ScanBuilder::WithRowRange(uint64_t row_range_start, uint64_t row_range_end) {
    try {
        ffi::scan_builder_with_row_range(*impl_->rust_impl, row_range_start, row_range_end);
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
    return *this;
}

ScanBuilder &ScanBuilder::WithLimit(uint64_t limit) {
    ffi::scan_builder_with_limit(*impl_->rust_impl, limit);
    return *this;
}

ArrowArrayStream ScanBuilder::IntoStream() {
    try {
        ArrowArrayStream stream;
        ffi::scan_builder_into_stream(std::move(impl_->rust_impl), reinterpret_cast<uint8_t *>(&stream));
        return stream;
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}
} // namespace vortex