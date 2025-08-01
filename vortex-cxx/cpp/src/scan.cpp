// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/scan.hpp"
#include "vortex/exception.hpp"
#include "scan_impl.hpp"
#include "rust/cxx.h"

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

ScanBuilder &&ScanBuilder::WithRowRange(uint64_t row_range_start, uint64_t row_range_end) && {
    try {
        ffi::scan_builder_with_row_range(*impl_->rust_impl, row_range_start, row_range_end);
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
    return std::move(*this);
}

ScanBuilder &&ScanBuilder::WithLimit(uint64_t limit) && {
    ffi::scan_builder_with_limit(*impl_->rust_impl, limit);
    return std::move(*this);
}

ScanBuilder &&ScanBuilder::WithIncludeByIndex(const uint64_t *indices, std::size_t size) && {
    try {
        ffi::scan_builder_with_include_by_index(*impl_->rust_impl,
                                                rust::Slice<const uint64_t>(indices, size));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
    return std::move(*this);
}

ScanBuilder &&ScanBuilder::WithOutputSchema(ArrowSchema &output_schema) && {
    try {
        ffi::scan_builder_with_output_schema(*impl_->rust_impl, reinterpret_cast<uint8_t *>(&output_schema));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
    return std::move(*this);
}

ArrowArrayStream ScanBuilder::IntoStream() && {
    try {
        ArrowArrayStream stream;
        ffi::scan_builder_into_stream(std::move(impl_->rust_impl), reinterpret_cast<uint8_t *>(&stream));
        return stream;
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

StreamDriver ScanBuilder::IntoStreamDriver() && {
    try {
        rust::Box<ffi::ThreadsafeCloneableReader> reader =
            ffi::scan_builder_into_threadsafe_cloneable_reader(std::move(impl_->rust_impl));
        return StreamDriver(std::make_unique<StreamDriver::Impl>(std::move(reader)));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

StreamDriver::StreamDriver(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}

StreamDriver::StreamDriver(StreamDriver &&other) noexcept : impl_(std::move(other.impl_)) {
}

StreamDriver &StreamDriver::operator=(StreamDriver &&other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

StreamDriver::~StreamDriver() = default;

struct StreamDriver::Impl {
    rust::Box<ffi::ThreadsafeCloneableReader> rust_impl;

    explicit Impl(rust::Box<ffi::ThreadsafeCloneableReader> impl) : rust_impl(std::move(impl)) {
    }
};

ArrowArrayStream StreamDriver::CreateArrayStream() const {
    ArrowArrayStream stream;
    ffi::threadsafe_cloneable_reader_clone_a_stream(*impl_->rust_impl, reinterpret_cast<uint8_t *>(&stream));
    return stream;
}
} // namespace vortex