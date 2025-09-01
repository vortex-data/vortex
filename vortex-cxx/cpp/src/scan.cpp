// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/scan.hpp"
#include "vortex/exception.hpp"
#include "rust/cxx.h"
#include "vortex/expr.hpp"

namespace vortex {
ScanBuilder &ScanBuilder::WithFilter(expr::Expr &&expr) & {
    impl_->with_filter(std::move(expr).IntoImpl());
    return *this;
}
ScanBuilder &ScanBuilder::WithFilter(const expr::Expr &expr) & {
    impl_->with_filter_ref(expr.Impl());
    return *this;
}
ScanBuilder &&ScanBuilder::WithFilter(expr::Expr &&expr) && {
    impl_->with_filter(std::move(expr).IntoImpl());
    return std::move(*this);
}
ScanBuilder &&ScanBuilder::WithFilter(const expr::Expr &expr) && {
    impl_->with_filter_ref(expr.Impl());
    return std::move(*this);
}

ScanBuilder &ScanBuilder::WithProjection(expr::Expr &&expr) & {
    impl_->with_projection(std::move(expr).IntoImpl());
    return *this;
}
ScanBuilder &ScanBuilder::WithProjection(const expr::Expr &expr) & {
    impl_->with_projection_ref(expr.Impl());
    return *this;
}
ScanBuilder &&ScanBuilder::WithProjection(expr::Expr &&expr) && {
    impl_->with_projection(std::move(expr).IntoImpl());
    return std::move(*this);
}
ScanBuilder &&ScanBuilder::WithProjection(const expr::Expr &expr) && {
    impl_->with_projection_ref(expr.Impl());
    return std::move(*this);
}

ScanBuilder &ScanBuilder::WithRowRange(uint64_t row_range_start, uint64_t row_range_end) & {
    impl_->with_row_range(row_range_start, row_range_end);
    return *this;
}
ScanBuilder &&ScanBuilder::WithRowRange(uint64_t row_range_start, uint64_t row_range_end) && {
    impl_->with_row_range(row_range_start, row_range_end);
    return std::move(*this);
}

ScanBuilder &ScanBuilder::WithLimit(uint64_t limit) & {
    impl_->with_limit(limit);
    return *this;
}

ScanBuilder &&ScanBuilder::WithLimit(uint64_t limit) && {
    impl_->with_limit(limit);
    return std::move(*this);
}

ScanBuilder &ScanBuilder::WithIncludeByIndex(const uint64_t *indices, std::size_t size) & {
    impl_->with_include_by_index(rust::Slice<const uint64_t>(indices, size));
    return *this;
}

ScanBuilder &&ScanBuilder::WithIncludeByIndex(const uint64_t *indices, std::size_t size) && {
    impl_->with_include_by_index(rust::Slice<const uint64_t>(indices, size));
    return std::move(*this);
}

ScanBuilder &ScanBuilder::WithOutputSchema(ArrowSchema &output_schema) & {
    try {
        impl_->with_output_schema(reinterpret_cast<uint8_t *>(&output_schema));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
    return *this;
}

ScanBuilder &&ScanBuilder::WithOutputSchema(ArrowSchema &output_schema) && {
    try {
        impl_->with_output_schema(reinterpret_cast<uint8_t *>(&output_schema));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
    return std::move(*this);
}

ArrowArrayStream ScanBuilder::IntoStream() && {
    try {
        ArrowArrayStream stream;
        ffi::scan_builder_into_stream(std::move(impl_), reinterpret_cast<uint8_t *>(&stream));
        return stream;
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

StreamDriver ScanBuilder::IntoStreamDriver() && {
    try {
        rust::Box<ffi::ThreadsafeCloneableReader> reader =
            ffi::scan_builder_into_threadsafe_cloneable_reader(std::move(impl_));
        return StreamDriver(std::move(reader));
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

ArrowArrayStream StreamDriver::CreateArrayStream() const {
    ArrowArrayStream stream;
    impl_->clone_a_stream(reinterpret_cast<uint8_t *>(&stream));
    return stream;
}
} // namespace vortex