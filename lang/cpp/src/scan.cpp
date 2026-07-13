// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/array.hpp"
#include "vortex/common.hpp"
#include "vortex/dtype.hpp"
#include "vortex/error.hpp"
#include "vortex/scan.hpp"

#include <vortex.h>

#include <mutex>
#include <optional>

namespace vortex {

using detail::Access;
using detail::throw_on_error;

ArrowStream::ArrowStream(ArrowStream &&other) noexcept
    : session_(std::move(other.session_)), stream_(other.stream_) {
    other.stream_ = ArrowArrayStream {};
}

ArrowStream &ArrowStream::operator=(ArrowStream &&other) noexcept {
    if (this != &other) {
        if (stream_.release != nullptr) {
            stream_.release(&stream_);
        }
        session_ = std::move(other.session_);
        stream_ = other.stream_;
        other.stream_ = ArrowArrayStream {};
    }
    return *this;
}

ArrowStream::~ArrowStream() {
    if (stream_.release != nullptr) {
        stream_.release(&stream_);
    }
}

void Partition::Deleter::operator()(vx_partition *ptr) const noexcept {
    vx_partition_free(ptr);
}

Estimate Partition::row_count() const {
    vx_estimate raw {};
    vx_error *error = nullptr;
    vx_partition_row_count(handle_.get(), &raw, &error);
    throw_on_error(error);
    return Estimate(raw);
}

std::optional<Array> Partition::next() {
    vx_error *error = nullptr;
    const vx_array *array = vx_partition_next(handle_.get(), &error);
    throw_on_error(error);
    if (array == nullptr) {
        return std::nullopt;
    }
    return Access::adopt<Array>(array);
}

ArrowStream Partition::into_arrow_stream() && {
    ArrowArrayStream stream {};
    vx_error *error = nullptr;
    // Consumes handle even on error
    vx_partition_scan_arrow(Access::c_ptr(session_), handle_.release(), &stream, &error);
    throw_on_error(error);
    return Access::adopt<ArrowStream>(std::move(session_), stream);
}

void Scan::Deleter::operator()(vx_scan *ptr) const noexcept {
    vx_scan_free(ptr);
}

DataType Scan::dtype() const {
    vx_error *error = nullptr;
    const vx_dtype *out = vx_scan_dtype(handle_.get(), &error);
    throw_on_error(error);
    return Access::adopt<DataType>(out);
}

std::optional<Partition> Scan::next_partition() {
    vx_partition *partition = nullptr;
    {
        const std::lock_guard<std::mutex> guard(*mutex_);
        vx_error *error = nullptr;
        partition = vx_scan_next_partition(handle_.get(), &error);
        throw_on_error(error);
    }
    if (partition == nullptr) {
        return std::nullopt;
    }
    return Access::adopt<Partition>(partition, session_);
}
} // namespace vortex
