// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/common.hpp"
#include "vortex/data_source.hpp"
#include "vortex/dtype.hpp"
#include "vortex/error.hpp"

#include <vortex.h>

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vortex {

using detail::Access;
using detail::throw_on_error;
using detail::to_view;

void DataSource::Deleter::operator()(const vx_data_source *ptr) const noexcept {
    vx_data_source_free(ptr);
}

DataSource::DataSource(const DataSource &other)
    : handle_(vx_data_source_clone(other.handle_.get())), session_(other.session_) {
}

DataSource &DataSource::operator=(const DataSource &other) {
    if (this != &other) {
        handle_.reset(vx_data_source_clone(other.handle_.get()));
        session_ = other.session_;
    }
    return *this;
}

template <class T>
static DataSource open_paths(const Session &session, std::span<T> paths) {
    std::vector<vx_view> raw;
    raw.reserve(paths.size());
    for (const auto &path : paths) {
        raw.push_back(to_view(path));
    }

    vx_data_source_options options {};
    options.paths = raw.data();
    options.paths_len = raw.size();

    vx_error *error = nullptr;
    const vx_data_source *ds = vx_data_source_new(Access::c_ptr(session), &options, &error);
    throw_on_error(error);
    return Access::adopt<DataSource>(ds, session);
}

DataSource DataSource::open(const Session &session, std::initializer_list<std::string_view> paths) {
    std::span span {paths.begin(), paths.end()};
    return open_paths(session, span);
}

DataSource DataSource::open(const Session &session, std::span<const std::string> paths) {
    return open_paths(session, paths);
}

DataSource DataSource::open(const Session &session, std::span<std::string_view> paths) {
    return open_paths(session, paths);
}

DataSource DataSource::from_buffer(const Session &session, std::span<const std::byte> data) {
    vx_error *error = nullptr;
    const vx_data_source *ds =
        vx_data_source_new_buffer(Access::c_ptr(session), data.data(), data.size(), &error);
    throw_on_error(error);
    return Access::adopt<DataSource>(ds, session);
}

Estimate DataSource::row_count() const {
    vx_estimate raw {};
    vx_data_source_get_row_count(handle_.get(), &raw);
    return Estimate(raw);
}

DataType DataSource::dtype() const {
    return Access::adopt<DataType>(vx_data_source_dtype(handle_.get()));
}

Scan DataSource::scan(const ScanOptions &options) const {
    vx_scan_options raw {};
    raw.projection = options.projection.has_value() ? Access::c_ptr(*options.projection) : nullptr;
    raw.filter = options.filter.has_value() ? Access::c_ptr(*options.filter) : nullptr;
    if (options.row_range.has_value()) {
        raw.row_range_begin = options.row_range->begin;
        raw.row_range_end = options.row_range->end;
    }
    if (options.selection.has_value()) {
        raw.selection.idx = options.selection->indices.data();
        raw.selection.idx_len = options.selection->indices.size();
        raw.selection.include = static_cast<vx_scan_selection_include>(options.selection->kind);
    } else {
        raw.selection.include = VX_SELECTION_INCLUDE_ALL;
    }
    raw.limit = options.limit;
    raw.ordered = options.ordered;

    vx_estimate estimate {};
    vx_error *error = nullptr;
    vx_scan *scan = vx_data_source_scan(handle_.get(), &raw, &estimate, &error);
    throw_on_error(error);
    return Access::adopt<Scan>(scan, Estimate(estimate), session_);
}

} // namespace vortex
