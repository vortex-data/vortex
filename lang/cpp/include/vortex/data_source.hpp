// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include "vortex/common.hpp"
#include "vortex/estimate.hpp"
#include "vortex/scan.hpp"
#include "vortex/session.hpp"

#include <vortex.h>

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace vortex {

/**
 * A reference to one or more (possibly remote, possibly glob) paths.
 *
 * Creating a DataSource opens the first matched path to read the schema. All
 * other IO is deferred until a scan is requested. Multiple scans may be
 * requested from one data source.
 */
class DataSource {
public:
    static DataSource open(const Session &session, std::initializer_list<std::string_view> paths);
    static DataSource open(const Session &session, std::span<const std::string> paths);
    static DataSource open(const Session &session, std::span<std::string_view> paths);

    /**
     * Create a DataSource from an in-memory Vortex file. Borrows the buffer:
     * caller must keep it alive and unmodified while DataSource, Scans,
     * Partitions, and Arrays obtained from this buffer are alive.
     */
    static DataSource from_buffer(const Session &session, std::span<const std::byte> data);

    DataSource(const DataSource &other);
    DataSource(DataSource &&) noexcept = default;
    DataSource &operator=(const DataSource &other);
    DataSource &operator=(DataSource &&) noexcept = default;

    Estimate row_count() const;
    DataType dtype() const;
    Scan scan(const ScanOptions &options = {}) const;

private:
    friend struct detail::Access;
    DataSource(const vx_data_source *owned, Session session) : handle_(owned), session_(std::move(session)) {
    }

    struct Deleter {
        void operator()(const vx_data_source *ptr) const noexcept;
    };
    std::unique_ptr<const vx_data_source, Deleter> handle_;
    Session session_;
};
} // namespace vortex
