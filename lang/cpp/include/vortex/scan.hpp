// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include "vortex/array.hpp"
#include "vortex/common.hpp"
#include "vortex/estimate.hpp"
#include "vortex/expression.hpp"
#include "vortex/session.hpp"

#include <vortex.h>

#include <cstdint>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace vortex {

namespace detail {
// range-for support for Scan and Partition
template <class Source, class Item, auto Next>
class PullRange {
public:
    class iterator {
    public:
        using value_type = Item;
        using difference_type = std::ptrdiff_t;

        iterator() = default;
        iterator(Source *src, std::optional<Item> first) : src_(src), cur_(std::move(first)) {
        }

        Item &operator*() const {
            return *cur_;
        }
        iterator &operator++() {
            cur_ = Next(src_);
            return *this;
        }
        void operator++(int) {
            ++*this;
        }
        bool operator==(std::default_sentinel_t) const {
            return !cur_.has_value();
        }

    private:
        Source *src_ = nullptr;
        mutable std::optional<Item> cur_;
    };

    explicit PullRange(Source &src) : src_(&src) {
    }
    iterator begin() {
        return iterator(src_, Next(src_));
    }
    std::default_sentinel_t end() {
        return std::default_sentinel;
    }

private:
    Source *src_;
};
} // namespace detail

// Wrapper around ArrowArrayStream which releases stream in destructor
class ArrowStream {
public:
    ArrowStream(const ArrowStream &) = delete;
    ArrowStream &operator=(const ArrowStream &) = delete;
    ArrowStream(ArrowStream &&other) noexcept;
    ArrowStream &operator=(ArrowStream &&other) noexcept;
    ~ArrowStream();

    ArrowArrayStream *raw() noexcept {
        return &stream_;
    }

private:
    friend struct detail::Access;
    ArrowStream(Session session, ArrowArrayStream stream) noexcept
        : session_(std::move(session)), stream_(stream) {
    }

    Session session_;
    ArrowArrayStream stream_ {};
};

/**
 * An independent unit of scan work.
 *
 * Partition's methods are thread-unsafe: drive each partition from
 * one worker thread.
 * Calling methods of a moved-out Partition is UB.
 */
class Partition {
public:
    Partition(const Partition &) = delete;
    Partition &operator=(const Partition &) = delete;
    Partition(Partition &&) noexcept = default;
    Partition &operator=(Partition &&) noexcept = default;

    // Estimated row count. Throws if called after next().
    Estimate row_count() const;

    // Return next Array or nullopt when partition is exhausted.
    std::optional<Array> next();

    // range-for over Arrays
    auto batches() & {
        return detail::PullRange<Partition, Array, &Partition::next>(*this);
    }
    auto batches() && = delete;

    /*
     * Consume the partition into an Arrow stream.
     * Blocks until partition is drained.
     */
    ArrowStream into_arrow_stream() &&;

private:
    friend struct detail::Access;
    Partition(vx_partition *owned, Session session) : handle_(owned), session_(std::move(session)) {
    }

    struct Deleter {
        void operator()(vx_partition *ptr) const noexcept;
    };
    std::unique_ptr<vx_partition, Deleter> handle_;
    Session session_;
};

/*
 * Row range [begin; end) to apply over filtering.
 * [0; 0) or convenience constant AllRows means "return all rows".
 */
struct RowRange {
    uint64_t begin = 0;
    uint64_t end = 0;
};

// Return all rows
constexpr RowRange AllRows = RowRange {0, 0};

struct Selection {
    enum class Kind {
        Include = VX_SELECTION_INCLUDE_RANGE,
        Exclude = VX_SELECTION_EXCLUDE_RANGE,
    };
    Kind kind = Kind::Include;
    std::vector<uint64_t> indices;
};

/**
 * Scan configuration. Fields are append-only and must be set via designated
 * initializers.
 * Default fields have reasonable behaviour: default projection returns all
 * fields, default filter, row_range, and selection don't filter etc.
 *
 * Example:
 *
 * DataSource ds = DataSource::open(session, {"file.vortex"});
 * Scan scan = ds.scan({.limit = 100});
 */
struct ScanOptions {
    std::optional<Expression> projection;
    std::optional<Expression> filter;
    /*
     * Row range [begin; end) to apply over filtering.
     * [0; 0) or convenience constant AllRows means "return all rows".
     */
    std::optional<RowRange> row_range;
    // Row-index filter applied after row_range.
    std::optional<Selection> selection;
    /*
     * Maximum number of rows to return. 0 means no limit.
     * You can either pass a limit or a filter but not both.
     */
    uint64_t limit = 0;
    // If true, return rows in storage order.
    bool ordered = false;
};

/**
 * A single traversal of a DataSource. A scan can be consumed only once.
 *
 * next_partition() is internally synchronized, give each partition to its own
 * worker thread.
 *
 * Calling methods of a moved-out Scan is UB.
 */
class Scan {
public:
    Scan(const Scan &) = delete;
    Scan &operator=(const Scan &) = delete;
    Scan(Scan &&) noexcept = default;
    Scan &operator=(Scan &&) noexcept = default;

    Estimate partition_count() const noexcept {
        return estimate_;
    }

    /**
     * Scan's dtype.
     *
     * Throws if called after next_partition().
     * UB if called in parallel with next_partition().
     */
    DataType dtype() const;

    // Next partition or nullopt when the scan is exhausted. Thread-safe.
    std::optional<Partition> next_partition();

    // range-for over partitions
    auto partitions() & {
        return detail::PullRange<Scan, Partition, &Scan::next_partition>(*this);
    }
    auto partitions() && = delete;

private:
    friend struct detail::Access;
    Scan(vx_scan *owned, Estimate estimate, Session session)
        : handle_(owned), mutex_(std::make_unique<std::mutex>()), estimate_(estimate),
          session_(std::move(session)) {
    }

    struct Deleter {
        void operator()(vx_scan *ptr) const noexcept;
    };
    std::unique_ptr<vx_scan, Deleter> handle_;
    std::unique_ptr<std::mutex> mutex_;
    Estimate estimate_;
    Session session_;
};
} // namespace vortex
