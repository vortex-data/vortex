// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include "vortex/array.hpp"
#include "vortex/dtype.hpp"
#include "vortex/session.hpp"

#include <vortex.h>

#include <memory>
#include <string_view>

namespace vortex {

/**
 * Writes arrays into a Vortex file.
 *
 * finish() writes the footer and finalizes the file.
 * Not calling finish() leaves file corrupted.
 */
class Writer {
public:
    static Writer open(const Session &session, std::string_view path, const DataType &dtype);

    Writer(const Writer &) = delete;
    Writer &operator=(const Writer &) = delete;
    Writer(Writer &&) noexcept = default;
    Writer &operator=(Writer &&) noexcept = default;

    /*
     * Append Array to output file.
     * Throws if "array"'s DataType doesn't match writer's DataType.
     */
    void push(const Array &array);

    /*
     * Write footer and finalize the file.
     * Throws on failure. Writer is closed afterwards and further uses throws.
     */
    void finish();

private:
    explicit Writer(vx_array_sink *sink) : handle_(sink) {
    }

    struct Deleter {
        void operator()(vx_array_sink *ptr) const noexcept;
    };
    std::unique_ptr<vx_array_sink, Deleter> handle_;
};
} // namespace vortex
