// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cassert>
#include <exception>

#include "duckdb_vx/duckdb_diagnostics.h"
DUCKDB_INCLUDES_BEGIN
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/vector_buffer.hpp"
#include "duckdb/common/types/vector.hpp"
DUCKDB_INCLUDES_END

#include "duckdb_vx.h"

extern "C" duckdb_vx_error duckdb_vx_error_create(const char *message, size_t message_length) {
    return reinterpret_cast<duckdb_vx_error>(new std::string(message, message_length));
}

extern "C" const char *duckdb_vx_error_value(duckdb_vx_error err) {
    auto str = reinterpret_cast<std::string *>(err);
    return str->c_str();
}

extern "C" void duckdb_vx_error_free(duckdb_vx_error err) {
    auto str = reinterpret_cast<std::string *>(err);
    delete str;
}

namespace vortex {

std::string IntoErrString(duckdb_vx_error error) {
    if (!error) {
        return {};
    }
    return *reinterpret_cast<std::string *>(error);
}

duckdb_state SetError(duckdb_vx_error *error_out, std::string_view message) {
    assert(error_out != nullptr && "SetError called with null error_out");
    *error_out = duckdb_vx_error_create(message.data(), message.size());
    return DuckDBError;
}

duckdb_state HandleException(std::exception_ptr ex, duckdb_vx_error *error_out) {
    if (!ex) {
        return SetError(error_out, "Unknown error");
    }

    try {
        std::rethrow_exception(ex);
    } catch (const std::exception &caught) {
        return SetError(error_out, caught.what());
    } catch (...) {
        return SetError(error_out, "Unknown error");
    }
}
} // namespace vortex
