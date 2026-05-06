// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <string>
#include "duckdb_vx/error.h"
#include "duckdb_vx/duckdb_diagnostics.h"

DUCKDB_INCLUDES_BEGIN
#include "duckdb.h"
#include "duckdb/common/assert.hpp"
DUCKDB_INCLUDES_END

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
    std::string *const error_str = reinterpret_cast<std::string *>(error);
    std::string out = std::move(*error_str);
    duckdb_vx_error_free(error);
    return out;
}

duckdb_state SetError(duckdb_vx_error *error_out, std::string_view message) {
    D_ASSERT(error_out != nullptr && "SetError called with null error_out");
    *error_out = duckdb_vx_error_create(message.data(), message.size());
    return DuckDBError;
}
} // namespace vortex
