// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <exception>
#include <string>
#include <string_view>

#include "vortex_duckdb.h"

std::string IntoErrString(duckdb_vx_error error);
duckdb_state SetError(duckdb_vx_error *error_out, std::string_view message);

template <typename Fn>
duckdb_state TryWrap(duckdb_vx_error *error_out, Fn &&fn) noexcept {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    try {
        fn();
        return DuckDBSuccess;
    } catch (const std::exception &e) {
        if (error_out != nullptr) {
            try {
                return SetError(error_out, e.what());
            } catch (...) {
                return DuckDBError;
            }
        }
        return DuckDBError;
    } catch (...) {
        if (error_out != nullptr) {
            try {
                return SetError(error_out, "unknown C++ exception");
            } catch (...) {
                return DuckDBError;
            }
        }
        return DuckDBError;
    }
}
