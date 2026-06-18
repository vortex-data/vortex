// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb/function/scalar_function.hpp"

#include "duckdb_vx.h"

using namespace duckdb;

extern "C" const char *duckdb_vx_sfunc_name(duckdb_vx_sfunc ffi_func) {
    if (!ffi_func) {
        return nullptr;
    }
    auto func = reinterpret_cast<ScalarFunction *>(ffi_func);
    return func->name.c_str();
}

extern "C" duckdb_logical_type duckdb_vx_sfunc_return_type(duckdb_vx_sfunc ffi_func) {
    if (!ffi_func) {
        return nullptr;
    }
    auto func = reinterpret_cast<ScalarFunction *>(ffi_func);
    return reinterpret_cast<duckdb_logical_type>(&func->return_type);
}
