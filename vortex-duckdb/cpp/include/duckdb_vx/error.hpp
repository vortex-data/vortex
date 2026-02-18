// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <exception>
#include <string>

#include "duckdb.h"
#include "duckdb_vx/error.h"

namespace vortex {
std::string IntoErrString(duckdb_vx_error error);
void SetError(duckdb_vx_error *error_out, std::string_view message);
duckdb_state HandleException(std::exception_ptr ex, duckdb_vx_error *error_out);
} // namespace vortex
