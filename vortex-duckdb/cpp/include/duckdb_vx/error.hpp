// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <exception>
#include <string>

#include "duckdb_vx/duckdb_diagnostics.h"

DUCKDB_INCLUDES_BEGIN
#include "duckdb.h"
DUCKDB_INCLUDES_END

#include "duckdb_vx/error.h"

namespace vortex {
std::string IntoErrString(duckdb_vx_error error);
duckdb_state SetError(duckdb_vx_error *error_out, std::string_view message);
} // namespace vortex
