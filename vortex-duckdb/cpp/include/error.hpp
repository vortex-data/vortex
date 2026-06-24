// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <string>

#include "vortex_duckdb.h"

std::string IntoErrString(duckdb_vx_error error);
duckdb_state SetError(duckdb_vx_error *error_out, std::string_view message);
