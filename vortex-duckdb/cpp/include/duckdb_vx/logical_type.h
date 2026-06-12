// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

char *duckdb_vx_logical_type_stringify(duckdb_logical_type ty);
duckdb_logical_type duckdb_vx_logical_type_copy(duckdb_logical_type ty);

/// Creates a GEOMETRY logical type with the given CRS (Coordinate Reference System).
/// `crs` must be a NUL-terminated UTF-8 string. Pass an empty string for no CRS.
duckdb_logical_type duckdb_vx_create_geometry(const char *crs);

#ifdef __cplusplus /* End C ABI */
}
#endif
