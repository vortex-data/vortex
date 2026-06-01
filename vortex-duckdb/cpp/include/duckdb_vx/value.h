// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

// Create a null value with a reference to a logical type.
duckdb_value duckdb_vx_value_create_null(duckdb_logical_type ty);

/// Creates a GEOMETRY value containing the given WKB bytes and CRS.
///
/// `wkb` points to `len` bytes of well-known-binary geometry data; the bytes are not validated.
/// `crs` must be a NUL-terminated UTF-8 string; pass NULL or an empty string for no CRS.
duckdb_value duckdb_vx_value_create_geometry(const uint8_t *wkb, idx_t len, const char *crs);

/// Extracts the raw WKB bytes from a GEOMETRY value as a duckdb_blob.
///
/// This bypasses the GEOMETRY -> BLOB default cast (which would require the spatial extension to
/// be loaded). The returned `data` pointer must be freed with `duckdb_free`. Returns `{nullptr, 0}`
/// if `value` is null or not a GEOMETRY value.
duckdb_blob duckdb_vx_value_get_geometry(duckdb_value value);

#ifdef __cplusplus /* End C ABI */
}
#endif
