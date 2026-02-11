// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "common.h"

// Column conversion utilities for ClickHouse integration.
//
// These functions provide type conversion between ClickHouse column types
// and Vortex array types.

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Type Information
// =============================================================================

/// Get the size in bytes for a ClickHouse type.
///
/// @param clickhouse_type ClickHouse type string (e.g., "Int64", "Float64").
/// @return Size in bytes for the type, or 0 for variable-length types or errors.
size_t vortex_clickhouse_type_size(const char* clickhouse_type);

/// Check if a ClickHouse type is fixed-width.
///
/// @param clickhouse_type ClickHouse type string.
/// @return 1 if fixed-width, 0 if variable-length or error.
int32_t vortex_clickhouse_type_is_fixed(const char* clickhouse_type);

/// Check if a ClickHouse type is nullable.
///
/// @param clickhouse_type ClickHouse type string (e.g., "Nullable(Int64)").
/// @return 1 if nullable, 0 if not nullable.
int32_t vortex_clickhouse_type_is_nullable(const char* clickhouse_type);

/// Get the inner type of a Nullable type.
///
/// @param clickhouse_type ClickHouse type string (e.g., "Nullable(Int64)").
/// @param inner_type Buffer to write the inner type string.
/// @param buffer_size Size of the buffer.
/// @return Length of the inner type string, or negative on error.
int32_t vortex_clickhouse_type_unwrap_nullable(
    const char* clickhouse_type,
    char* inner_type,
    size_t buffer_size
);

// =============================================================================
// Column Data Helpers
// =============================================================================

/// Calculate the required buffer size for exporting a column.
///
/// For fixed-width types, this returns num_rows * type_size.
/// For variable-length types, use vortex_scanner_column_data_size() instead.
///
/// @param clickhouse_type ClickHouse type string.
/// @param num_rows Number of rows.
/// @return Required buffer size in bytes, or 0 on error.
size_t vortex_column_buffer_size(const char* clickhouse_type, size_t num_rows);

#ifdef __cplusplus
}
#endif
