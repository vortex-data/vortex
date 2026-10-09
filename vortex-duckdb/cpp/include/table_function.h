// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once
#include "duckdb.h"
#include "table_filter.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct duckdb_bind_result_ *duckdb_bind_result;

// Add a result column to the bind info.
void duckdb_vx_tfunc_bind_result_add_column(duckdb_bind_result ffi_result,
                                            const char *name_str,
                                            size_t name_len,
                                            duckdb_logical_type ffi_type);

typedef struct duckdb_vx_string_map_ *duckdb_vx_string_map;
// Add a key-value pair to the string map
void duckdb_vx_string_map_insert(duckdb_vx_string_map map, const char *key, const char *value);

// Input data passed into the init_global and init_local callbacks.
typedef struct {
    const void *bind_data;

    /**
     * Projected columns that are requested to be read. These are not
     * all columns, only the ones DuckDB optimizer thinks we should read.
     */
    idx_t *column_ids;
    size_t column_ids_count;

    /**
     * Post filter projected columns. Our table function implements filter
     * pushdown so this list is a subset of columns referenced in column_ids
     * after filter pushdown and filter pruning. May be empty, in which case
     * column_ids should be used.
     * Indices in this list reference values from column_ids. I.e. if
     * column_ids=[1,5,6], projection_ids=[1], output column should be
     * column_ids[1] = 5
     *
     * Example usage:
     * https://github.com/duckdb/duckdb/blob/dc11eadd8f0a7c600f0034810706605ebe10d5b9/src/include/duckdb/function/table_function.hpp#L147
     */
    const idx_t *projection_ids;
    size_t projection_ids_count;

    duckdb_vx_table_filter_set filters;
    duckdb_client_context client_context;

    /**
     * Struct extract paths pushed down by DuckDB for the entries in
     * `column_ids` where DuckDB only needs a struct field
     * (`ColumnIndex::IsPushdownExtract()`).
     *
     * `column_extract_indexes` holds every path concatenated, in the order of
     * `column_ids`, and `column_extract_offsets` has `column_ids_count + 1`
     * entries giving each column the slice `[offsets[i], offsets[i + 1])`.
     * A column that is read whole has an empty slice. The indexes are struct
     * child positions, outermost first: `SELECT s.a.b` on `s = {a: {b: …}}`
     * yields `[0, 0]`.
     */
    const idx_t *column_extract_indexes;
    size_t column_extract_indexes_count;
    const size_t *column_extract_offsets;

    /**
     * For each entry in `column_ids`, the type DuckDB expects the scan to emit
     * (`ColumnIndex::GetScanType()`, which is the cast target when the query
     * casts the extracted field), or NULL when the column is read whole.
     * The types are borrowed from the bind data and only valid for the duration
     * of the init_global call.
     */
    const duckdb_logical_type *column_extract_types;
    size_t column_extract_types_count;
} duckdb_vx_tfunc_init_input;

// Result data returned from the cardinality callback.
typedef struct {
    idx_t estimated_cardinality;
    bool has_estimated_cardinality;
    idx_t max_cardinality;
    bool has_max_cardinality;
} duckdb_vx_node_statistics;

typedef struct {
    // Set only for strings and primitive types
    duckdb_value min;
    duckdb_value max;
    // upper bit: "length is set". lower 32 bits: DuckDB's max string length.
    // set only for strings
    uint64_t max_string_length;
    bool has_null;
    // owned column type
    duckdb_logical_type type;
} duckdb_column_statistics;

// File-level statistics of a written Vortex file, for the DuckLake
// WRITTEN_FILE_STATISTICS return path. Filled by Rust from the WriteSummary.
typedef struct {
    uint64_t row_count;
    uint64_t file_size_bytes;
    uint64_t footer_size_bytes;
    uint64_t num_columns;
} duckdb_vx_written_file_statistics;

// Per-column statistics of a written Vortex file.
typedef struct {
    // Owned values, null if absent; the caller must destroy them.
    duckdb_value min;
    duckdb_value max;
    bool has_null_count;
    uint64_t null_count;
    uint64_t num_values;
    bool has_column_size;
    uint64_t column_size_bytes;
    // Whether a NaN-count statistic was available (float columns), and whether it saw any NaN.
    bool has_nan_stat;
    bool contains_nan;
} duckdb_vx_written_column_statistics;

duckdb_state duckdb_vx_register_table_functions(duckdb_database ffi_db);
duckdb_state duckdb_vx_register_version_function(duckdb_database ffi_db, const char *version);

typedef struct duckdb_vx_agg_input_ *duckdb_vx_agg_input;
idx_t duckdb_vx_aggregate_len(duckdb_vx_agg_input ffi);
duckdb_vx_expr duckdb_vx_aggregate_at(duckdb_vx_agg_input ffi, idx_t index, idx_t *proj_idx);

#ifdef __cplusplus
}
#endif
