// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * C ABI for registering a DuckDB MultiFileFunction-backed table function.
 *
 * Unlike duckdb_vx_tfunc_register (which wraps a single TableFunction), this exposes
 * DuckDB's templated MultiFileFunction<OP> machinery: file globbing, per-file readers,
 * hive partitioning, virtual columns, etc. are all driven by DuckDB itself; the
 * extension only supplies a per-format reader.
 *
 * Owned-pointer convention: every non-null pointer the extension returns is owned by
 * DuckDB and must be released by the corresponding free_* callback. Borrowed pointers
 * (passed in to callbacks) must not be freed.
 */
#pragma once

#include "duckdb_vx/data.h"
#include "error.h"
#include "table_function.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque, extension-owned. Lifetime is tied to the corresponding free_* callback.
typedef struct duckdb_vx_mff_options_ *duckdb_vx_mff_options;
typedef struct duckdb_vx_mff_bind_data_ *duckdb_vx_mff_bind_data;
typedef struct duckdb_vx_mff_global_ *duckdb_vx_mff_global;
typedef struct duckdb_vx_mff_local_ *duckdb_vx_mff_local;
typedef struct duckdb_vx_mff_reader_ *duckdb_vx_mff_reader;

// Opaque writers populated by the extension during bind.
typedef struct duckdb_vx_mff_schema_writer_ *duckdb_vx_mff_schema_writer;

// A single projected column passed to prepare_reader. `name` is borrowed for
// the duration of the call.
typedef struct {
    const char *name;
    size_t name_len;
} duckdb_vx_mff_column;

// Opaque writer for EXPLAIN/to_string output. Same shape as
// duckdb_vx_string_map but kept distinct for FFI hygiene.
typedef duckdb_vx_string_map duckdb_vx_mff_string_map;

/**
 * Append a column to the bind schema. The name is copied; the logical type is
 * cloned. Both arguments remain owned by the caller.
 */
void duckdb_vx_mff_schema_writer_add_column(duckdb_vx_mff_schema_writer writer,
                                            const char *name,
                                            size_t name_len,
                                            duckdb_logical_type type);

// vtable mirroring the subset of MultiFileReaderInterface + BaseFileReader we expose.
// All callbacks are required and must be non-null.
typedef struct {
    /** Function name, e.g. "read_vortex". Must outlive the registered function. */
    const char *name;

    // ---------------------------------------------------------------------
    // Options lifecycle
    // ---------------------------------------------------------------------

    /** Create a fresh, default options object. Called once per bind. */
    duckdb_vx_mff_options (*create_options)(duckdb_client_context ctx, duckdb_vx_error *error);
    /** Release options created by create_options. Must accept null. */
    void (*free_options)(duckdb_vx_mff_options options);

    // ---------------------------------------------------------------------
    // Bind lifecycle
    // ---------------------------------------------------------------------

    /**
     * Initialize bind data from options. Called once per bind, after options.
     * Takes ownership of `options` (must be freed via free_options if the
     * extension does not retain it).
     */
    duckdb_vx_mff_bind_data (*initialize_bind_data)(duckdb_vx_mff_options options,
                                                    duckdb_vx_error *error);
    /** Release bind data. Must accept null. */
    void (*free_bind_data)(duckdb_vx_mff_bind_data bind_data);

    /**
     * Bind the reader's schema. Called by DuckDB after the first file in the
     * file list is known. The extension should open the file (or a metadata-
     * only handle) and append result columns via the schema_writer.
     *
     * `first_file_path` is borrowed (not nul-terminated, length given).
     */
    void (*bind_reader)(duckdb_client_context ctx,
                        duckdb_vx_mff_bind_data bind_data,
                        const char *first_file_path,
                        size_t path_len,
                        duckdb_vx_mff_schema_writer schema_out,
                        duckdb_vx_error *error);

    // ---------------------------------------------------------------------
    // Per-query state lifecycle
    // ---------------------------------------------------------------------

    duckdb_vx_mff_global (*init_global)(duckdb_client_context ctx,
                                        duckdb_vx_mff_bind_data bind_data,
                                        duckdb_vx_error *error);
    void (*free_global)(duckdb_vx_mff_global global);

    duckdb_vx_mff_local (*init_local)(duckdb_vx_mff_global global);
    void (*free_local)(duckdb_vx_mff_local local);

    // ---------------------------------------------------------------------
    // Per-file reader lifecycle
    // ---------------------------------------------------------------------

    /**
     * Open a per-file reader. Called once per file when DuckDB first opens
     * that file for scanning.
     */
    duckdb_vx_mff_reader (*create_reader)(duckdb_client_context ctx,
                                          duckdb_vx_mff_global global,
                                          duckdb_vx_mff_bind_data bind_data,
                                          const char *file_path,
                                          size_t path_len,
                                          size_t file_idx,
                                          duckdb_vx_error *error);
    void (*free_reader)(duckdb_vx_mff_reader reader);

    /**
     * Configure the reader with the columns it should produce and any filters
     * pushed down by DuckDB. Called once per (reader, scan) pair before any
     * try_initialize_scan / scan calls. `projection` is the ordered list of
     * column names the output chunks must contain (one entry per chunk
     * column). `filters` may be null when no filters were pushed down.
     */
    void (*prepare_reader)(duckdb_vx_mff_reader reader,
                           const duckdb_vx_mff_column *projection,
                           size_t projection_count,
                           duckdb_vx_table_filter_set filters,
                           duckdb_vx_error *error);

    /**
     * Try to initialize a scan over `reader`. Returns true if a scan can begin,
     * false if the reader is exhausted. Called with the multi-file global lock
     * held; must not block on I/O.
     */
    bool (*try_initialize_scan)(duckdb_vx_mff_reader reader,
                                duckdb_vx_mff_global global,
                                duckdb_vx_mff_local local,
                                duckdb_vx_error *error);

    /**
     * Produce the next batch of data into `chunk_out`. Returns:
     *   - true with chunk size > 0  : more data may follow.
     *   - true with chunk size == 0 : reader is exhausted; DuckDB will move on.
     *   - false                     : an error occurred (see error_out).
     */
    bool (*scan)(duckdb_vx_mff_reader reader,
                 duckdb_vx_mff_global global,
                 duckdb_vx_mff_local local,
                 duckdb_data_chunk chunk_out,
                 duckdb_vx_error *error);

    /**
     * Get per-column statistics by name. Returns false if no stats are
     * available. Same convention as duckdb_vx_tfunc_vtab_t::statistics.
     */
    bool (*get_statistics)(duckdb_vx_mff_reader reader,
                           const char *col_name,
                           size_t name_len,
                           duckdb_column_statistics *stats_out);

    /** Scan progress within a file in [0.0, 100.0]. */
    double (*progress_in_file)(duckdb_vx_mff_reader reader);

    /**
     * Estimated cardinality across `file_count` files. Returning false leaves
     * cardinality unknown (DuckDB falls back to its own heuristic).
     */
    bool (*cardinality)(duckdb_vx_mff_bind_data bind_data,
                        size_t file_count,
                        duckdb_vx_node_statistics *out);

    /**
     * Populate the bind-time EXPLAIN map with key/value pairs (e.g. "Filters",
     * "Projection"). Called whenever DuckDB renders the table function in an
     * EXPLAIN output.
     */
    void (*to_string)(duckdb_vx_mff_bind_data bind_data, duckdb_vx_mff_string_map map);
} duckdb_vx_mff_vtab_t;

/**
 * Register the multi-file function described by `vtab` against `ffi_db`. The
 * vtab is copied into a TableFunctionInfo owned by the catalog, so the caller
 * may free it after this returns.
 */
duckdb_state duckdb_vx_mff_register(duckdb_database ffi_db, const duckdb_vx_mff_vtab_t *vtab);

#ifdef __cplusplus
}
#endif
