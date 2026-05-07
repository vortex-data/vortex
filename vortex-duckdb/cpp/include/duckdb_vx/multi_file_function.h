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
 * Lifecycle, mirroring DuckDB's Parquet reader:
 *   1. create_options / initialize_bind_data / bind_reader collect bind-time options,
 *      metadata, and schema.
 *   2. init_global / init_local create per-query and per-worker state.
 *   3. create_reader opens one file. DuckDB has dropped the global multi-file scheduling
 *      mutex before this call and holds a per-file mutex for this reader.
 *   4. prepare_reader maps the projection and filters onto the opened reader.
 *   5. try_initialize_scan is called with DuckDB's global multi-file scheduling mutex held.
 *      It must only claim one cheap unit of scan work into local state.
 *   6. prepare_scan runs outside that scheduling mutex and initializes local scan state
 *      for the work claimed by try_initialize_scan.
 *   7. scan drains the local state prepared by prepare_scan into DuckDB chunks.
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

// A single scan column passed to prepare_reader. `name` is borrowed for the
// duration of the call. `is_projected` distinguishes final output columns from
// filter-only scan columns.
typedef struct {
    const char *name;
    size_t name_len;
    uint64_t column_id;
    bool is_virtual;
    bool is_projected;
} duckdb_vx_mff_column;

// Exact per-file partition statistics for DuckDB's aggregate/statistics
// optimizer. Currently only row counts are exposed.
typedef struct {
    uint64_t row_count;
} duckdb_vx_mff_partition_stats;

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

    /** Whether DuckDB may pass pushed table filters to prepare_reader. */
    bool filter_pushdown;

    /** Whether DuckDB may omit filter-only columns from final table-scan output. */
    bool filter_prune;

    /**
     * Try to push a complex filter expression into bind data. Returns true when
     * the filter is handled exactly and DuckDB may remove the standalone filter.
     */
    bool (*pushdown_complex_filter)(duckdb_vx_mff_bind_data bind_data,
                                    duckdb_vx_expr expr,
                                    duckdb_vx_error *error_out);

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
    /** Clone bind data. Used when DuckDB rewrites plans, e.g. late materialization. */
    duckdb_vx_mff_bind_data (*clone_bind_data)(duckdb_vx_mff_bind_data bind_data,
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
     * that file for scanning. This may open file metadata, but should not do
     * per-scan work because projection/filter state has not been prepared yet.
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
     * intermediate scan columns DuckDB needs the chunks to contain. Columns
     * marked `is_projected=false` are only needed for pushed filters and are
     * not referenced by DuckDB's final output expressions. `filters` may be
     * null when no filters were pushed down.
     */
    void (*prepare_reader)(duckdb_vx_mff_reader reader,
                           const duckdb_vx_mff_column *projection,
                           size_t projection_count,
                           duckdb_vx_table_filter_set filters,
                           duckdb_vx_error *error);

    /**
     * Try to initialize a scan over `reader`. Returns true if a scan can begin,
     * false if the reader is exhausted. Called with DuckDB's multi-file global
     * scheduling mutex held; must not block on I/O, run async work, or build
     * expensive scan pipelines. Store only the claimed work descriptor in `local`.
     */
    bool (*try_initialize_scan)(duckdb_vx_mff_reader reader,
                                duckdb_vx_mff_global global,
                                duckdb_vx_mff_local local,
                                duckdb_vx_error *error);

    /**
     * Prepare local scan state for the work claimed by try_initialize_scan.
     * Called outside DuckDB's multi-file global scheduling mutex, mirroring
     * DuckDB's BaseFileReader::PrepareScan hook.
     */
    void (*prepare_scan)(duckdb_vx_mff_reader reader,
                         duckdb_vx_mff_global global,
                         duckdb_vx_mff_local local,
                         duckdb_vx_error *error);

    /**
     * Produce the next batch of data into `chunk_out`. Called outside DuckDB's
     * multi-file global scheduling mutex after prepare_scan. Returns:
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
     * Get bind-time per-column statistics by name. Used when DuckDB asks for
     * scan stats after copying bind data, before a per-file reader exists.
     * Returns false if no stats are available.
     */
    bool (*statistics)(duckdb_vx_mff_bind_data bind_data,
                       const char *col_name,
                       size_t name_len,
                       duckdb_column_statistics *stats_out);

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
     * Get exact row count statistics for one file. Returning false means the
     * stats are not currently available; DuckDB will skip statistics-based
     * aggregate rewrites unless every file returns exact stats.
     */
    bool (*partition_stats)(duckdb_client_context ctx,
                            duckdb_vx_mff_bind_data bind_data,
                            const char *file_path,
                            size_t path_len,
                            duckdb_vx_mff_partition_stats *out,
                            duckdb_vx_error *error);

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
