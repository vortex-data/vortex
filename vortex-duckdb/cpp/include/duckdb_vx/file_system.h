// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"
#include "duckdb_vx/client_context.h"
#include "duckdb_vx/error.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

typedef struct duckdb_vx_file_handle_ *duckdb_vx_file_handle;

typedef struct {
    const char **entries;
    size_t count;
} duckdb_vx_uri_list;

// Open a file using DuckDB's filesystem (supports httpfs, s3, etc.).
duckdb_vx_file_handle duckdb_vx_fs_open(duckdb_vx_client_context ctx, const char *path,
                                        duckdb_vx_error *error_out);

// Close a previously opened file handle.
void duckdb_vx_fs_close(duckdb_vx_file_handle *handle);

// Get the size of an opened file.
duckdb_state duckdb_vx_fs_get_size(duckdb_vx_file_handle handle, idx_t *size_out,
                                   duckdb_vx_error *error_out);

// Read up to len bytes at the given offset into buffer. Returns bytes read via out_len.
duckdb_state duckdb_vx_fs_read(duckdb_vx_file_handle handle, idx_t offset, idx_t len, uint8_t *buffer,
                               idx_t *out_len, duckdb_vx_error *error_out);

// Expand a glob using DuckDB's filesystem.
duckdb_vx_uri_list duckdb_vx_fs_glob(duckdb_vx_client_context ctx, const char *pattern,
                                     duckdb_vx_error *error_out);

// Free a string list allocated by duckdb_vx_fs_glob.
void duckdb_vx_uri_list_free(duckdb_vx_uri_list *list);

// Create/truncate a file for writing using DuckDB's filesystem.
duckdb_vx_file_handle duckdb_vx_fs_create(duckdb_vx_client_context ctx, const char *path,
                                          duckdb_vx_error *error_out);

// Write len bytes at the given offset from buffer.
duckdb_state duckdb_vx_fs_write(duckdb_vx_file_handle handle, idx_t offset, idx_t len, uint8_t *buffer,
                                idx_t *out_len, duckdb_vx_error *error_out);

// Flush pending writes to storage.
duckdb_state duckdb_vx_fs_sync(duckdb_vx_file_handle handle, duckdb_vx_error *error_out);

#ifdef __cplusplus /* End C ABI */
}
#endif
