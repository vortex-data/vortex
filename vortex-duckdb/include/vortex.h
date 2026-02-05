// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//
// THIS FILE IS AUTO-GENERATED, DO NOT MAKE EDITS DIRECTLY
//

// clang-format off

#ifdef __cplusplus
extern "C" {
#endif

#include "duckdb.h"


/**
 * Global symbol visibility in the Vortex extension:
 * - Rust functions use C ABI with "_rust" suffix (e.g., vortex_init_rust)
 * - C++ wrapper functions have the expected name without suffix (e.g., vortex_init)
 * - C++ wrappers are annotated with DUCKDB_EXTENSION_API to ensure global visibility
 * - C++ wrappers call the corresponding Rust functions
 *
 * This ensures DuckDB can find the symbols when loading the extension.
 *
 * The DuckDB extension ABI initialization function.
 */
void vortex_init_rust(duckdb_database db);

/**
 * The DuckDB extension ABI version function.
 * This function returns the version of the DuckDB library the extension is built against.
 */
const char *vortex_version_rust(void);

/**
 * An additional function we export to expose the version of the extension itself to C++ code.
 */
const char *vortex_extension_version_rust(void);

#ifdef __cplusplus
}
#endif

// clang-format on
