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


#define DUCKDB_STANDARD_VECTOR_SIZE 2048

/**
 * The DuckDB extension ABI initialization function.
 */
void vortex_init(duckdb_database db);

/**
 * The DuckDB extension ABI version function.
 * This function returns the version of the DuckDB library the extension is built against.
 */
const char *vortex_version(void);

/**
 * An additional function we export to expose the version of the extension itself to C++ code.
 */
const char *vortex_extension_version(void);

#ifdef __cplusplus
}
#endif

// clang-format on
