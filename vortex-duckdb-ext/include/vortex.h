//
// THIS FILE IS AUTO-GENERATED, DO NOT MAKE EDITS DIRECTLY
//

// (c) Copyright 2025 SpiralDB Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
