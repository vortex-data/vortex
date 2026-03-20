// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Suppress warnings from DuckDB headers that we include but do not own.
//
// Usage:
//   DUCKDB_INCLUDES_BEGIN
//   #include "duckdb.h"
//   #include "duckdb/main/client_context.hpp"
//   DUCKDB_INCLUDES_END

#pragma once

// clang-format off
#if defined(__clang__) || defined(__GNUC__)
#define DUCKDB_INCLUDES_BEGIN                                                  \
    _Pragma("GCC diagnostic push")                                             \
    _Pragma("GCC diagnostic ignored \"-Wall\"")                                \
    _Pragma("GCC diagnostic ignored \"-Wextra\"")                              \
    _Pragma("GCC diagnostic ignored \"-Wpedantic\"")                           \
    _Pragma("GCC diagnostic ignored \"-Wunused-parameter\"")                   \
    _Pragma("GCC diagnostic ignored \"-Wtype-limits\"")
#define DUCKDB_INCLUDES_END _Pragma("GCC diagnostic pop")
#else
#define DUCKDB_INCLUDES_BEGIN
#define DUCKDB_INCLUDES_END
#endif
// clang-format on
