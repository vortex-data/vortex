// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once
#include "duckdb.h"

// TODO(joe): expose via c api
// typedef enum copy_function_execution_mode_ {
// REGULAR_COPY_TO_FILE = 1,
// PARALLEL_COPY_TO_FILE,
// BATCH_COPY_TO_FILE
// } copy_function_execution_mode;
//
// TODO(joe): expose via c api
// copy_function_execution_mode (*execution_mode)(bool preserve_insertion_order, bool

#ifdef __cplusplus
extern "C" duckdb_state duckdb_vx_register_copy_function(duckdb_database ffi_db);
#else
duckdb_state duckdb_vx_register_copy_function(duckdb_database ffi_db);
#endif
