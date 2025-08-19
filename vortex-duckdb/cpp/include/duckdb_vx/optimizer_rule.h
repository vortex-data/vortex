// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"

#ifdef __cplusplus
extern "C" {
#endif

// Register the Vortex optimizer extension that rewrites len(column) -> column$length
void duckdb_vx_register_optimizer(duckdb_database db_handle);

#ifdef __cplusplus
}
#endif