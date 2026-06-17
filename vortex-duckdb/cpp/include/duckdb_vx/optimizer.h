// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once
#include "duckdb.h"

#ifdef __cplusplus
extern "C" {
#endif

duckdb_state duckdb_vx_optimizer_extension_register(duckdb_database ffi_db);

#ifdef __cplusplus
}
#endif
