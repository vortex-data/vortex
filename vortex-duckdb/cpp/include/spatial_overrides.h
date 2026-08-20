// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

/// Shadow the spatial functions that block filter pushdown with pushable copies; see
/// `SPATIAL_OVERRIDES` in spatial_overrides.cpp for the list. Call after `LOAD spatial`;
/// does nothing when spatial is not loaded.
duckdb_state duckdb_vx_register_spatial_overrides(duckdb_database ffi_db);

#ifdef __cplusplus
}
#endif
