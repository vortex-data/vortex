// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb_vx/duckdb_diagnostics.h"

DUCKDB_INCLUDES_BEGIN
#include "duckdb.h"
DUCKDB_INCLUDES_END

#include "duckdb_vx/error.h"

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

duckdb_vector duckdb_vx_variant_to_parquet(duckdb_vector variant, idx_t len, duckdb_vx_error *err);

void duckdb_vx_variant_from_parquet(duckdb_vector metadata,
                                    duckdb_vector value,
                                    duckdb_vector typed_value,
                                    bool has_typed_value,
                                    duckdb_vector out,
                                    idx_t len,
                                    duckdb_vx_error *err);

#ifdef __cplusplus /* End C ABI */
}
#endif
