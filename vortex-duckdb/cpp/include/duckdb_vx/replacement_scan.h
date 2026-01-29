// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#ifdef __cplusplus /* If compiled as C++, use C ABI */
extern "C" {
#endif

duckdb_state duckdb_vx_register_scan_replacement(duckdb_database duckdb_database);

#ifdef __cplusplus /* End C ABI */
}
#endif
