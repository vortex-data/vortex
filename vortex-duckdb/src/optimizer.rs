// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optimizer extension for DuckDB to rewrite len(column) -> column$length

use vortex::error::VortexResult;

use crate::cpp::duckdb_database;
use crate::duckdb::Database;

unsafe extern "C" {
    /// Register the Vortex optimizer extension that rewrites len(column) -> column$length
    fn duckdb_vx_register_optimizer(db_handle: duckdb_database);
}

/// Register the Vortex optimizer extension with DuckDB
pub fn register_optimizer(db: &mut Database) -> VortexResult<()> {
    unsafe {
        duckdb_vx_register_optimizer(db.as_ptr());
    }
    Ok(())
}
