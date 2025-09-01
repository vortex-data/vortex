// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optimizer extension for DuckDB to rewrite len(column) -> column$length
//!
//! This module provides both the legacy C++ optimizer and the new pure Rust
//! optimizer implementation for better maintainability and customization.

use vortex::error::VortexResult;

use crate::cpp::duckdb_database;
use crate::duckdb::Database;
// Re-export types for backwards compatibility
pub use crate::duckdb::expr::{ColumnBinding, Expression, LogicalExpressionType as ExpressionType};
pub use crate::duckdb::logical_operator::{LogicalOperator, LogicalOperatorType};
pub use crate::rust_optimizer::{LengthReplacement, RustLengthOptimizer};

unsafe extern "C" {
    /// Register the legacy C++ Vortex optimizer extension that rewrites len(column) -> column$length
    fn duckdb_vx_register_optimizer(db_handle: duckdb_database);
}

/// Register the legacy C++ Vortex optimizer extension with DuckDB
///
/// This registers the original C++ implementation of the length optimization
/// that automatically rewrites len(column) function calls to use virtual column references.
///
/// **Note**: Consider using `register_rust_optimizer` for better maintainability.
pub fn register_optimizer(db: &mut Database) -> VortexResult<()> {
    unsafe {
        duckdb_vx_register_optimizer(db.as_ptr());
    }
    Ok(())
}

/// Register the new Rust-based length optimizer with DuckDB
///
/// This registers the pure Rust implementation of the length optimization
/// which provides the same functionality as the C++ version but with better
/// maintainability and easier customization.
///
/// **Recommended**: This is the preferred way to register the optimizer.
pub fn register_rust_optimizer(db: &mut Database) -> VortexResult<()> {
    crate::rust_optimizer::register_rust_optimizer(db)
}
