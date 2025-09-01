// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optimizer extension for DuckDB to rewrite len(column) -> column$length
//!
//! This module provides both the legacy C++ optimizer and the new pure Rust
//! optimizer implementation for better maintainability and customization.

use vortex::error::VortexResult;

use crate::duckdb::Database;
// Re-export types for backwards compatibility
pub use crate::duckdb::expr::{ColumnBinding, Expression, LogicalExpressionType as ExpressionType};
pub use crate::duckdb::logical_operator::{LogicalOperator, LogicalOperatorType};
pub use crate::rust_optimizer::{LengthReplacement, RustLengthOptimizer};


/// Register the Rust-based length optimizer with DuckDB
///
/// This registers the pure Rust implementation of the length optimization
/// that automatically rewrites len(column) function calls to use virtual column references.
pub fn register_rust_optimizer(db: &mut Database) -> VortexResult<()> {
    crate::rust_optimizer::register_rust_optimizer(db)
}

/// Legacy alias for backwards compatibility
pub fn register_optimizer(db: &mut Database) -> VortexResult<()> {
    register_rust_optimizer(db)
}
