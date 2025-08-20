//! Legacy optimizer compatibility module
//!
//! This module provides backwards compatibility for existing code.
//! New code should use `crate::duckdb::logical_plan` for generic functionality
//! and `crate::rust_optimizer` for the length optimization.

// Legacy re-exports - marked as deprecated to guide users to new locations
#[deprecated(note = "Use crate::duckdb::expr for Expression types")]
#[allow(unused_imports)]
pub use crate::duckdb::expr::{Expression, ColumnBinding, LogicalExpressionType as ExpressionType};
#[deprecated(note = "Use crate::duckdb::logical_operator for LogicalOperator types")]
#[allow(unused_imports)]
pub use crate::duckdb::logical_operator::{LogicalOperator, LogicalOperatorType};
#[deprecated(note = "Use crate::duckdb::logical_plan for utilities")]
#[allow(unused_imports)]
pub use crate::duckdb::logical_plan::LogicalPlanUtils;

#[deprecated(note = "Use crate::rust_optimizer::LengthReplacement instead")]
#[allow(unused_imports)]
pub use crate::rust_optimizer::LengthReplacement;