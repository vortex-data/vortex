// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DuckDB logical operator manipulation API with downcasting
//!
//! This module provides safe Rust wrappers around DuckDB's logical plan operators,
//! following a similar pattern to the Expression API with downcasting to specific
//! operator types for type-safe manipulation.

use std::ffi::CStr;
use std::fmt::{Display, Formatter};

use vortex::error::{VortexResult, vortex_bail};

use crate::cpp::*;
use crate::duckdb::expr::Expression;
use crate::wrapper;

wrapper!(LogicalOperator, duckdb_vx_logical_operator, |_ptr| {
    // TODO: Free memory
    // LogicalOperator doesn't need destruction as it's owned by DuckDB's plan tree
});

impl Display for LogicalOperator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.to_debug_string() {
            Ok(s) => write!(f, "{}", s),
            Err(_) => write!(f, "<LogicalOperator>"),
        }
    }
}

/// Represents the type of a logical operator in DuckDB's query plan
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LogicalOperatorType {
    Get = DUCKDB_VX_LOGICAL_OPERATOR_TYPE_DUCKDB_VX_LOGICAL_GET,
    Projection = DUCKDB_VX_LOGICAL_OPERATOR_TYPE_DUCKDB_VX_LOGICAL_PROJECTION,
    Filter = DUCKDB_VX_LOGICAL_OPERATOR_TYPE_DUCKDB_VX_LOGICAL_FILTER,
    Join = DUCKDB_VX_LOGICAL_OPERATOR_TYPE_DUCKDB_VX_LOGICAL_JOIN,
    Aggregate = DUCKDB_VX_LOGICAL_OPERATOR_TYPE_DUCKDB_VX_LOGICAL_AGGREGATE,
    Unknown = DUCKDB_VX_LOGICAL_OPERATOR_TYPE_DUCKDB_VX_LOGICAL_UNKNOWN,
}

impl LogicalOperator {
    /// Get the type of this logical operator
    pub fn operator_type(&self) -> LogicalOperatorType {
        let op_type = unsafe { duckdb_vx_get_operator_type(self.as_ptr()) };
        match op_type {
            DUCKDB_VX_LOGICAL_OPERATOR_TYPE_DUCKDB_VX_LOGICAL_GET => LogicalOperatorType::Get,
            DUCKDB_VX_LOGICAL_OPERATOR_TYPE_DUCKDB_VX_LOGICAL_PROJECTION => {
                LogicalOperatorType::Projection
            }
            DUCKDB_VX_LOGICAL_OPERATOR_TYPE_DUCKDB_VX_LOGICAL_FILTER => LogicalOperatorType::Filter,
            DUCKDB_VX_LOGICAL_OPERATOR_TYPE_DUCKDB_VX_LOGICAL_JOIN => LogicalOperatorType::Join,
            DUCKDB_VX_LOGICAL_OPERATOR_TYPE_DUCKDB_VX_LOGICAL_AGGREGATE => {
                LogicalOperatorType::Aggregate
            }
            _ => LogicalOperatorType::Unknown,
        }
    }

    /// Downcast the operator to its specific type with specialized methods
    pub fn as_class(&self) -> Option<LogicalOperatorClass<'_>> {
        match self.operator_type() {
            LogicalOperatorType::Get => Some(LogicalOperatorClass::Get(LogicalGet { op: self })),
            LogicalOperatorType::Projection => {
                Some(LogicalOperatorClass::Projection(LogicalProjection {
                    op: self,
                }))
            }
            LogicalOperatorType::Filter
            | LogicalOperatorType::Join
            | LogicalOperatorType::Aggregate
            | LogicalOperatorType::Unknown => None,
        }
    }

    /// Get string representation of this operator for debugging
    pub fn to_debug_string(&self) -> VortexResult<String> {
        unsafe {
            let str_ptr = duckdb_vx_logical_operator_to_string(self.as_ptr());
            if str_ptr.is_null() {
                vortex_bail!("Failed to convert logical operator to string");
            }
            let result = CStr::from_ptr(str_ptr).to_string_lossy().into_owned();
            duckdb_vx_free_string(str_ptr);
            Ok(result)
        }
    }

    /// Get the number of child operators
    pub fn children_count(&self) -> usize {
        unsafe { duckdb_vx_get_children_count(self.as_ptr()) as usize }
    }

    /// Get a child operator by index
    pub fn get_child(&self, index: usize) -> Option<LogicalOperator> {
        unsafe {
            let child_ptr = duckdb_vx_get_child(self.as_ptr(), index as u64);
            if child_ptr.is_null() {
                None
            } else {
                Some(LogicalOperator::borrow(child_ptr))
            }
        }
    }

    /// Get the number of expressions in this operator
    pub fn expressions_count(&self) -> usize {
        unsafe { duckdb_vx_get_expressions_count(self.as_ptr()) as usize }
    }

    /// Get an expression by index
    pub fn get_expression(&self, index: usize) -> Option<Expression> {
        unsafe {
            let expr_ptr = duckdb_vx_get_expression(self.as_ptr(), index as u64);
            if expr_ptr.is_null() {
                None
            } else {
                Some(Expression::borrow(expr_ptr))
            }
        }
    }

    /// Set an expression by index (transfers ownership)
    pub fn set_expression(&self, index: usize, expression: Expression) {
        unsafe {
            duckdb_vx_set_expression(self.as_ptr(), index as u64, expression.as_ptr());
            // Prevent the expression from being dropped since ownership was transferred
            std::mem::forget(expression);
        }
    }
}

/// Enum representing different logical operator types with specialized methods
pub enum LogicalOperatorClass<'a> {
    Get(LogicalGet<'a>),
    Projection(LogicalProjection<'a>),
}

/// LogicalGet operator (table scan) with table-specific methods
pub struct LogicalGet<'a> {
    op: &'a LogicalOperator,
}

impl<'a> LogicalGet<'a> {
    /// Get the table function name
    pub fn function_name(&self) -> VortexResult<Option<String>> {
        unsafe {
            let name_ptr = duckdb_vx_get_function_name(self.op.as_ptr());
            if name_ptr.is_null() {
                Ok(None)
            } else {
                let name = CStr::from_ptr(name_ptr).to_string_lossy().into_owned();
                duckdb_vx_free_string(name_ptr);
                Ok(Some(name))
            }
        }
    }

    /// Check if this is a vortex_scan table function
    pub fn is_vortex_scan(&self) -> VortexResult<bool> {
        let function_name = self.function_name()?;
        Ok(function_name.as_deref() == Some("vortex_scan"))
    }

    /// Get column names from the table schema
    pub fn column_names(&self) -> VortexResult<Vec<String>> {
        unsafe {
            let mut count = 0u64;
            let names_ptr = duckdb_vx_get_column_names(self.op.as_ptr(), &mut count);

            if names_ptr.is_null() {
                return Ok(Vec::new());
            }

            let mut names = Vec::with_capacity(count as usize);
            for i in 0..count {
                let name_ptr = *names_ptr.add(i as usize);
                if !name_ptr.is_null() {
                    let name = CStr::from_ptr(name_ptr).to_string_lossy().into_owned();
                    names.push(name);
                }
            }

            duckdb_vx_free_string_array(names_ptr, count);
            Ok(names)
        }
    }

    /// Get the current projection IDs
    pub fn get_projection_ids(&self) -> VortexResult<Vec<u64>> {
        unsafe {
            let mut count = 0u64;
            let ids_ptr = duckdb_vx_get_projection_ids(self.op.as_ptr(), &mut count);

            if ids_ptr.is_null() {
                return Ok(Vec::new());
            }

            let mut ids = Vec::with_capacity(count as usize);
            for i in 0..count {
                ids.push(*ids_ptr.add(i as usize));
            }

            duckdb_vx_free_uint64_array(ids_ptr);
            Ok(ids)
        }
    }

    /// Update the projection IDs for this table scan
    pub fn update_projection_ids(&self, new_projection_ids: &[u64]) -> VortexResult<()> {
        unsafe {
            duckdb_vx_update_projection_ids(
                self.op.as_ptr(),
                new_projection_ids.as_ptr() as *mut u64,
                new_projection_ids.len() as u64,
            );
        }
        Ok(())
    }

    /// Add a column ID to the scan
    pub fn add_column_id(&self, column_id: u64) {
        unsafe {
            duckdb_vx_add_column_id(self.op.as_ptr(), column_id);
        }
    }

    /// Clear all column IDs
    pub fn clear_column_ids(&self) {
        unsafe {
            duckdb_vx_clear_column_ids(self.op.as_ptr());
        }
    }

    /// Get column names (wrapper for convenience)
    pub fn get_column_names(&self) -> VortexResult<Vec<String>> {
        self.column_names()
    }
}

/// LogicalProjection operator with projection-specific methods
pub struct LogicalProjection<'a> {
    op: &'a LogicalOperator,
}

impl<'a> LogicalProjection<'a> {
    /// Get the projection expressions
    pub fn projections(&self) -> impl Iterator<Item = Option<Expression>> {
        (0..self.op.expressions_count()).map(move |i| self.op.get_expression(i))
    }

    /// Set a projection expression at the given index
    pub fn set_projection(&self, index: usize, expression: Expression) {
        self.op.set_expression(index, expression);
    }
}
