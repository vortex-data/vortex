//! Generic DuckDB logical plan manipulation API
//!
//! This module provides safe Rust wrappers around DuckDB's logical plan structures,
//! allowing for custom optimization rules and plan transformations.

use std::ffi::{CStr, c_void};

use vortex::error::{VortexResult, vortex_bail};

use crate::cpp::*;
use crate::duckdb::expr::{Expression, LogicalExpressionType as ExpressionType};

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



/// Safe wrapper around DuckDB's LogicalOperator
pub struct LogicalOperator {
    ptr: duckdb_vx_logical_operator,
}

impl LogicalOperator {
    /// Create a LogicalOperator from a raw pointer (internal use only)
    pub(crate) unsafe fn from_ptr(ptr: duckdb_vx_logical_operator) -> Option<Self> {
        if ptr.is_null() {
            None
        } else {
            Some(LogicalOperator { ptr })
        }
    }

    /// Get the type of this logical operator
    pub fn operator_type(&self) -> LogicalOperatorType {
        let op_type = unsafe { duckdb_vx_get_operator_type(self.ptr) };
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

    /// Get the number of child operators
    pub fn children_count(&self) -> usize {
        unsafe { duckdb_vx_get_children_count(self.ptr) as usize }
    }

    /// Get a child operator by index
    pub fn get_child(&self, index: usize) -> Option<LogicalOperator> {
        unsafe {
            let child_ptr = duckdb_vx_get_child(self.ptr, index as u64);
            LogicalOperator::from_ptr(child_ptr)
        }
    }

    /// Get the number of expressions in this operator
    pub fn expressions_count(&self) -> usize {
        unsafe { duckdb_vx_get_expressions_count(self.ptr) as usize }
    }

    /// Get an expression by index
    pub fn get_expression(&self, index: usize) -> Option<Expression> {
        unsafe {
            let expr_ptr = duckdb_vx_get_expression(self.ptr, index as u64);
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
            duckdb_vx_set_expression(self.ptr, index as u64, expression.as_ptr());
            // Prevent the expression from being dropped since ownership was transferred
            std::mem::forget(expression);
        }
    }

    /// Get the function name if this is a LogicalGet operator
    pub fn function_name(&self) -> VortexResult<Option<String>> {
        unsafe {
            let name_ptr = duckdb_vx_get_function_name(self.ptr);
            if name_ptr.is_null() {
                Ok(None)
            } else {
                let name = CStr::from_ptr(name_ptr).to_string_lossy().into_owned();
                duckdb_vx_free_string(name_ptr);
                Ok(Some(name))
            }
        }
    }

    /// Get column names if this is a LogicalGet operator
    pub fn column_names(&self) -> VortexResult<Vec<String>> {
        unsafe {
            let mut count = 0u64;
            let names_ptr = duckdb_vx_get_column_names(self.ptr, &mut count);

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

    /// Get projection IDs if this is a LogicalGet operator
    pub fn projection_ids(&self) -> VortexResult<Vec<u64>> {
        unsafe {
            let mut count = 0u64;
            let ids_ptr = duckdb_vx_get_projection_ids(self.ptr, &mut count);

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

    /// Update projection IDs if this is a LogicalGet operator
    pub fn update_projection_ids(&self, new_ids: &[u64]) -> VortexResult<()> {
        unsafe {
            duckdb_vx_update_projection_ids(
                self.ptr,
                new_ids.as_ptr() as *mut u64,
                new_ids.len() as u64,
            );
        }
        Ok(())
    }

    /// Add a column ID if this is a LogicalGet operator
    pub fn add_column_id(&self, column_id: u64) {
        unsafe {
            duckdb_vx_add_column_id(self.ptr, column_id);
        }
    }

    /// Clear all column IDs if this is a LogicalGet operator
    pub fn clear_column_ids(&self) {
        unsafe {
            duckdb_vx_clear_column_ids(self.ptr);
        }
    }

    /// Get projection IDs (convenience method)
    pub fn get_projection_ids(&self) -> VortexResult<Vec<u64>> {
        self.projection_ids()
    }

    /// Get column names (convenience method)
    pub fn get_column_names(&self) -> VortexResult<Vec<String>> {
        self.column_names()
    }

    /// Get string representation of this logical operator
    pub fn to_string(&self) -> VortexResult<String> {
        unsafe {
            let str_ptr = duckdb_vx_logical_operator_to_string(self.ptr);
            if str_ptr.is_null() {
                vortex_bail!("Failed to convert logical operator to string");
            }
            let result = CStr::from_ptr(str_ptr).to_string_lossy().into_owned();
            unsafe extern "C" {
                fn free(ptr: *mut c_void);
            }
            free(str_ptr as *mut c_void);
            Ok(result)
        }
    }

    /// Get the raw pointer (for internal use)
    #[allow(dead_code)]
    pub(crate) fn as_ptr(&self) -> duckdb_vx_logical_operator {
        self.ptr
    }
}


/// Utility functions for visiting and manipulating logical plans
pub struct LogicalPlanUtils;

impl LogicalPlanUtils {
    /// Visit all operators in a plan tree with a custom visitor function
    pub fn visit_operators<F>(plan: &LogicalOperator, visitor: &mut F) -> VortexResult<()>
    where
        F: FnMut(&LogicalOperator) -> VortexResult<()>,
    {
        // Visit this operator
        visitor(plan)?;

        // Recursively visit children
        for i in 0..plan.children_count() {
            if let Some(child) = plan.get_child(i) {
                Self::visit_operators(&child, visitor)?;
            }
        }

        Ok(())
    }

    /// Find all operators of a specific type in the plan tree
    pub fn find_operators_by_type(
        plan: &LogicalOperator,
        target_type: LogicalOperatorType,
    ) -> VortexResult<Vec<*const LogicalOperator>> {
        let mut operators = Vec::new();
        Self::visit_operators(plan, &mut |op| {
            if op.operator_type() == target_type {
                operators.push(op as *const LogicalOperator);
            }
            Ok(())
        })?;
        Ok(operators)
    }

    /// Check if the plan contains any operator of the specified type
    pub fn contains_operator_type(
        plan: &LogicalOperator,
        target_type: LogicalOperatorType,
    ) -> VortexResult<bool> {
        let mut found = false;
        Self::visit_operators(plan, &mut |op| {
            if op.operator_type() == target_type {
                found = true;
            }
            Ok(())
        })?;
        Ok(found)
    }

    /// Find all expressions of a specific type in the plan tree
    pub fn find_expressions_by_type(
        plan: &LogicalOperator,
        target_type: ExpressionType,
    ) -> VortexResult<Vec<*const Expression>> {
        let mut expressions = Vec::new();
        Self::visit_operators(plan, &mut |op| {
            for i in 0..op.expressions_count() {
                if let Some(expr) = op.get_expression(i) {
                    if expr.logical_expression_type() == target_type {
                        expressions.push(&expr as *const Expression);
                    }
                }
            }
            Ok(())
        })?;
        Ok(expressions)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn test_column_binding_conversion() {
        let rust_binding = ColumnBinding {
            table_index: 1,
            column_index: 2,
        };

        let c_binding: duckdb_vx_column_binding = rust_binding.into();
        assert_eq!(c_binding.table_index, 1);
        assert_eq!(c_binding.column_index, 2);

        let back_to_rust: ColumnBinding = c_binding.into();
        assert_eq!(back_to_rust.table_index, 1);
        assert_eq!(back_to_rust.column_index, 2);
    }

    #[test]
    fn test_operator_type_enum() {
        assert_eq!(
            LogicalOperatorType::Get as u32,
            DUCKDB_VX_LOGICAL_OPERATOR_TYPE_DUCKDB_VX_LOGICAL_GET
        );
        assert_eq!(
            LogicalOperatorType::Projection as u32,
            DUCKDB_VX_LOGICAL_OPERATOR_TYPE_DUCKDB_VX_LOGICAL_PROJECTION
        );
    }

    #[test]
    fn test_expression_type_enum() {
        assert_eq!(
            ExpressionType::BoundColumnRef as u32,
            DUCKDB_VX_EXPRESSION_TYPE_DUCKDB_VX_BOUND_COLUMN_REF
        );
        assert_eq!(
            ExpressionType::BoundFunction as u32,
            DUCKDB_VX_EXPRESSION_TYPE_DUCKDB_VX_BOUND_FUNCTION
        );
    }

    // Mock logical plan builder for testing
    struct MockPlanBuilder {
        operators: Vec<MockOperator>,
        expressions: Vec<MockExpression>,
    }

    struct MockOperator {
        op_type: LogicalOperatorType,
        children: Vec<usize>,    // indices into operators vec
        expressions: Vec<usize>, // indices into expressions vec
        function_name: Option<String>,
        column_names: Vec<String>,
        projection_ids: Vec<u64>,
    }

    struct MockExpression {
        expr_type: ExpressionType,
        function_name: Option<String>,
        function_args: Vec<usize>, // indices into expressions vec
        column_alias: Option<String>,
        column_binding: ColumnBinding,
    }

    impl MockPlanBuilder {
        fn new() -> Self {
            Self {
                operators: Vec::new(),
                expressions: Vec::new(),
            }
        }

        fn add_get_operator(&mut self, function_name: &str, columns: Vec<&str>) -> usize {
            let op = MockOperator {
                op_type: LogicalOperatorType::Get,
                children: Vec::new(),
                expressions: Vec::new(),
                function_name: Some(function_name.to_string()),
                column_names: columns.iter().map(|s| s.to_string()).collect(),
                projection_ids: (0..columns.len() as u64).collect(),
            };
            self.operators.push(op);
            self.operators.len() - 1
        }

        fn add_projection_operator(&mut self, child: usize, expressions: Vec<usize>) -> usize {
            let op = MockOperator {
                op_type: LogicalOperatorType::Projection,
                children: vec![child],
                expressions,
                function_name: None,
                column_names: Vec::new(),
                projection_ids: Vec::new(),
            };
            self.operators.push(op);
            self.operators.len() - 1
        }

        fn add_column_ref(&mut self, alias: &str, table_idx: u64, col_idx: u64) -> usize {
            let expr = MockExpression {
                expr_type: ExpressionType::BoundColumnRef,
                function_name: None,
                function_args: Vec::new(),
                column_alias: Some(alias.to_string()),
                column_binding: ColumnBinding {
                    table_index: table_idx,
                    column_index: col_idx,
                },
            };
            self.expressions.push(expr);
            self.expressions.len() - 1
        }

        fn add_function(&mut self, name: &str, args: Vec<usize>) -> usize {
            let expr = MockExpression {
                expr_type: ExpressionType::BoundFunction,
                function_name: Some(name.to_string()),
                function_args: args,
                column_alias: None,
                column_binding: ColumnBinding {
                    table_index: 0,
                    column_index: 0,
                },
            };
            self.expressions.push(expr);
            self.expressions.len() - 1
        }
    }

    #[test]
    fn test_mock_plan_builder() {
        let mut builder = MockPlanBuilder::new();

        // Create a simple plan: Projection(len(title)) -> Get(vortex_scan)
        let get_op = builder.add_get_operator(
            "vortex_scan",
            vec!["title", "description", "title$length", "description$length"],
        );

        let col_ref = builder.add_column_ref("title", 0, 0);
        let len_func = builder.add_function("len", vec![col_ref]);
        let proj_op = builder.add_projection_operator(get_op, vec![len_func]);

        // Verify the structure
        assert_eq!(builder.operators.len(), 2);
        assert_eq!(builder.expressions.len(), 2);
        assert_eq!(
            builder.operators[proj_op].op_type,
            LogicalOperatorType::Projection
        );
        assert_eq!(builder.operators[get_op].op_type, LogicalOperatorType::Get);
        assert_eq!(
            builder.expressions[len_func].function_name,
            Some("len".to_string())
        );
    }

    #[test]
    fn test_visitor_pattern_simulation() {
        // Simulate visiting operators and collecting information
        let mut builder = MockPlanBuilder::new();

        // Build a plan with multiple operators
        let get1 = builder.add_get_operator("vortex_scan", vec!["col1", "col2"]);
        let get2 = builder.add_get_operator("regular_scan", vec!["col3", "col4"]);

        let col_ref1 = builder.add_column_ref("col1", 0, 0);
        let len_func1 = builder.add_function("length", vec![col_ref1]);
        let _proj1 = builder.add_projection_operator(get1, vec![len_func1]);

        let col_ref2 = builder.add_column_ref("col3", 1, 0);
        let _proj2 = builder.add_projection_operator(get2, vec![col_ref2]);

        // Simulate visitor pattern to find vortex_scan operators
        let mut vortex_scans = Vec::new();
        for (idx, op) in builder.operators.iter().enumerate() {
            if op.op_type == LogicalOperatorType::Get {
                if let Some(ref name) = op.function_name {
                    if name == "vortex_scan" {
                        vortex_scans.push(idx);
                    }
                }
            }
        }

        assert_eq!(vortex_scans.len(), 1);
        assert_eq!(vortex_scans[0], get1);

        // Simulate finding length functions
        let mut length_functions = Vec::new();
        for (idx, expr) in builder.expressions.iter().enumerate() {
            if expr.expr_type == ExpressionType::BoundFunction {
                if let Some(ref name) = expr.function_name {
                    if name == "length" || name == "len" {
                        length_functions.push(idx);
                    }
                }
            }
        }

        assert_eq!(length_functions.len(), 1);
        assert_eq!(length_functions[0], len_func1);
    }

    #[test]
    fn test_expression_replacement_simulation() {
        let mut builder = MockPlanBuilder::new();

        // Create expressions: len(title) that we want to replace
        let title_ref = builder.add_column_ref("title", 0, 0);
        let len_func = builder.add_function("len", vec![title_ref]);

        // Simulate replacement with virtual column reference
        let virtual_col_ref = builder.add_column_ref("title$length", 0, 2);

        // Track the replacement
        let mut replacements = HashMap::new();
        replacements.insert(len_func, virtual_col_ref);

        // Verify we can look up the replacement
        assert_eq!(replacements.get(&len_func), Some(&virtual_col_ref));

        // Verify the new expression has correct properties
        let new_expr = &builder.expressions[virtual_col_ref];
        assert_eq!(new_expr.column_alias, Some("title$length".to_string()));
        assert_eq!(new_expr.column_binding.column_index, 2);
    }

    #[test]
    fn test_projection_update_simulation() {
        let mut builder = MockPlanBuilder::new();

        // Create a vortex_scan with projection IDs
        let get_op =
            builder.add_get_operator("vortex_scan", vec!["title", "description", "title$length"]);

        // Original projection: [0, 1] (title, description)
        builder.operators[get_op].projection_ids = vec![0, 1];

        // Simulate adding virtual column to projections
        builder.operators[get_op].projection_ids.push(2);

        // Verify the update
        assert_eq!(builder.operators[get_op].projection_ids, vec![0, 1, 2]);

        // Simulate replacing a projection with virtual column
        builder.operators[get_op].projection_ids[0] = 2; // Replace title with title$length

        assert_eq!(builder.operators[get_op].projection_ids, vec![2, 1, 2]);
    }
}
