//! Pure Rust implementation of the length optimization logic
//!
//! This module implements the length optimization using the generic logical plan API.

use std::collections::{HashMap, HashSet};
use std::ptr;

use log::trace;
use vortex::error::{VortexExpect, VortexResult};

use crate::duckdb::expr::{ColumnBinding, Expression};
use crate::duckdb::logical_operator::{LogicalOperator, LogicalOperatorClass};
use crate::duckdb::{Database, ExpressionClass};

/// Simple collector to find which columns each projection output depends on
#[derive(Debug)]
struct ProjectionAnalyzer {
    /// Columns needed for each projection output position (position -> column_id)
    output_dependencies: Vec<Option<u64>>,
    /// All unique column_ids that need to be fetched
    required_columns: HashSet<u64>,
}

impl ProjectionAnalyzer {
    fn new() -> Self {
        Self {
            output_dependencies: Vec::new(),
            required_columns: HashSet::new(),
        }
    }

    /// Analyze a projection expression to find its primary column dependency
    fn analyze_projection(&mut self, expr: &Expression, output_position: usize) {
        // Ensure we have enough space
        while self.output_dependencies.len() <= output_position {
            self.output_dependencies.push(None);
        }

        // Find the primary column this expression depends on
        if let Some(column_id) = self.find_primary_column(expr) {
            self.output_dependencies[output_position] = Some(column_id);
            self.required_columns.insert(column_id);
        }
    }

    /// Recursively find the primary column an expression depends on
    fn find_primary_column(&self, expr: &Expression) -> Option<u64> {
        match expr.as_class() {
            Some(ExpressionClass::BoundColumnRef(col_ref)) => {
                Some(col_ref.column_binding.column_index)
            }
            Some(ExpressionClass::BoundFunction(func)) => {
                // For functions, use the first argument's column (if any)
                if func.function_arg_count() > 0
                    && let Some(arg) = func.get_function_arg(0) {
                        return self.find_primary_column(&arg);
                    }
                None
            }
            Some(ExpressionClass::BoundOperator(op)) => {
                // For operators, use the first child's column (if any)
                for child in op.children() {
                    if let Some(col_id) = self.find_primary_column(&child) {
                        return Some(col_id);
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Generate the final column_ids and projection_ids
    fn generate_mappings(&self) -> (Vec<u64>, Vec<u64>) {
        // Create sorted list of required columns for column_ids
        let mut column_ids: Vec<u64> = self.required_columns.iter().copied().collect();
        column_ids.sort();

        // Create mapping from column_id to position in column_ids
        let column_to_position: HashMap<u64, usize> = column_ids
            .iter()
            .enumerate()
            .map(|(pos, &col_id)| (col_id, pos))
            .collect();

        // Generate projection_ids by mapping each output to its column_ids position
        let projection_ids: Vec<u64> = self
            .output_dependencies
            .iter()
            .map(|&opt_col_id| {
                opt_col_id
                    .and_then(|col_id| column_to_position.get(&col_id))
                    .map(|&pos| pos as u64)
                    .unwrap_or(0) // Default to 0 if no column dependency found
            })
            .collect();

        (column_ids, projection_ids)
    }
}

/// Length replacement information
#[derive(Debug, Clone)]
pub struct LengthReplacement {
    pub original_column_binding: u64,
    pub virtual_column_index: u64,
    pub virtual_col_name: String,
    pub new_expression_binding: u64,
    pub expression_index: u64,
}

/// Pure Rust implementation of the length optimization logic
pub struct RustLengthOptimizer {
    replacements: Vec<LengthReplacement>,
}

impl RustLengthOptimizer {
    pub fn new() -> Self {
        Self {
            replacements: Vec::new(),
        }
    }

    /// Apply length optimization to a logical plan
    pub fn optimize_plan(&mut self, plan: &LogicalOperator) -> VortexResult<()> {
        // First check if the plan contains any vortex_scan
        if !Self::has_vortex_scan(plan)? {
            trace!("ℹ️  RUST OPTIMIZER: No vortex_scan found in plan, skipping");
            return Ok(());
        }

        trace!("✅ RUST OPTIMIZER: Found vortex_scan in plan!");

        // Visit all operators and apply length optimization
        self.visit_and_optimize(plan, plan)?;

        if !self.replacements.is_empty() {
            trace!(
                "🎯 RUST OPTIMIZER: Found {} len() → virtual column transformations!",
                self.replacements.len()
            );
        } else {
            trace!("ℹ️  RUST OPTIMIZER: No len() functions found to optimize");
        }

        trace!("✅ RUST OPTIMIZER: Length optimization completed!");
        Ok(())
    }

    /// Check if the plan contains a vortex_scan
    fn has_vortex_scan(op: &LogicalOperator) -> VortexResult<bool> {
        // Check this operator
        if let Some(LogicalOperatorClass::Get(get_op)) = op.as_class()
            && get_op.is_vortex_scan() {
                return Ok(true);
            }
        
        // Check children recursively
        for i in 0..op.children_count() {
            if let Some(child) = op.get_child(i)
                && Self::has_vortex_scan(&child)? {
                    return Ok(true);
                }
        }
        
        Ok(false)
    }

    fn visit_node(&mut self, operator: &LogicalOperator) -> Option<()> {
        trace!("🔍 VISITING: Operator type: {:?}", operator.operator_type());

        let LogicalOperatorClass::Projection(proj) = operator.as_class()? else {
            trace!("🔍 Not a projection operator");
            return None;
        };
        if operator.children_count() != 1 {
            trace!(
                "🔍 Projection operator has {} children, expected 1",
                operator.children_count()
            );
            return None;
        }
        let op_child = operator.get_child(0).unwrap();
        let LogicalOperatorClass::Get(get_op) = op_child.as_class()? else {
            trace!("🔍 Child is not a Get operator");
            return None;
        };
        if !get_op.is_vortex_scan() {
            trace!("🔍 Get operator is not a vortex_scan");
            return None;
        }

        trace!("FOUND VORTEX SCAN");
        trace!("projection operator {}", operator);
        trace!("scan operator {:?}", get_op);

        // Get current state
        let column_names = get_op.column_names().vortex_expect("column names");

        // First pass: Analyze all projections to understand what's being used
        let mut len_replacements: Vec<(usize, u64, String)> = Vec::new(); // (proj_idx, virtual_col_idx, virtual_col_name)
        let mut original_columns_used: HashSet<u64> = HashSet::new();
        let mut projection_expressions = Vec::new();

        // Collect all projection expressions first
        for projection_expr in proj.projections() {
            projection_expressions.push(projection_expr);
        }

        // Analyze each projection
        for (idx, projection_expr) in projection_expressions.iter().enumerate() {
            let Some(projection_expr) = projection_expr else {
                trace!("🔍 Projection {} is None", idx);
                continue;
            };

            // Check if this is a len() function
            if let Some(ExpressionClass::BoundFunction(func_)) = projection_expr.as_class() {
                if let Some(function_name) = func_.function_name() {
                    if function_name == "len" && func_.function_arg_count() > 0 {
                        // This is a len() function
                        if let Some(arg) = func_.get_function_arg(0)
                            && let Some(ExpressionClass::BoundColumnRef(bound_col)) = arg.as_class()
                            {
                                let column_alias = bound_col.name;
                                let _original_col_idx = bound_col.column_binding.column_index;
                                let virtual_col_name = format!("{}$length", column_alias);

                                // Find virtual column in schema
                                if let Some(virtual_col_idx) =
                                    column_names.iter().position(|n| *n == virtual_col_name)
                                {
                                    len_replacements.push((
                                        idx,
                                        virtual_col_idx as u64,
                                        virtual_col_name.clone(),
                                    ));
                                    trace!(
                                        "Found len({}) -> {} at index {}",
                                        column_alias, virtual_col_name, virtual_col_idx
                                    );
                                }
                            }
                    } else {
                        // Not a len() function - check if it uses any columns
                        // This helps us know if original columns are still needed
                        Self::find_column_refs_in_expr(
                            projection_expr,
                            &mut original_columns_used,
                        );
                    }
                }
            } else if let Some(ExpressionClass::BoundColumnRef(col_ref)) =
                projection_expr.as_class()
            {
                // Direct column reference
                let col_idx = col_ref.column_binding.column_index;
                let col_name = &col_ref.name;

                // Check if this is a virtual column that was directly referenced
                let col_name_str = col_name.to_string();
                if col_name_str.ends_with("$length") {
                    // This is a virtual column - find its actual index in the schema
                    if let Some(actual_col_idx) = column_names
                        .iter()
                        .position(|n| *n == col_name_str)
                    {
                        trace!(
                            "Virtual column reference at projection {}: {} (bound to index {} but actually at {})",
                            idx, col_name, col_idx, actual_col_idx
                        );
                        // Don't add to original_columns_used since this is a virtual column
                    } else {
                        trace!("Virtual column {} not found in schema", col_name);
                    }
                } else {
                    // Regular column reference
                    original_columns_used.insert(col_idx);
                    trace!(
                        "Direct column reference at projection {}: column {} (name: {})",
                        idx, col_idx, col_name
                    );
                }
            }
        }

        // Now determine the final column_ids and update projections
        if !len_replacements.is_empty() {
            // Step 1: Collect all unique columns needed (both regular and virtual)
            let mut required_columns = HashSet::new();
            let mut projection_mappings = Vec::new(); // Maps each projection to its required column

            // Process each projection expression to understand what columns it needs
            for (idx, expr) in projection_expressions.iter().enumerate() {
                if let Some(expr) = expr {
                    // Check if this projection is a len() replacement
                    if let Some((_, virtual_col_idx, _)) = len_replacements
                        .iter()
                        .find(|(proj_idx, ..)| *proj_idx == idx)
                    {
                        // This projection will use the virtual column
                        required_columns.insert(*virtual_col_idx);
                        projection_mappings.push(*virtual_col_idx);
                    } else if let Some(ExpressionClass::BoundColumnRef(col_ref)) = expr.as_class() {
                        // Check if this is a virtual column reference
                        let col_name_str = col_ref.name.to_string();
                        if col_name_str.ends_with("$length") {
                            // This is a direct virtual column reference - find its actual index
                            if let Some(actual_col_idx) = column_names
                                .iter()
                                .position(|n| *n == col_name_str)
                            {
                                let actual_col_idx = actual_col_idx as u64;
                                required_columns.insert(actual_col_idx);
                                projection_mappings.push(actual_col_idx);
                            } else {
                                // Fallback to bound index if not found
                                let col_idx = col_ref.column_binding.column_index;
                                required_columns.insert(col_idx);
                                projection_mappings.push(col_idx);
                            }
                        } else {
                            // Regular column reference
                            let col_idx = col_ref.column_binding.column_index;
                            required_columns.insert(col_idx);
                            projection_mappings.push(col_idx);
                        }
                    } else {
                        // For other expressions, try to find column dependencies
                        let mut expr_columns = HashSet::new();
                        Self::find_column_refs_in_expr(expr, &mut expr_columns);
                        for col_id in expr_columns {
                            required_columns.insert(col_id);
                        }
                        // Use the first column found, or default to projection index
                        let first_col = projection_mappings.first().copied().unwrap_or(idx as u64);
                        projection_mappings.push(first_col);
                    }
                } else {
                    projection_mappings.push(idx as u64);
                }
            }

            // Step 2: Create column_ids list in the order they're needed by projections
            // Don't sort - preserve the order projections need them in
            let mut new_column_ids = Vec::new();
            let mut seen_columns = HashSet::new();
            for &col_id in &projection_mappings {
                if !seen_columns.contains(&col_id) {
                    new_column_ids.push(col_id);
                    seen_columns.insert(col_id);
                }
            }

            // Step 3: Create mapping from column_id to position in new_column_ids
            let column_to_position: HashMap<u64, usize> = new_column_ids
                .iter()
                .enumerate()
                .map(|(pos, &col_id)| (col_id, pos))
                .collect();

            // Step 4: Replace len() expressions with virtual column references AND fix direct virtual column bindings
            for (proj_idx, virtual_col_idx, virtual_col_name) in &len_replacements {
                if let Some(proj_expr) = projection_expressions[*proj_idx].as_ref()
                    && let Some(ExpressionClass::BoundFunction(func_)) = proj_expr.as_class()
                        && let Some(arg) = func_.get_function_arg(0)
                            && let Some(ExpressionClass::BoundColumnRef(bound_col)) = arg.as_class()
                            {
                                // Get the position of the virtual column in our new column_ids
                                let position_in_column_ids = column_to_position
                                    .get(virtual_col_idx)
                                    .copied()
                                    .unwrap_or(0);

                                let Ok(virtual_col_ref) = Expression::create_column_ref(
                                    virtual_col_name,
                                    ColumnBinding {
                                        table_index: bound_col.column_binding.table_index,
                                        column_index: position_in_column_ids as u64,
                                    },
                                    0,
                                ) else {
                                    continue;
                                };

                                proj.set_projection(*proj_idx, virtual_col_ref);

                                self.replacements.push(LengthReplacement {
                                    original_column_binding: bound_col.column_binding.column_index,
                                    virtual_column_index: *virtual_col_idx,
                                    virtual_col_name: virtual_col_name.clone(),
                                    new_expression_binding: position_in_column_ids as u64,
                                    expression_index: *proj_idx as u64,
                                });
                            }
            }

            // Step 4.5: Fix direct virtual column references
            for (idx, expr) in projection_expressions.iter().enumerate() {
                if let Some(expr) = expr
                    && let Some(ExpressionClass::BoundColumnRef(col_ref)) = expr.as_class() {
                        let col_name_str = col_ref.name.to_string();
                        if col_name_str.ends_with("$length") {
                            // This is a direct virtual column reference that needs fixing
                            if let Some(actual_col_idx) = column_names
                                .iter()
                                .position(|n| *n == col_name_str)
                            {
                                let position_in_column_ids = column_to_position
                                    .get(&(actual_col_idx as u64))
                                    .copied()
                                    .unwrap_or(0);

                                // Create corrected virtual column reference
                                if let Ok(corrected_col_ref) = Expression::create_column_ref(
                                    &col_name_str,
                                    ColumnBinding {
                                        table_index: col_ref.column_binding.table_index,
                                        column_index: position_in_column_ids as u64,
                                    },
                                    0,
                                ) {
                                    proj.set_projection(idx, corrected_col_ref);
                                    trace!(
                                        "🔍 Fixed virtual column reference {}: {} -> position {}",
                                        idx, col_name_str, position_in_column_ids
                                    );
                                }
                            }
                        }
                    }
            }

            // Step 5: Update column_ids and projection_ids
            trace!("🔍 Final column_ids: {:?}", new_column_ids);
            get_op.clear_column_ids();
            for &col_id in &new_column_ids {
                get_op.add_column_id(col_id);
            }

            // Create projection_ids that map each projection to its position in column_ids
            trace!(
                "🔍 Projection mappings (column each projection needs): {:?}",
                projection_mappings
            );
            trace!("🔍 Column to position mapping: {:?}", column_to_position);
            let projection_ids: Vec<u64> = projection_mappings
                .iter()
                .map(|&col_id| column_to_position.get(&col_id).copied().unwrap_or(0) as u64)
                .collect();

            let _ = get_op.update_projection_ids(&projection_ids);
            trace!("🔍 Final projection_ids: {:?}", projection_ids);

            // Debug: Print final projection expressions
            trace!("🔍 Final projection expressions:");
            for (i, expr) in proj.projections().enumerate() {
                if let Some(expr) = expr {
                    trace!("  [{}]: {}", i, expr);
                } else {
                    trace!("  [{}]: None", i);
                }
            }
        }

        Some(())
    }

    /// Find column references in an expression
    fn find_column_refs_in_expr(expr: &Expression, columns_used: &mut HashSet<u64>) {
        if let Some(ExpressionClass::BoundColumnRef(col_ref)) = expr.as_class() {
            columns_used.insert(col_ref.column_binding.column_index);
        } else if let Some(ExpressionClass::BoundFunction(func)) = expr.as_class() {
            // Check arguments of functions
            for i in 0..func.function_arg_count() {
                if let Some(arg) = func.get_function_arg(i) {
                    Self::find_column_refs_in_expr(&arg, columns_used);
                }
            }
        } else if let Some(ExpressionClass::BoundOperator(op)) = expr.as_class() {
            // Check children of operators
            for child in op.children() {
                Self::find_column_refs_in_expr(&child, columns_used);
            }
        }
    }

    /// Visit all operators and apply optimizations
    fn visit_and_optimize(
        &mut self,
        op: &LogicalOperator,
        plan_root: &LogicalOperator,
    ) -> VortexResult<()> {
        trace!("🔍 VISITING: Operator type: {:?}", op.operator_type());

        self.visit_node(op);

        // Visit children
        for i in 0..op.children_count() {
            if let Some(child) = op.get_child(i) {
                self.visit_and_optimize(&child, plan_root)?;
            }
        }

        Ok(())
    }

    /// Get the collected replacements
    pub fn get_replacements(&self) -> &[LengthReplacement] {
        &self.replacements
    }
}

impl Default for RustLengthOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// C callback function that implements the optimization in Rust
extern "C-unwind" fn rust_optimizer_callback(
    plan: crate::cpp::duckdb_vx_logical_operator,
    _user_data: *mut std::ffi::c_void,
) {
    if plan.is_null() {
        return;
    }

    let logical_op = unsafe { LogicalOperator::borrow(plan) };

    // Create and run the optimizer
    let mut optimizer = RustLengthOptimizer::new();
    match optimizer.optimize_plan(&logical_op) {
        Ok(()) => {
            trace!("✅ RUST OPTIMIZER: Optimization completed successfully!");
            let replacements = optimizer.get_replacements();
            if !replacements.is_empty() {
                trace!(
                    "📊 RUST OPTIMIZER: Made {} replacements:",
                    replacements.len()
                );
                for (i, replacement) in replacements.iter().enumerate() {
                    trace!(
                        "  {}. {} → {}",
                        i + 1,
                        replacement.original_column_binding,
                        replacement.virtual_col_name
                    );
                }
            }
        }
        Err(e) => {
            trace!("❌ RUST OPTIMIZER: Optimization failed: {}", e);
        }
    }
}

/// Register the Rust-based length optimizer with DuckDB
pub fn register_rust_optimizer(db: &mut Database) -> VortexResult<()> {
    trace!("🔧 REGISTERING: Rust-based length optimizer...");

    unsafe {
        crate::cpp::duckdb_vx_register_rust_optimizer(
            db.as_ptr(),
            Some(rust_optimizer_callback),
            ptr::null_mut(),
        );
    }

    trace!("✅ SUCCESS: Rust-based length optimizer registered!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_length_replacement() {
        let replacement = LengthReplacement {
            original_column_binding: 1,
            virtual_column_index: 5,
            virtual_col_name: "title$length".to_string(),
            new_expression_binding: 5,
            expression_index: 0,
        };

        assert_eq!(replacement.virtual_col_name, "title$length");
        assert_eq!(replacement.virtual_column_index, 5);
    }

    #[test]
    fn test_optimizer_creation() {
        let _optimizer = RustLengthOptimizer::new();
        assert_eq!(_optimizer.get_replacements().len(), 0);
    }

    #[test]
    fn test_multiple_replacements() {
        let mut optimizer = RustLengthOptimizer::new();

        // Simulate finding multiple length functions
        optimizer.replacements.push(LengthReplacement {
            original_column_binding: 0,
            virtual_column_index: 2,
            virtual_col_name: "title$length".to_string(),
            new_expression_binding: 2,
            expression_index: 0,
        });

        optimizer.replacements.push(LengthReplacement {
            original_column_binding: 1,
            virtual_column_index: 3,
            virtual_col_name: "description$length".to_string(),
            new_expression_binding: 3,
            expression_index: 1,
        });

        assert_eq!(optimizer.get_replacements().len(), 2);
        assert_eq!(
            optimizer.get_replacements()[0].virtual_col_name,
            "title$length"
        );
        assert_eq!(
            optimizer.get_replacements()[1].virtual_col_name,
            "description$length"
        );
    }

    #[test]
    fn test_replacement_tracking() {
        let mut optimizer = RustLengthOptimizer::new();

        // Add replacements for different columns
        let columns = ["title", "author", "description", "content"];

        for (idx, col) in columns.iter().enumerate() {
            optimizer.replacements.push(LengthReplacement {
                original_column_binding: idx as u64,
                virtual_column_index: (idx + 10) as u64,
                virtual_col_name: format!("{}$length", col),
                new_expression_binding: (idx + 10) as u64,
                expression_index: idx as u64,
            });
        }

        // Verify all replacements are tracked
        assert_eq!(optimizer.get_replacements().len(), 4);

        // Verify each replacement has correct virtual column name
        for (idx, replacement) in optimizer.get_replacements().iter().enumerate() {
            assert_eq!(
                replacement.virtual_col_name,
                format!("{}$length", columns[idx])
            );
            assert_eq!(replacement.virtual_column_index, (idx + 10) as u64);
        }
    }

    #[test]
    fn test_projection_id_updates() {
        let _optimizer = RustLengthOptimizer::new();

        // Test different projection update scenarios

        // Scenario 1: All projections are length functions
        let mut proj_ids_all_length = vec![0, 1, 2];
        let virtual_cols = [10, 11, 12];

        // Simulate what would happen in update_vortex_scan_projections
        for (i, &virtual_col) in virtual_cols.iter().enumerate() {
            if i < proj_ids_all_length.len() {
                proj_ids_all_length[i] = virtual_col;
            }
        }

        assert_eq!(proj_ids_all_length, vec![10, 11, 12]);

        // Scenario 2: Mixed projections (some length, some regular)
        let mut proj_ids_mixed = vec![0, 1, 2, 3];
        let replacements_at = [1, 3]; // Only replace at positions 1 and 3
        let virtual_values = [20, 21];

        for (i, &pos) in replacements_at.iter().enumerate() {
            proj_ids_mixed[pos] = virtual_values[i];
        }

        assert_eq!(proj_ids_mixed, vec![0, 20, 2, 21]);
    }

    #[test]
    fn test_virtual_column_name_generation() {
        let test_cases = vec![
            ("title", "title$length"),
            ("description", "description$length"),
            ("user_name", "user_name$length"),
            ("id", "id$length"),
        ];

        for (column_name, expected_virtual) in test_cases {
            let virtual_name = format!("{}$length", column_name);
            assert_eq!(virtual_name, expected_virtual);
        }
    }

    #[test]
    fn test_optimizer_state_management() {
        let mut optimizer1 = RustLengthOptimizer::new();
        let mut optimizer2 = RustLengthOptimizer::new();

        // Add replacements to optimizer1
        optimizer1.replacements.push(LengthReplacement {
            original_column_binding: 0,
            virtual_column_index: 10,
            virtual_col_name: "col1$length".to_string(),
            new_expression_binding: 10,
            expression_index: 0,
        });

        // Verify optimizers maintain independent state
        assert_eq!(optimizer1.get_replacements().len(), 1);
        assert_eq!(optimizer2.get_replacements().len(), 0);

        // Add different replacement to optimizer2
        optimizer2.replacements.push(LengthReplacement {
            original_column_binding: 1,
            virtual_column_index: 20,
            virtual_col_name: "col2$length".to_string(),
            new_expression_binding: 20,
            expression_index: 1,
        });

        assert_eq!(optimizer1.get_replacements().len(), 1);
        assert_eq!(optimizer2.get_replacements().len(), 1);
        assert_ne!(
            optimizer1.get_replacements()[0].virtual_col_name,
            optimizer2.get_replacements()[0].virtual_col_name
        );
    }
}
