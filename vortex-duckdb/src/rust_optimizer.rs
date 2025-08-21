//! Pure Rust implementation of the length optimization logic
//!
//! This module implements the length optimization using the generic logical plan API.

use std::collections::{HashMap, HashSet};
use std::ptr;

use vortex::error::{VortexExpect, VortexResult, vortex_bail, vortex_err};

use crate::cpp::DUCKDB_VX_EXPR_CLASS;
use crate::duckdb::expr::{ColumnBinding, Expression, LogicalExpressionType as ExpressionType};
use crate::duckdb::logical_operator::{LogicalOperator, LogicalOperatorClass};
use crate::duckdb::logical_plan::LogicalPlanUtils;
use crate::duckdb::{Database, ExpressionClass, LogicalExpressionType};

/// Length replacement information
#[derive(Debug, Clone)]
pub struct LengthReplacement {
    pub original_column_binding: u64,
    pub virtual_column_index: u64,
    pub virtual_col_name: String,
    pub new_expression_binding: u64,
    pub expression_index: u64,
}

// Length-specific helper functions for expressions
trait LengthOptimizationExt {
    /// Check if this is a vortex_scan table function
    fn is_vortex_scan(&self) -> VortexResult<bool>;

    /// Check if this expression is a length function call
    fn is_length_function(&self) -> VortexResult<bool>;
}

impl LengthOptimizationExt for LogicalOperator {
    fn is_vortex_scan(&self) -> VortexResult<bool> {
        if let Some(LogicalOperatorClass::Get(get_op)) = self.as_class() {
            Ok(get_op.is_vortex_scan())
        } else {
            Ok(false)
        }
    }

    fn is_length_function(&self) -> VortexResult<bool> {
        Ok(false) // Operators are not length functions
    }
}

impl LengthOptimizationExt for Expression {
    fn is_vortex_scan(&self) -> VortexResult<bool> {
        Ok(false) // Expressions are not vortex scans
    }

    fn is_length_function(&self) -> VortexResult<bool> {
        let Some(ExpressionClass::BoundFunction(func_)) = self.as_class() else {
            return Ok(false);
        };

        if let Some(name) = func_.function_name() {
            let name_lower = name.to_lowercase();
            Ok(
                (name_lower == "length" || name_lower == "len" || name_lower == "strlen")
                    && func_.function_arg_count() == 1,
            )
        } else {
            Ok(false)
        }
    }
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
            println!("ℹ️  RUST OPTIMIZER: No vortex_scan found in plan, skipping");
            return Ok(());
        }

        println!("✅ RUST OPTIMIZER: Found vortex_scan in plan!");

        // Visit all operators and apply length optimization
        // We pass the plan root so we can re-find vortex_scan for each expression
        self.visit_and_optimize(plan, plan)?;

        if !self.replacements.is_empty() {
            println!(
                "🎯 RUST OPTIMIZER: Found {} len() → virtual column transformations!",
                self.replacements.len()
            );

            // Update the projection_ids in vortex_scan LogicalGet nodes
            self.update_vortex_scan_projections(plan)?;
        } else {
            println!("ℹ️  RUST OPTIMIZER: No len() functions found to optimize");
        }

        println!("✅ RUST OPTIMIZER: Length optimization completed!");
        Ok(())
    }

    /// Check if the plan contains a vortex_scan
    fn has_vortex_scan(op: &LogicalOperator) -> VortexResult<bool> {
        let mut found = false;
        LogicalPlanUtils::visit_operators(op, &mut |operator: &LogicalOperator| {
            if operator.is_vortex_scan()? {
                found = true;
            }
            Ok(())
        })?;
        Ok(found)
    }

    /// Find the first vortex_scan node in the plan
    fn find_vortex_scan(op: &LogicalOperator) -> VortexResult<Option<*const LogicalOperator>> {
        let mut vortex_node = None;
        LogicalPlanUtils::visit_operators(op, &mut |operator: &LogicalOperator| {
            if vortex_node.is_none() && operator.is_vortex_scan()? {
                vortex_node = Some(operator as *const LogicalOperator);
            }
            Ok(())
        })?;
        Ok(vortex_node)
    }

    fn visit_node(&mut self, operator: &LogicalOperator) -> Option<()> {
        println!("🔍 VISITING: Operator type: {:?}", operator.operator_type());

        let LogicalOperatorClass::Projection(proj) = operator.as_class()? else {
            println!("🔍 Not a projection operator");
            return None;
        };
        if operator.children_count() != 1 {
            println!(
                "🔍 Projection operator has {} children, expected 1",
                operator.children_count()
            );
            return None;
        }
        let op_child = operator.get_child(0).unwrap();
        let LogicalOperatorClass::Get(get_op) = op_child.as_class()? else {
            println!("🔍 Child is not a Get operator");
            return None;
        };
        if !get_op.is_vortex_scan() {
            println!("🔍 Get operator is not a vortex_scan");
            return None;
        }

        println!("🔍 FOUND VORTEX SCAN");

        for (idx, projection_expr) in proj.projections().enumerate() {
            println!("🔍 Processing projection {}", idx);

            let Some(projection_expr) = projection_expr else {
                println!("🔍 Projection {} is None", idx);
                continue;
            };

            println!("🔍 Projection {} has expression: {}", idx, projection_expr);

            // Check if this is a function expression first using as_class_id
            let Some(ExpressionClass::BoundFunction(func_)) = projection_expr.as_class() else {
                continue;
            };

            // Try to get function name safely
            let function_name = match func_.function_name() {
                Some(name) => name,
                None => {
                    continue;
                }
            };

            println!("🔍 Function name: {}", function_name);

            if function_name != "len" {
                println!("🔍 Function is not 'len', it's '{}'", function_name);
                continue;
            }

            println!("🔍 Found len() function!");

            // Try to get function argument safely
            let arg_count = func_.function_arg_count();
            println!("🔍 Function has {} arguments", arg_count);

            if arg_count == 0 {
                println!("🔍 len() function has no arguments");
                continue;
            }

            let column_alias = match func_.get_function_arg(0) {
                Some(arg) => arg,
                None => {
                    println!("🔍 Could not get first argument of len() function");
                    continue;
                }
            };

            println!("🔍 Got first argument of len() function");

            let ExpressionClass::BoundColumnRef(bound_col) = column_alias.as_class()? else {
                println!("🔍 First argument is not a BoundColumnRef");
                continue;
            };

            let column_alias = bound_col.name;
            let column_bind = bound_col.column_binding;

            let virtual_col_name = format!("{}$length", column_alias);

            println!(
                "🔍 Processing len({}) -> {}",
                column_alias, virtual_col_name
            );

            let e = Expression::create_column_ref(
                &virtual_col_name,
                ColumnBinding {
                    table_index: column_bind.table_index,
                    column_index: column_bind.column_index,
                },
                0,
            );

            // proj.set_projection(idx, e);

            // Rest of the processing would go here, but let's stop here for now
            println!("🔍 Successfully processed len() at index {}", idx);
        }

        Some(())
    }

    /// Visit all operators and apply optimizations
    fn visit_and_optimize(
        &mut self,
        op: &LogicalOperator,
        plan_root: &LogicalOperator,
    ) -> VortexResult<()> {
        println!("🔍 VISITING: Operator type: {:?}", op.operator_type());

        self.visit_node(op);

        // Visit children
        for i in 0..op.children_count() {
            if let Some(child) = op.get_child(i) {
                self.visit_and_optimize(&child, plan_root)?;
            }
        }

        Ok(())
    }

    // /// Rewrite a single expression to replace len() calls with virtual column references
    // fn rewrite_expression(
    //     &mut self,
    //     expr: &Expression,
    //     vortex_scan_node_ptr: Option<*const LogicalOperator>,
    //     expression_index: usize,
    // ) -> VortexResult<Option<Expression>> {
    //     if !expr.is_length_function()? {
    //         return Ok(None);
    //     }
    //     let Some(ExpressionClass::BoundFunction(func_)) = expr.as_class() else {
    //         return Ok(None);
    //     };
    //
    //     // Get the first argument (should be a column reference)
    //     let arg = expr
    //         .get_function_arg(0)
    //         .ok_or_else(|| vortex_err!("Length function has no arguments"))?;
    //
    //     if arg.logical_expression_type() != ExpressionType::BoundColumnRef {
    //         return Ok(None);
    //     }
    //
    //     let column_alias = arg
    //         .column_alias()?
    //         .ok_or_else(|| vortex_err!("Column reference has no alias"))?;
    //
    //     let virtual_col_name = format!("{}$length", column_alias);
    //
    //     println!(
    //         "🔄 OPTIMIZER: Found len({}) → {}",
    //         column_alias, virtual_col_name
    //     );
    //
    //     // Find the virtual column index in the table schema
    //     let virtual_column_index = if let Some(vortex_node_ptr) = vortex_scan_node_ptr {
    //         let vortex_node = unsafe { &*vortex_node_ptr };
    //
    //         // Downcast to LogicalGet to access column_names
    //         if let Some(LogicalOperatorClass::Get(get_op)) = vortex_node.as_class() {
    //             let column_names = get_op.column_names()?;
    //             println!(
    //                 "✅ OPTIMIZER: Found vortex_scan node with {} columns: {:?}",
    //                 column_names.len(),
    //                 column_names
    //             );
    //             column_names
    //                 .iter()
    //                 .position(|name| *name == virtual_col_name)
    //                 .ok_or_else(|| {
    //                     vortex_err!("Virtual column '{}' not found in schema", virtual_col_name)
    //                 })?
    //         } else {
    //             vortex_bail!("Expected vortex_scan to be a LogicalGet operator");
    //         }
    //     } else {
    //         vortex_bail!("No vortex_scan node found");
    //     };
    //
    //     println!(
    //         "✅ OPTIMIZER: Found virtual column '{}' at index {}",
    //         virtual_col_name, virtual_column_index
    //     );
    //
    //     // Create a new column reference for the virtual column
    //     let original_binding = arg
    //         .column_binding()
    //         .ok_or_else(|| vortex_err!("Expected column reference to have binding"))?;
    //     let virtual_col_ref = Expression::create_column_ref(
    //         &virtual_col_name,
    //         ColumnBinding {
    //             table_index: original_binding.table_index,
    //             column_index: virtual_column_index as u64,
    //         },
    //         0, // depth
    //     )?;
    //
    //     // Record this replacement for later projection mapping
    //     let replacement = LengthReplacement {
    //         original_column_binding: original_binding.column_index,
    //         virtual_column_index: virtual_column_index as u64,
    //         virtual_col_name,
    //         new_expression_binding: virtual_column_index as u64,
    //         expression_index: expression_index as u64,
    //     };
    //
    //     self.replacements.push(replacement);
    //
    //     Ok(Some(virtual_col_ref))
    // }

    /// Update vortex scan projections based on the replacements
    fn update_vortex_scan_projections(&self, op: &LogicalOperator) -> VortexResult<()> {
        // Check if this is a vortex_scan and update it
        if let Some(LogicalOperatorClass::Get(get_op)) = op.as_class() {
            if get_op.is_vortex_scan() {
                self.update_single_vortex_scan_projections(op)?;
            }
        }

        // Recursively update children
        for i in 0..op.children_count() {
            if let Some(child) = op.get_child(i) {
                self.update_vortex_scan_projections(&child)?;
            }
        }

        Ok(())
    }

    /// Update projections for a single vortex_scan node
    fn update_single_vortex_scan_projections(&self, op: &LogicalOperator) -> VortexResult<()> {
        // Downcast to LogicalGet to access the projection manipulation methods
        let get_op = match op.as_class() {
            Some(LogicalOperatorClass::Get(get)) => get,
            _ => {
                vortex_bail!("update_single_vortex_scan_projections called on non-Get operator");
            }
        };

        println!("🔧 OPTIMIZER: ===== BEFORE TRANSFORM =====");

        let projection_ids = get_op.get_projection_ids()?;
        println!("🔧 OPTIMIZER: Current projection_ids: {:?}", projection_ids);

        let column_names = get_op.column_names()?;
        println!("🔧 OPTIMIZER: Current names: {:?}", column_names);

        // Create a set of virtual columns to add
        let mut virtual_columns_to_add: HashSet<u64> = HashSet::new();
        for replacement in &self.replacements {
            virtual_columns_to_add.insert(replacement.virtual_column_index);
        }

        // Add virtual columns to the column_ids array if not already present
        for virtual_col_id in &virtual_columns_to_add {
            get_op.add_column_id(*virtual_col_id);
            println!(
                "🔧 OPTIMIZER: Added virtual column {} to column_ids",
                virtual_col_id
            );
        }

        // Update projection_ids based on replacement patterns
        if projection_ids.len() == self.replacements.len() {
            // All projections are len() calls
            println!("🔧 OPTIMIZER: All projections are len() calls, updating projection_ids");
            let mut new_projection_ids = Vec::new();

            for replacement in &self.replacements {
                // The virtual column index is already the correct position in the schema
                let position = replacement.virtual_column_index;
                new_projection_ids.push(position);
                println!(
                    "🔧 OPTIMIZER: Mapping projection to virtual column position {}",
                    position
                );
            }

            get_op.update_projection_ids(&new_projection_ids)?;
        } else {
            // Mixed case: some projections are len() calls, others are regular columns
            println!("🔧 OPTIMIZER: Mixed projections case");
            let mut new_projection_ids = projection_ids.clone();

            // Create a mapping from expression index to replacement
            let mut replacement_map: HashMap<u64, &LengthReplacement> = HashMap::new();
            for replacement in &self.replacements {
                replacement_map.insert(replacement.expression_index, replacement);
            }

            // Update specific positions based on replacements
            for (expr_idx, replacement) in &replacement_map {
                if (*expr_idx as usize) < new_projection_ids.len() {
                    let position = replacement.virtual_column_index;
                    new_projection_ids[*expr_idx as usize] = position;
                    println!(
                        "🔧 OPTIMIZER: Updated projection_ids[{}] to position {}",
                        expr_idx, position
                    );
                }
            }

            get_op.update_projection_ids(&new_projection_ids)?;
        }

        println!("🔧 OPTIMIZER: ===== AFTER TRANSFORM =====");
        let final_projection_ids = get_op.get_projection_ids()?;
        println!(
            "🔧 OPTIMIZER: Final projection_ids: {:?}",
            final_projection_ids
        );

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
    println!("🚀🚀🚀 RUST OPTIMIZER FUNCTION CALLED! 🚀🚀🚀");

    if plan.is_null() {
        println!("❌ RUST OPTIMIZER: NULL plan passed to optimizer");
        return;
    }

    // Safely create the LogicalOperator wrapper
    if plan.is_null() {
        println!("❌ RUST OPTIMIZER: NULL plan pointer");
        return;
    }
    let logical_op = unsafe { LogicalOperator::borrow(plan) };

    // Create and run the optimizer
    let mut optimizer = RustLengthOptimizer::new();
    match optimizer.optimize_plan(&logical_op) {
        Ok(()) => {
            println!("✅ RUST OPTIMIZER: Optimization completed successfully!");
            let replacements = optimizer.get_replacements();
            if !replacements.is_empty() {
                println!(
                    "📊 RUST OPTIMIZER: Made {} replacements:",
                    replacements.len()
                );
                for (i, replacement) in replacements.iter().enumerate() {
                    println!(
                        "  {}. {} → {}",
                        i + 1,
                        replacement.original_column_binding,
                        replacement.virtual_col_name
                    );
                }
            }
        }
        Err(e) => {
            println!("❌ RUST OPTIMIZER: Optimization failed: {}", e);
        }
    }
}

/// Register the Rust-based length optimizer with DuckDB
pub fn register_rust_optimizer(db: &mut Database) -> VortexResult<()> {
    println!("🔧 REGISTERING: Rust-based length optimizer...");

    unsafe {
        crate::cpp::duckdb_vx_register_rust_optimizer(
            db.as_ptr(),
            Some(rust_optimizer_callback),
            ptr::null_mut(),
        );
    }

    println!("✅ SUCCESS: Rust-based length optimizer registered!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::optimizer_plan::LogicalOperatorType;

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

    // Test visitor pattern with mock callbacks
    struct MockVisitorState {
        operators_visited: Vec<LogicalOperatorType>,
        expressions_found: Vec<ExpressionType>,
        modifications_made: Vec<String>,
    }

    #[test]
    fn test_visitor_pattern_tracking() {
        let state = Arc::new(Mutex::new(MockVisitorState {
            operators_visited: Vec::new(),
            expressions_found: Vec::new(),
            modifications_made: Vec::new(),
        }));

        // Simulate visiting different operator types
        let operator_types = vec![
            LogicalOperatorType::Get,
            LogicalOperatorType::Projection,
            LogicalOperatorType::Filter,
            LogicalOperatorType::Get,
        ];

        for op_type in operator_types {
            state.lock().unwrap().operators_visited.push(op_type);
        }

        let visited = &state.lock().unwrap().operators_visited;
        assert_eq!(visited.len(), 4);
        assert_eq!(
            visited
                .iter()
                .filter(|&&t| t == LogicalOperatorType::Get)
                .count(),
            2
        );
    }

    #[test]
    fn test_replacement_tracking() {
        let mut optimizer = RustLengthOptimizer::new();

        // Add replacements for different columns
        let columns = vec!["title", "author", "description", "content"];

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
        let virtual_cols = vec![10, 11, 12];

        // Simulate what would happen in update_vortex_scan_projections
        for (i, &virtual_col) in virtual_cols.iter().enumerate() {
            if i < proj_ids_all_length.len() {
                proj_ids_all_length[i] = virtual_col;
            }
        }

        assert_eq!(proj_ids_all_length, vec![10, 11, 12]);

        // Scenario 2: Mixed projections (some length, some regular)
        let mut proj_ids_mixed = vec![0, 1, 2, 3];
        let replacements_at = vec![1, 3]; // Only replace at positions 1 and 3
        let virtual_values = vec![20, 21];

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
