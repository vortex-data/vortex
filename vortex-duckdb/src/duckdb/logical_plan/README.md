# DuckDB Logical Plan API

This module provides generic, safe Rust wrappers around DuckDB's logical plan structures. It was extracted from the optimizer-specific code to create reusable building blocks for any DuckDB logical plan manipulation.

## What's Generic (moved here)

### Core Types
- `LogicalOperatorType` - Enum for logical operator types (Get, Projection, Filter, etc.)
- `ExpressionType` - Enum for expression types (BoundColumnRef, BoundFunction, etc.)
- `ColumnBinding` - Structure for column binding information
- `LogicalOperator` - Safe wrapper around DuckDB logical operators
- `Expression` - Safe wrapper around DuckDB expressions

### Utility Functions
- `LogicalPlanUtils::visit_operators()` - Generic plan traversal
- `LogicalPlanUtils::find_operators_by_type()` - Find operators of specific types
- `LogicalPlanUtils::contains_operator_type()` - Check for operator type existence
- `LogicalPlanUtils::find_expressions_by_type()` - Find expressions of specific types

### Core Operations
- Operator tree navigation (children, expressions)
- Expression inspection and creation
- LogicalGet operator management (column names, projection IDs)
- Safe memory management with proper RAII

## What's Specific (remained in rust_optimizer.rs)

### Length Optimization Logic
- `LengthReplacement` - Information about len() → virtual column replacements
- `RustLengthOptimizer` - The actual optimization implementation
- `LengthOptimizationExt` - Trait with length-specific helper methods:
  - `is_vortex_scan()` - Check if operator is a vortex_scan table function
  - `is_length_function()` - Check if expression is a len() function call

### Optimization Algorithm
- Finding vortex_scan nodes in plans
- Detecting len() function calls
- Creating virtual column references
- Updating projection mappings
- Plan tree transformation logic

## Benefits of This Structure

1. **Reusability**: The generic logical plan API can be used for any DuckDB optimizer
2. **Separation of Concerns**: Generic plan manipulation vs. specific optimization logic
3. **Type Safety**: Safe Rust wrappers prevent memory issues
4. **Extensibility**: Easy to add new optimizations using the same building blocks
5. **Maintainability**: Clear boundaries between generic and specific code

## Usage Example

```rust
use crate::duckdb::logical_plan::{LogicalPlanUtils, LogicalOperatorType};

// Generic plan analysis
let get_operators = LogicalPlanUtils::find_operators_by_type(&plan, LogicalOperatorType::Get)?;

// Custom visitor
LogicalPlanUtils::visit_operators(&plan, |op| {
    println!("Visiting operator: {:?}", op.operator_type());
    Ok(())
})?;

// Length-specific optimization (in rust_optimizer.rs)
let mut optimizer = RustLengthOptimizer::new();
optimizer.optimize_plan(&plan)?;
```

This structure allows for clean separation between generic DuckDB plan manipulation utilities and specific optimization implementations.