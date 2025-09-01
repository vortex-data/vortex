// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"
#include "expr.h"

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================
// Forward Declarations
// ==============================================

// Forward declarations for DuckDB types - opaque pointers
typedef void* duckdb_vx_logical_operator;

// ==============================================
// Type Definitions
// ==============================================

// Logical operator types enum (subset of DuckDB's LogicalOperatorType)
typedef enum {
    DUCKDB_VX_LOGICAL_GET = 0,
    DUCKDB_VX_LOGICAL_PROJECTION = 1,
    DUCKDB_VX_LOGICAL_FILTER = 2,
    DUCKDB_VX_LOGICAL_JOIN = 3,
    DUCKDB_VX_LOGICAL_AGGREGATE = 4,
    DUCKDB_VX_LOGICAL_UNKNOWN = 999
} DUCKDB_VX_LOGICAL_OPERATOR_TYPE;

// Expression types enum (subset of DuckDB's ExpressionType) for logical plan compatibility
typedef enum {
    DUCKDB_VX_BOUND_COLUMN_REF = 0,
    DUCKDB_VX_BOUND_FUNCTION = 1,
    DUCKDB_VX_CONSTANT = 2,
    DUCKDB_VX_EXPRESSION_UNKNOWN = 999
} DUCKDB_VX_EXPRESSION_TYPE;

// Column binding structure for logical plans
typedef struct {
    uint64_t table_index;
    uint64_t column_index;
} duckdb_vx_column_binding;

// Rust callback function type for visiting operators
typedef void (*duckdb_vx_rust_visitor_callback)(duckdb_vx_logical_operator op, void* user_data);

// ==============================================
// Basic Logical Operator Inspection
// ==============================================

// Get operator type
DUCKDB_VX_LOGICAL_OPERATOR_TYPE duckdb_vx_get_operator_type(duckdb_vx_logical_operator op);

// Get string representation of operator
char* duckdb_vx_logical_operator_to_string(duckdb_vx_logical_operator op);

// Get operator children count
uint64_t duckdb_vx_get_children_count(duckdb_vx_logical_operator op);

// Get operator child by index
duckdb_vx_logical_operator duckdb_vx_get_child(duckdb_vx_logical_operator op, uint64_t index);

// Get operator expressions count
uint64_t duckdb_vx_get_expressions_count(duckdb_vx_logical_operator op);

// Get operator expression by index
duckdb_vx_expr duckdb_vx_get_expression(duckdb_vx_logical_operator op, uint64_t index);

// Set operator expression by index
void duckdb_vx_set_expression(duckdb_vx_logical_operator op, uint64_t index, duckdb_vx_expr expr);

// ==============================================
// LogicalGet (Table Scan) Functions
// ==============================================

// Get table function name from LogicalGet
char* duckdb_vx_get_function_name(duckdb_vx_logical_operator get_op);

// Get column names count from LogicalGet
uint64_t duckdb_vx_get_column_names_count(duckdb_vx_logical_operator get_op);

// Get individual column name by index from LogicalGet
char* duckdb_vx_get_column_name(duckdb_vx_logical_operator get_op, uint64_t index);

// Get projection IDs count from LogicalGet
uint64_t duckdb_vx_get_projection_ids_count(duckdb_vx_logical_operator get_op);

// Get individual projection ID by index from LogicalGet
uint64_t duckdb_vx_get_projection_id(duckdb_vx_logical_operator get_op, uint64_t index);

// Update projection IDs in LogicalGet
void duckdb_vx_update_projection_ids(duckdb_vx_logical_operator get_op, 
                                     uint64_t* new_projection_ids,
                                     uint64_t count);

// Add column ID to LogicalGet
void duckdb_vx_add_column_id(duckdb_vx_logical_operator get_op, uint64_t column_id);

// Clear column IDs in LogicalGet
void duckdb_vx_clear_column_ids(duckdb_vx_logical_operator get_op);

// Get detailed string representation of LogicalGet operator
char* duckdb_vx_logical_get_to_string(duckdb_vx_logical_operator get_op);

// ==============================================
// LogicalProjection Functions
// ==============================================

// Get detailed string representation of LogicalProjection operator
char* duckdb_vx_logical_projection_to_string(duckdb_vx_logical_operator proj_op);

// ==============================================
// Expression Functions
// ==============================================

// Logical plan expression functions - using simplified type enum
DUCKDB_VX_EXPRESSION_TYPE duckdb_vx_get_expression_type(duckdb_vx_expr expr);
char* duckdb_vx_get_function_name_from_expr(duckdb_vx_expr expr);
uint64_t duckdb_vx_get_function_arg_count(duckdb_vx_expr expr);
duckdb_vx_expr duckdb_vx_get_function_arg(duckdb_vx_expr expr, uint64_t index);
char* duckdb_vx_get_column_alias(duckdb_vx_expr expr);
duckdb_vx_column_binding duckdb_vx_get_column_binding(duckdb_vx_expr expr);
duckdb_vx_expr duckdb_vx_create_column_ref(const char* name, duckdb_vx_column_binding binding, uint64_t depth);
void duckdb_vx_update_column_binding(duckdb_vx_expr expr, duckdb_vx_column_binding binding);

// ==============================================
// Visitor Pattern
// ==============================================

// Visit all operators in plan tree with Rust callback
void duckdb_vx_visit_operators(duckdb_vx_logical_operator plan,
                              duckdb_vx_rust_visitor_callback callback,
                              void* user_data);

// ==============================================
// Optimizer Registration
// ==============================================

// Register a Rust-based optimizer function
void duckdb_vx_register_rust_optimizer(duckdb_database db_handle,
                                       duckdb_vx_rust_visitor_callback optimizer_func,
                                       void* user_data);

// ==============================================
// Memory Management
// ==============================================

// Memory management functions
void duckdb_vx_free_string(char* str);

#ifdef __cplusplus
}
#endif