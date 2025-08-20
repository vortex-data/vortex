// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for DuckDB types - opaque pointers
typedef void* duckdb_vx_logical_operator;
// Use the same expr type as defined in expr.h
#include "expr.h"

// Logical operator types enum (subset of DuckDB's LogicalOperatorType)
typedef enum {
    DUCKDB_VX_LOGICAL_GET = 0,
    DUCKDB_VX_LOGICAL_PROJECTION = 1,
    DUCKDB_VX_LOGICAL_FILTER = 2,
    DUCKDB_VX_LOGICAL_JOIN = 3,
    DUCKDB_VX_LOGICAL_AGGREGATE = 4,
    DUCKDB_VX_LOGICAL_UNKNOWN = 999
} DUCKDB_VX_LOGICAL_OPERATOR_TYPE;

// Expression types and column binding are now defined in expr.h

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

// Get column names from LogicalGet
char** duckdb_vx_get_column_names(duckdb_vx_logical_operator get_op, uint64_t* count);

// Get projection IDs from LogicalGet
uint64_t* duckdb_vx_get_projection_ids(duckdb_vx_logical_operator get_op, uint64_t* count);

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

// Expression functions are now declared in expr.h

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

// Memory management functions are now declared in expr.h

// ==============================================
// Legacy Functions (for backwards compatibility)
// ==============================================

// Register the original Vortex optimizer extension
void duckdb_vx_register_optimizer(duckdb_database db_handle);

#ifdef __cplusplus
}
#endif