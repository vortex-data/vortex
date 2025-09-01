// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <algorithm>

#include "duckdb_vx/optimizer_rule.h"

using namespace duckdb;

namespace vortex {

// Global variables to store Rust optimizer callback
static duckdb_vx_rust_visitor_callback g_rust_optimizer_callback = nullptr;
static void *g_rust_optimizer_user_data = nullptr;

// C++ wrapper for Rust optimizer callback - this is the actual optimizer function
static void VortexLengthOptimizeFunction(OptimizerExtensionInput &input, duckdb::unique_ptr<LogicalOperator> &plan) {
    if (g_rust_optimizer_callback && plan) {
        g_rust_optimizer_callback(plan.get(), g_rust_optimizer_user_data);
    }
}

class VortexLengthExtension : public OptimizerExtension {
public:
    VortexLengthExtension() {
        optimize_function = VortexLengthOptimizeFunction;
    }
    
    static void Register(DatabaseInstance &db) {
        try {
            auto &config = DBConfig::GetConfig(db);
            
            // Create the extension and ensure function pointer is set
            OptimizerExtension optimizer;
            optimizer.optimize_function = VortexLengthOptimizeFunction;

            config.optimizer_extensions.push_back(std::move(optimizer));
        } catch (std::exception &e) {
            throw e;
        }
    }
};

} // namespace vortex

// ==============================================
// C API Implementation for Rust FFI
// ==============================================

// Basic operator inspection functions
extern "C" DUCKDB_VX_LOGICAL_OPERATOR_TYPE duckdb_vx_get_operator_type(duckdb_vx_logical_operator op) {
    if (!op)
        return DUCKDB_VX_LOGICAL_UNKNOWN;

    auto &logical_op = *reinterpret_cast<LogicalOperator *>(op);
    switch (logical_op.type) {
    case LogicalOperatorType::LOGICAL_GET:
        return DUCKDB_VX_LOGICAL_GET;
    case LogicalOperatorType::LOGICAL_PROJECTION:
        return DUCKDB_VX_LOGICAL_PROJECTION;
    case LogicalOperatorType::LOGICAL_FILTER:
        return DUCKDB_VX_LOGICAL_FILTER;
    case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
        return DUCKDB_VX_LOGICAL_JOIN;
    case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
        return DUCKDB_VX_LOGICAL_AGGREGATE;
    default:
        return DUCKDB_VX_LOGICAL_UNKNOWN;
    }
}

extern "C" uint64_t duckdb_vx_get_children_count(duckdb_vx_logical_operator op) {
    if (!op)
        return 0;
    auto &logical_op = *reinterpret_cast<LogicalOperator *>(op);
    return logical_op.children.size();
}

extern "C" duckdb_vx_logical_operator duckdb_vx_get_child(duckdb_vx_logical_operator op, uint64_t index) {
    if (!op)
        return nullptr;
    auto &logical_op = *reinterpret_cast<LogicalOperator *>(op);
    if (index >= logical_op.children.size())
        return nullptr;
    return logical_op.children[index].get();
}

extern "C" uint64_t duckdb_vx_get_expressions_count(duckdb_vx_logical_operator op) {
    if (!op)
        return 0;
    auto &logical_op = *reinterpret_cast<LogicalOperator *>(op);
    return logical_op.expressions.size();
}

extern "C" duckdb_vx_expr duckdb_vx_get_expression(duckdb_vx_logical_operator op, uint64_t index) {
    if (!op)
        return nullptr;
    auto &logical_op = *reinterpret_cast<LogicalOperator *>(op);
    if (index >= logical_op.expressions.size())
        return nullptr;
    return reinterpret_cast<duckdb_vx_expr>(logical_op.expressions[index].get());
}

extern "C" void duckdb_vx_set_expression(duckdb_vx_logical_operator op, uint64_t index, duckdb_vx_expr expr) {
    if (!op || !expr)
        return;
    auto &logical_op = *reinterpret_cast<LogicalOperator *>(op);
    if (index >= logical_op.expressions.size())
        return;

    // Transfer ownership of the expression
    logical_op.expressions[index].reset(reinterpret_cast<Expression *>(expr));
}

// LogicalGet specific functions
extern "C" char *duckdb_vx_get_function_name(duckdb_vx_logical_operator get_op) {
    if (!get_op)
        return nullptr;
    auto &logical_op = *reinterpret_cast<LogicalOperator *>(get_op);
    if (logical_op.type != LogicalOperatorType::LOGICAL_GET)
        return nullptr;

    auto &get = logical_op.Cast<LogicalGet>();
    return strdup(get.function.name.c_str());
}

extern "C" char **duckdb_vx_get_column_names(duckdb_vx_logical_operator get_op, uint64_t *count) {
    if (!get_op || !count)
        return nullptr;
    auto &logical_op = *reinterpret_cast<LogicalOperator *>(get_op);
    if (logical_op.type != LogicalOperatorType::LOGICAL_GET)
        return nullptr;

    auto &get = logical_op.Cast<LogicalGet>();
    *count = get.names.size();

    char **names = (char **)malloc(sizeof(char *) * get.names.size());
    for (size_t i = 0; i < get.names.size(); i++) {
        names[i] = strdup(get.names[i].c_str());
    }
    return names;
}

extern "C" uint64_t *duckdb_vx_get_projection_ids(duckdb_vx_logical_operator get_op, uint64_t *count) {
    if (!get_op || !count)
        return nullptr;
    auto &logical_op = *reinterpret_cast<LogicalOperator *>(get_op);
    if (logical_op.type != LogicalOperatorType::LOGICAL_GET)
        return nullptr;

    auto &get = logical_op.Cast<LogicalGet>();
    *count = get.projection_ids.size();

    uint64_t *ids = (uint64_t *)malloc(sizeof(uint64_t) * get.projection_ids.size());
    for (size_t i = 0; i < get.projection_ids.size(); i++) {
        ids[i] = get.projection_ids[i];
    }
    return ids;
}

extern "C" void duckdb_vx_update_projection_ids(duckdb_vx_logical_operator get_op,
                                                uint64_t *new_projection_ids, uint64_t count) {
    if (!get_op || !new_projection_ids)
        return;
    auto &logical_op = *reinterpret_cast<LogicalOperator *>(get_op);
    if (logical_op.type != LogicalOperatorType::LOGICAL_GET)
        return;

    auto &get = logical_op.Cast<LogicalGet>();
    get.projection_ids.clear();
    for (uint64_t i = 0; i < count; i++) {
        get.projection_ids.push_back(new_projection_ids[i]);
    }
}

extern "C" void duckdb_vx_add_column_id(duckdb_vx_logical_operator get_op, uint64_t column_id) {
    if (!get_op)
        return;
    auto &logical_op = *reinterpret_cast<LogicalOperator *>(get_op);
    if (logical_op.type != LogicalOperatorType::LOGICAL_GET)
        return;

    auto &get = logical_op.Cast<LogicalGet>();
    get.AddColumnId(column_id);
}

extern "C" void duckdb_vx_clear_column_ids(duckdb_vx_logical_operator get_op) {
    if (!get_op)
        return;
    auto &logical_op = *reinterpret_cast<LogicalOperator *>(get_op);
    if (logical_op.type != LogicalOperatorType::LOGICAL_GET)
        return;

    auto &get = logical_op.Cast<LogicalGet>();
    get.ClearColumnIds();
}

// Get detailed string representation of LogicalGet operator
extern "C" char *duckdb_vx_logical_get_to_string(duckdb_vx_logical_operator get_op) {
    try {
        if (!get_op) {
            return nullptr;
        }

        auto &logical_op = *reinterpret_cast<LogicalOperator *>(get_op);
        if (logical_op.type != LogicalOperatorType::LOGICAL_GET) {
            return nullptr;
        }

        auto &get = logical_op.Cast<LogicalGet>();

        // Create detailed string representation
        std::string str = "LogicalGet:\n";
        str += "  Function: " + get.function.name + "\n";
        str += "  Table Index: " + std::to_string(get.table_index) + "\n";
        str += "  Columns Idx: [";

        auto &column_ids = get.GetColumnIds();
        for (size_t i = 0; i < column_ids.size(); i++) {
            if (i > 0)
                str += ", ";
            str += std::to_string(column_ids[i].GetPrimaryIndex());
        }
        str += "]\n";

        str += "  Columns Names: [";

        if (!get.names.empty()) {
            for (size_t i = 0; i < get.names.size(); i++) {
                if (i > 0)
                    str += ", ";
                str += get.names[i];
            }
        }
        str += "]\n";

        str += "  Projection IDs: [";
        if (!get.projection_ids.empty()) {
            for (size_t i = 0; i < get.projection_ids.size(); i++) {
                if (i > 0)
                    str += ", ";
                str += std::to_string(get.projection_ids[i]);
            }
        }
        str += "]";

        // Allocate C string and copy
        char *result = static_cast<char *>(malloc(str.length() + 1));
        if (result) {
            strcpy(result, str.c_str());
        }
        return result;
    } catch (...) {
        return nullptr;
    }
}

// Get detailed string representation of LogicalProjection operator
extern "C" char *duckdb_vx_logical_projection_to_string(duckdb_vx_logical_operator proj_op) {
    try {
        if (!proj_op) {
            return nullptr;
        }

        auto &logical_op = *reinterpret_cast<LogicalOperator *>(proj_op);
        if (logical_op.type != LogicalOperatorType::LOGICAL_PROJECTION) {
            return nullptr;
        }

        // Create detailed string representation
        std::string str = "LogicalProjection:\n";
        str += "  Expressions: [\n";

        for (size_t i = 0; i < logical_op.expressions.size(); i++) {
            str += "    [" + std::to_string(i) + "] " + logical_op.expressions[i]->ToString() + "\n";
        }
        str += "  ]";

        // Allocate C string and copy
        char *result = static_cast<char *>(malloc(str.length() + 1));
        if (result) {
            strcpy(result, str.c_str());
        }
        return result;
    } catch (...) {
        return nullptr;
    }
}

// Expression functions
extern "C" DUCKDB_VX_EXPRESSION_TYPE duckdb_vx_get_expression_type(duckdb_vx_expr expr) {
    if (!expr)
        return DUCKDB_VX_EXPRESSION_UNKNOWN;

    auto &expression = *reinterpret_cast<Expression *>(expr);
    switch (expression.type) {
    case ExpressionType::BOUND_COLUMN_REF:
        return DUCKDB_VX_BOUND_COLUMN_REF;
    case ExpressionType::BOUND_FUNCTION:
        return DUCKDB_VX_BOUND_FUNCTION;
    case ExpressionType::VALUE_CONSTANT:
        return DUCKDB_VX_CONSTANT;
    default:
        return DUCKDB_VX_EXPRESSION_UNKNOWN;
    }
}

extern "C" char *duckdb_vx_get_function_name_from_expr(duckdb_vx_expr expr) {
    if (!expr)
        return nullptr;
    auto &expression = *reinterpret_cast<Expression *>(expr);

    if (expression.type == ExpressionType::BOUND_FUNCTION) {
        auto &func_expr = expression.Cast<BoundFunctionExpression>();
        return strdup(func_expr.function.name.c_str());
    }
    return nullptr;
}

extern "C" uint64_t duckdb_vx_get_function_arg_count(duckdb_vx_expr expr) {
    if (!expr)
        return 0;
    auto &expression = *reinterpret_cast<Expression *>(expr);

    if (expression.type == ExpressionType::BOUND_FUNCTION) {
        auto &func_expr = expression.Cast<BoundFunctionExpression>();
        return func_expr.children.size();
    }
    return 0;
}

extern "C" duckdb_vx_expr duckdb_vx_get_function_arg(duckdb_vx_expr expr, uint64_t index) {
    if (!expr)
        return nullptr;
    auto &expression = *reinterpret_cast<Expression *>(expr);

    if (expression.type == ExpressionType::BOUND_FUNCTION) {
        auto &func_expr = expression.Cast<BoundFunctionExpression>();
        if (index >= func_expr.children.size())
            return nullptr;
        return reinterpret_cast<duckdb_vx_expr>(func_expr.children[index].get());
    }
    return nullptr;
}

extern "C" char *duckdb_vx_get_column_alias(duckdb_vx_expr expr) {
    if (!expr)
        return nullptr;
    auto &expression = *reinterpret_cast<Expression *>(expr);

    if (expression.type == ExpressionType::BOUND_COLUMN_REF) {
        auto &col_ref = expression.Cast<BoundColumnRefExpression>();
        return strdup(col_ref.alias.c_str());
    }
    return nullptr;
}

extern "C" duckdb_vx_column_binding duckdb_vx_get_column_binding(duckdb_vx_expr expr) {
    duckdb_vx_column_binding binding = {0, 0};
    if (!expr)
        return binding;

    auto &expression = *reinterpret_cast<Expression *>(expr);
    if (expression.type == ExpressionType::BOUND_COLUMN_REF) {
        auto &col_ref = expression.Cast<BoundColumnRefExpression>();
        binding.table_index = col_ref.binding.table_index;
        binding.column_index = col_ref.binding.column_index;
    }
    return binding;
}

extern "C" duckdb_vx_expr duckdb_vx_create_column_ref(const char *name, duckdb_vx_column_binding binding,
                                                      uint64_t depth) {
    if (!name)
        return nullptr;

    Expression *col_ref = new BoundColumnRefExpression(
         std::string(name),
        LogicalType::INTEGER, ColumnBinding(binding.table_index, binding.column_index),
         depth
    );

    return reinterpret_cast<duckdb_vx_expr>(col_ref);
}

extern "C" void duckdb_vx_update_column_binding(duckdb_vx_expr expr, duckdb_vx_column_binding binding) {
    if (!expr)
        return;
    auto &expression = *reinterpret_cast<Expression *>(expr);

    if (expression.type == ExpressionType::BOUND_COLUMN_REF) {
        auto &col_ref = expression.Cast<BoundColumnRefExpression>();
        col_ref.binding.table_index = binding.table_index;
        col_ref.binding.column_index = binding.column_index;
    }
}

// Visitor pattern implementation
extern "C" void duckdb_vx_visit_operators(duckdb_vx_logical_operator plan,
                                          duckdb_vx_rust_visitor_callback callback, void *user_data) {
    if (!plan || !callback)
        return;

    auto &logical_op = *reinterpret_cast<LogicalOperator *>(plan);

    // Call the Rust callback on this operator
    callback(plan, user_data);

    // Recursively visit children
    for (auto &child : logical_op.children) {
        duckdb_vx_visit_operators(child.get(), callback, user_data);
    }
}

extern "C" void duckdb_vx_register_rust_optimizer(duckdb_database db_handle,
                                                  duckdb_vx_rust_visitor_callback optimizer_func,
                                                  void *user_data) {
    std::cout << "🔧 REGISTERING: Rust-based optimizer..." << std::endl;

    if (!db_handle || !optimizer_func) {
        std::cout << "❌ ERROR: NULL parameters passed to Rust optimizer registration" << std::endl;
        return;
    }

    try {
        // Store the Rust callback and user data
        vortex::g_rust_optimizer_callback = optimizer_func;
        vortex::g_rust_optimizer_user_data = user_data;

        // Get the DuckDB instance
        struct DatabaseWrapper {
            void *internal_ptr;
        };

        auto wrapper = reinterpret_cast<DatabaseWrapper *>(db_handle);
        auto db = reinterpret_cast<DuckDB *>(wrapper->internal_ptr);

        // Register the optimizer using VortexLengthExtension
        vortex::VortexLengthExtension::Register(*db->instance);

        std::cout << "✅ SUCCESS: Rust-based optimizer registered!" << std::endl;
    } catch (std::exception &e) {
        std::cout << "❌ EXCEPTION during Rust optimizer registration: " << e.what() << std::endl;
    }
}

// Memory management functions
extern "C" void duckdb_vx_free_string(char *str) {
    if (str)
        free(str);
}

extern "C" void duckdb_vx_free_string_array(char **arr, uint64_t count) {
    if (!arr)
        return;
    for (uint64_t i = 0; i < count; i++) {
        if (arr[i])
            free(arr[i]);
    }
    free(arr);
}

extern "C" void duckdb_vx_free_uint64_array(uint64_t *arr) {
    if (arr)
        free(arr);
}

// C API for registering the optimizer from Rust (deprecated - use duckdb_vx_register_rust_optimizer)
extern "C" void duckdb_vx_register_optimizer(duckdb_database db_handle) {
    std::cout << "⚠️  WARNING: duckdb_vx_register_optimizer is deprecated. Use duckdb_vx_register_rust_optimizer instead." << std::endl;
    
    // For backward compatibility, register with a null callback
    // This will just register the extension but won't actually optimize anything
    // unless duckdb_vx_register_rust_optimizer is called with a proper callback
    duckdb_vx_register_rust_optimizer(db_handle, nullptr, nullptr);
}

// Get string representation of logical operator
extern "C" char *duckdb_vx_logical_operator_to_string(duckdb_vx_logical_operator op) {
    try {
        if (!op) {
            return nullptr;
        }

        auto *logical_op = reinterpret_cast<duckdb::LogicalOperator *>(op);
        std::string str = logical_op->ToString();

        // Allocate C string and copy
        char *result = static_cast<char *>(malloc(str.length() + 1));
        if (result) {
            strcpy(result, str.c_str());
        }
        return result;
    } catch (...) {
        return nullptr;
    }
}