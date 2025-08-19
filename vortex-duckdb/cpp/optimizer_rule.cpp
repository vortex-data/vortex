// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <duckdb/main/client_context.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/optimizer/optimizer_extension.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/common/string_util.hpp>
#include <duckdb/planner/logical_operator_visitor.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <iostream>

#include "duckdb_vx/optimizer_rule.h"

using namespace duckdb;

namespace vortex {

// Expression rewriter that transforms len(column) to column$length
class LengthRewriter {
public:
    static bool RewriteExpression(unique_ptr<Expression> &expr) {
        bool changed = false;
        
        // Check if this is a function expression
        if (expr->type == ExpressionType::BOUND_FUNCTION) {
            auto &func_expr = expr->Cast<BoundFunctionExpression>();
            
            // Check if it's a length function
            auto func_name = StringUtil::Lower(func_expr.function.name);
            if (func_name == "length" || func_name == "len" || func_name == "strlen") {
                // Check if it has exactly one argument and it's a column reference
                if (func_expr.children.size() == 1 && 
                    func_expr.children[0]->type == ExpressionType::BOUND_COLUMN_REF) {
                    
                    auto &col_ref = func_expr.children[0]->Cast<BoundColumnRefExpression>();
                    
                    // Create new column reference for virtual column
                    string virtual_column_name = col_ref.GetName() + "$length";
                    
                    std::cout << "🔄 OPTIMIZER: Rewriting len(" << col_ref.GetName() 
                             << ") → " << virtual_column_name << std::endl;
                    
                    // Replace the function with a column reference to the virtual column
                    auto new_col_ref = make_uniq<BoundColumnRefExpression>(
                        virtual_column_name,
                        func_expr.return_type,
                        col_ref.binding
                    );
                    
                    expr = std::move(new_col_ref);
                    changed = true;
                }
            }
        }
        
        // Recursively process children
        ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) {
            if (RewriteExpression(child)) {
                changed = true;
            }
        });
        
        return changed;
    }
};

// Logical operator visitor that applies expression rewriting
class VortexOptimizerVisitor : public LogicalOperatorVisitor {
public:
    bool plan_changed = false;

    void VisitOperator(LogicalOperator &op) override {
        // Rewrite expressions in this operator
        for (auto &expr : op.expressions) {
            if (LengthRewriter::RewriteExpression(expr)) {
                plan_changed = true;
            }
        }
        
        // Visit children
        VisitOperatorChildren(op);
    }
};

// Optimizer function that will be called by DuckDB
void optimize_vortex_length(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
    std::cout << "🚀 OPTIMIZER: Vortex length optimization running..." << std::endl;
    
    VortexOptimizerVisitor visitor;
    visitor.VisitOperator(*plan);
    
    if (visitor.plan_changed) {
        std::cout << "✅ OPTIMIZER: Plan transformation completed!" << std::endl;
    } else {
        std::cout << "ℹ️  OPTIMIZER: No len() functions found to optimize" << std::endl;
    }
}

// Simple optimizer extension that logs function calls (for now)
class VortexOptimizerExtension : public OptimizerExtension {
public:
    VortexOptimizerExtension() {
        optimize_function = optimize_vortex_length;
    }

    static void Register(DatabaseInstance &db) {
        auto extension = VortexOptimizerExtension();
        
        auto &config = DBConfig::GetConfig(db);
        config.optimizer_extensions.push_back(extension);
    }
};

} // namespace vortex

// C API for registering the optimizer from Rust
extern "C" void duckdb_vx_register_optimizer(duckdb_database db_handle) {
    if (!db_handle) {
        return;
    }

    try {
    auto db = reinterpret_cast<DuckDB *>(db_handle);
    vortex::VortexOptimizerExtension::Register(*db->instance);
    } catch (std::exception e) {
    std::cout << e.what() << std::endl;
    }
}