// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include <iostream>

#include "duckdb_vx/optimizer_rule.h"

using namespace duckdb;

namespace vortex {

class VortexLengthExtension : public OptimizerExtension {
public:
    VortexLengthExtension() {
        optimize_function = VortexLengthOptimizeFunction;
    }

    // Check if we're dealing with a vortex_scan
    static bool HasVortexScan(LogicalOperator &op) {
        if (op.type == LogicalOperatorType::LOGICAL_GET) {
            auto &get = op.Cast<LogicalGet>();
            return get.function.name == "vortex_scan";
        }
        for (auto &child : op.children) {
            if (HasVortexScan(*child)) {
                return true;
            }
        }
        return false;
    }

    // Simple visitor that looks for len() functions and prints what it finds
    class LengthFunctionVisitor : public LogicalOperatorVisitor {
    public:
        bool found_length_function = false;

        void VisitOperator(LogicalOperator &op) override {
            std::cout << "🔍 VISITING: Operator type: " << (int)op.type << std::endl;
            
            // Look at all expressions in this operator
            for (size_t i = 0; i < op.expressions.size(); i++) {
                std::cout << "🔍 EXPRESSION " << i << ": " << op.expressions[i]->ToString() << std::endl;
                
                CheckExpression(*op.expressions[i]);
            }
            
            // Visit children
            VisitOperatorChildren(op);
        }

        void CheckExpression(Expression &expr) {
            // Check if this is a function expression
            if (expr.type == ExpressionType::BOUND_FUNCTION) {
                auto &func_expr = expr.Cast<BoundFunctionExpression>();
                std::cout << "🎯 FOUND FUNCTION: " << func_expr.function.name << std::endl;
                
                // Check if it's a length function
                auto func_name = StringUtil::Lower(func_expr.function.name);
                if (func_name == "length" || func_name == "len" || func_name == "strlen") {
                    std::cout << "✅ FOUND LENGTH FUNCTION: " << func_expr.function.name << " with " << func_expr.children.size() << " args" << std::endl;
                    found_length_function = true;
                    
                    for (size_t j = 0; j < func_expr.children.size(); j++) {
                        std::cout << "   ARG " << j << ": " << func_expr.children[j]->ToString() << std::endl;
                    }
                }
            }
            
            // Recursively check children expressions
            ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) {
                CheckExpression(child);
            });
        }
    };

    static void VortexLengthOptimizeFunction(OptimizerExtensionInput &input, duckdb::unique_ptr<LogicalOperator> &plan) {
        throw std::runtime_error("Something went wrong");
        std::cout << "🚀🚀🚀 OPTIMIZER FUNCTION CALLED! 🚀🚀🚀" << std::endl;
        std::cout << "🚀 OPTIMIZER: Vortex length optimization running!" << std::endl;
        
        if (!HasVortexScan(*plan)) {
            std::cout << "ℹ️  OPTIMIZER: No vortex_scan found in plan, skipping" << std::endl;
            return;
        }
        
        std::cout << "✅ OPTIMIZER: Found vortex_scan in plan!" << std::endl;
        
        // Use our visitor to detect length functions
        LengthFunctionVisitor visitor;
        visitor.VisitOperator(*plan);
        
        if (visitor.found_length_function) {
            std::cout << "🎯 OPTIMIZER: Found len() functions to potentially optimize!" << std::endl;
            // TODO: Implement actual transformation here
        } else {
            std::cout << "ℹ️  OPTIMIZER: No len() functions found to optimize" << std::endl;
        }
        
        std::cout << "✅ OPTIMIZER: Vortex length optimization completed!" << std::endl;
    }

    static void Register(DatabaseInstance &db) {
        std::cout << "🔧 REGISTER: Registering Vortex length optimizer extension..." << std::endl;
        
        try {
            auto &config = DBConfig::GetConfig(db);
            
            // Create the extension and ensure function pointer is set
            VortexLengthExtension extension;
            extension.optimize_function = VortexLengthOptimizeFunction;
            
            std::cout << "🔧 REGISTER: Function pointer: " << (void*)extension.optimize_function << std::endl;
            
            config.optimizer_extensions.push_back(std::move(extension));
            
            std::cout << "✅ SUCCESS: Vortex length optimizer extension registered!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "❌ EXCEPTION during registration: " << e.what() << std::endl;
            throw;
        }
    }
};

} // namespace vortex

// C API for registering the optimizer from Rust
extern "C" void duckdb_vx_register_optimizer(duckdb_database db_handle) {
    std::cout << "🔧 REGISTERING: Vortex optimizer extension..." << std::endl;
    
    if (!db_handle) {
        std::cout << "❌ ERROR: NULL database handle passed to optimizer registration" << std::endl;
        return;
    }

    try {
        auto db = reinterpret_cast<DuckDB *>(db_handle);
        vortex::VortexLengthExtension::Register(*db->instance);
        std::cout << "✅ SUCCESS: Vortex optimizer extension registered!" << std::endl;
    } catch (std::exception& e) {
        std::cout << "❌ EXCEPTION during optimizer registration: " << e.what() << std::endl;
    }
}