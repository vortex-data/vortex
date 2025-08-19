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

    // Helper class to rewrite len() function calls to virtual column references
    class LengthRewriter {
    public:
        static unique_ptr<Expression> RewriteExpression(unique_ptr<Expression> expr, LogicalOperator &op) {
            if (expr->type == ExpressionType::BOUND_FUNCTION) {
                auto &func_expr = expr->Cast<BoundFunctionExpression>();
                auto func_name = StringUtil::Lower(func_expr.function.name);
                
                // Check if it's a length function with one argument
                if ((func_name == "length" || func_name == "len" || func_name == "strlen") && 
                    func_expr.children.size() == 1) {
                    
                    auto &arg = func_expr.children[0];
                    
                    // Check if the argument is a column reference
                    if (arg->type == ExpressionType::BOUND_COLUMN_REF) {
                        auto &col_ref = arg->Cast<BoundColumnRefExpression>();
                        
                        // Create new column name with $length suffix
                        std::string virtual_col_name = col_ref.alias + "$length";
                        
                        std::cout << "🔄 OPTIMIZER: Rewriting " << func_expr.function.name 
                                  << "(" << col_ref.alias << ") → " << virtual_col_name << std::endl;
                        
                        // Try to find the LogicalGet node to access table schema
                        LogicalGet* get_node = nullptr;
                        
                        // Search current operator first
                        if (op.type == LogicalOperatorType::LOGICAL_GET) {
                            get_node = &op.Cast<LogicalGet>();
                        }
                        
                        // If not found in current op, search children
                        if (!get_node) {
                            std::function<void(LogicalOperator&)> find_get = [&](LogicalOperator &search_op) {
                                if (!get_node && search_op.type == LogicalOperatorType::LOGICAL_GET) {
                                    get_node = &search_op.Cast<LogicalGet>();
                                    return;
                                }
                                for (auto &child : search_op.children) {
                                    find_get(*child);
                                }
                            };
                            find_get(op);
                        }
                        
                        std::cout << "🔍 OPTIMIZER: Looking for virtual column '" << virtual_col_name 
                                  << "' in table " << col_ref.binding.table_index << std::endl;
                        
                        auto virtual_column_index = col_ref.binding.column_index;
                        
                        if (get_node) {
                            std::cout << "📊 OPTIMIZER: Found LogicalGet with " << get_node->names.size() << " columns" << std::endl;
                            
                            // Look for the virtual column in the bound column names
                            for (size_t col_idx = 0; col_idx <= get_node->names.size(); col_idx++) {
                                std::cout << "   Column " << col_idx << ": " << get_node->names[col_idx] << std::endl;
                                if (get_node->names[col_idx] == virtual_col_name) {
                                    virtual_column_index = col_idx;
                                    std::cout << "✅ OPTIMIZER: Found virtual column '" << virtual_col_name 
                                              << "' at index " << virtual_column_index << std::endl;
                                    break;
                                }
                            }
                        } else {
                            std::cout << "⚠️  OPTIMIZER: No LogicalGet found, using fallback calculation" << std::endl;
                            // Fallback: assume 2 real columns, virtual columns start at index 2
                            virtual_column_index = 2 + col_ref.binding.column_index;
                        }
                        
                        std::cout << "🔢 OPTIMIZER: Mapping " << col_ref.alias << " (idx:" << col_ref.binding.column_index 
                                  << ") → " << virtual_col_name << " (idx:" << virtual_column_index << ")" << std::endl;
                        
                        auto virtual_col_ref = make_uniq<BoundColumnRefExpression>(
                            virtual_col_name,
                            LogicalType::INTEGER,
                            ColumnBinding(col_ref.binding.table_index, virtual_column_index),
                            col_ref.depth
                        );
                        
                        return std::move(virtual_col_ref);
                    }
                }
            }
            
            // Recursively rewrite child expressions
            ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) {
                child = RewriteExpression(std::move(child), op);
            });
            
            return expr;
        }
    };

    // Visitor that applies length function rewriting to all expressions
    class VortexOptimizerVisitor : public LogicalOperatorVisitor {
    public:
        bool made_changes = false;
        OptimizerExtensionInput* optimizer_input = nullptr; // Access to optimizer context
        
        VortexOptimizerVisitor(OptimizerExtensionInput* input) : optimizer_input(input) {}

        void VisitOperator(LogicalOperator &op) override {
            std::cout << "🔍 VISITING: Operator type: " << (int)op.type << std::endl;

            // Rewrite all expressions in this operator
            for (size_t i = 0; i < op.expressions.size(); i++) {
                std::cout << "🔍 BEFORE: " << op.expressions[i]->ToString() << std::endl;
                
                auto original_str = op.expressions[i]->ToString();
                op.expressions[i] = LengthRewriter::RewriteExpression(std::move(op.expressions[i]), op);
                auto new_str = op.expressions[i]->ToString();
                
                if (original_str != new_str) {
                    made_changes = true;
                    std::cout << "🔄 AFTER:  " << new_str << std::endl;
                } else {
                    std::cout << "🔍 UNCHANGED: " << new_str << std::endl;
                }
            }

            // Visit children
            VisitOperatorChildren(op);
        }
    };

    static void VortexLengthOptimizeFunction(OptimizerExtensionInput &input,
                                             duckdb::unique_ptr<LogicalOperator> &plan) {
        std::cout << "🚀🚀🚀 OPTIMIZER FUNCTION CALLED! 🚀🚀🚀" << std::endl;
        std::cout << "🚀 OPTIMIZER: Vortex length optimization running!" << std::endl;

        if (!HasVortexScan(*plan)) {
            std::cout << "ℹ️  OPTIMIZER: No vortex_scan found in plan, skipping" << std::endl;
            return;
        }

        std::cout << "✅ OPTIMIZER: Found vortex_scan in plan!" << std::endl;

        // Apply length function rewriting
        VortexOptimizerVisitor visitor(&input);
        visitor.VisitOperator(*plan);

        if (visitor.made_changes) {
            std::cout << "🎯 OPTIMIZER: Successfully applied len() → virtual column transformations!" << std::endl;
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

            OptimizerExtension optimizer;
            optimizer.optimize_function = VortexLengthOptimizeFunction;

            std::cout << "🔧 REGISTER: Function pointer: " << (void *)optimizer.optimize_function
                      << std::endl;

            std::cout << "optimizer_extensions len: " << std::to_string(config.optimizer_extensions.size())
                      << std::endl;

            config.optimizer_extensions.push_back(std::move(optimizer));

            std::cout << "✅ SUCCESS: Vortex length optimizer extension registered!" << std::endl;
        } catch (std::exception &e) {
            std::cout << "❌ EXCEPTION during registration: " << e.what() << std::endl;
            throw e;
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
        // The duckdb_database is a pointer to _duckdb_database struct which has internal_ptr
        // The internal_ptr contains the actual DuckDB object
        struct DatabaseWrapper {
            void *internal_ptr;
        };
        
        auto wrapper = reinterpret_cast<DatabaseWrapper *>(db_handle);
        auto db = reinterpret_cast<DuckDB *>(wrapper->internal_ptr);
        
        // Test if we got the right object
        std::cout << "🔍 DB threads: " << std::to_string(db->NumberOfThreads()) << std::endl;
        
        vortex::VortexLengthExtension::Register(*db->instance);
        std::cout << "✅ SUCCESS: Vortex optimizer extension registered!" << std::endl;
    } catch (std::exception &e) {
        std::cout << "❌ EXCEPTION during optimizer registration: " << e.what() << std::endl;
    }
}