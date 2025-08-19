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
#include <set>
#include <algorithm>

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

    // Helper class to find and replace len() function calls with virtual column references
    class LengthRewriter {
    public:
        struct LengthReplacement {
            idx_t original_column_binding;  // The binding index of the original column
            idx_t virtual_column_index;     // The index of the virtual column to use instead
            std::string virtual_col_name;   // Name of the virtual column
            BoundColumnRefExpression *expression_ptr; // Pointer to the expression to update later
            idx_t original_expression_binding; // The original binding of the len() expression
        };

        static unique_ptr<Expression> RewriteExpression(unique_ptr<Expression> expr, LogicalGet *get_node, 
                                                        std::vector<LengthReplacement> &replacements) {
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

                        // Create virtual column name
                        std::string virtual_col_name = col_ref.alias + "$length";

                        std::cout << "🔄 OPTIMIZER: Found " << func_expr.function.name << "("
                                  << col_ref.alias << ") → " << virtual_col_name << std::endl;

                        // Find the virtual column index in the table schema
                        idx_t virtual_column_index = DConstants::INVALID_INDEX;
                        if (get_node) {
                            for (size_t col_idx = 0; col_idx < get_node->names.size(); col_idx++) {
                                if (get_node->names[col_idx] == virtual_col_name) {
                                    virtual_column_index = col_idx;
                                    std::cout << "✅ OPTIMIZER: Found virtual column '" << virtual_col_name
                                              << "' at index " << virtual_column_index << std::endl;
                                    break;
                                }
                            }
                        }

                        if (virtual_column_index != DConstants::INVALID_INDEX) {
                            // Create a column reference with a temporary binding (will be updated later)
                            auto virtual_col_ref = make_uniq<BoundColumnRefExpression>(
                                virtual_col_name, LogicalType::INTEGER,
                                ColumnBinding(col_ref.binding.table_index, virtual_column_index), 
                                col_ref.depth);

                            // Record this replacement for later projection mapping and binding update
                            replacements.push_back({
                                col_ref.binding.column_index,
                                virtual_column_index,
                                virtual_col_name,
                                virtual_col_ref.get(),  // Store pointer to update binding later
                                col_ref.binding.column_index  // Store original len() expression binding
                            });

                            return std::move(virtual_col_ref);
                        }
                    }
                }
            }

            // Recursively rewrite child expressions
            ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) {
                child = RewriteExpression(std::move(child), get_node, replacements);
            });

            return expr;
        }
    };

    // Visitor that applies length function rewriting to all expressions
    class VortexOptimizerVisitor : public LogicalOperatorVisitor {
    public:
        std::vector<LengthRewriter::LengthReplacement> replacements;

        void VisitOperator(LogicalOperator &op) override {
            std::cout << "🔍 VISITING: Operator type: " << (int)op.type << std::endl;

            // Find the LogicalGet node for schema information
            LogicalGet *get_node = nullptr;
            if (op.type == LogicalOperatorType::LOGICAL_GET) {
                get_node = &op.Cast<LogicalGet>();
                if (get_node->function.name != "vortex_scan") {
                    get_node = nullptr; // Only process vortex_scan tables
                }
            }

            // Search for LogicalGet in children if not found
            if (!get_node) {
                std::function<void(LogicalOperator &)> find_vortex_scan = [&](LogicalOperator &search_op) {
                    if (!get_node && search_op.type == LogicalOperatorType::LOGICAL_GET) {
                        auto &search_get = search_op.Cast<LogicalGet>();
                        if (search_get.function.name == "vortex_scan") {
                            get_node = &search_get;
                        }
                    }
                    for (auto &child : search_op.children) {
                        find_vortex_scan(*child);
                    }
                };
                find_vortex_scan(op);
            }

            // Rewrite expressions in this operator
            for (size_t i = 0; i < op.expressions.size(); i++) {
                std::cout << "🔍 BEFORE: " << op.expressions[i]->ToString() << std::endl;

                auto original_str = op.expressions[i]->ToString();
                op.expressions[i] = LengthRewriter::RewriteExpression(
                    std::move(op.expressions[i]), get_node, replacements);
                auto new_str = op.expressions[i]->ToString();

                if (original_str != new_str) {
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
        VortexOptimizerVisitor visitor;
        visitor.VisitOperator(*plan);

        if (!visitor.replacements.empty()) {
            std::cout << "🎯 OPTIMIZER: Found " << visitor.replacements.size() 
                      << " len() → virtual column transformations!" << std::endl;
            
            // Update the projection_ids in vortex_scan LogicalGet nodes
            UpdateVortexScanProjections(*plan, visitor.replacements);
        } else {
            std::cout << "ℹ️  OPTIMIZER: No len() functions found to optimize" << std::endl;
        }

        std::cout << "Plan: " << plan->ToString() << std::endl;
        std::cout << "✅ OPTIMIZER: Vortex length optimization completed!" << std::endl;
    }
    
    static void UpdateVortexScanProjections(LogicalOperator &op, 
                                             const std::vector<LengthRewriter::LengthReplacement> &replacements) {
        // If this is a vortex_scan, update its projection_ids to map bound columns to virtual columns
        if (op.type == LogicalOperatorType::LOGICAL_GET) {
            auto &get_op = op.Cast<LogicalGet>();
            if (get_op.function.name == "vortex_scan") {
                std::cout << "🔧 OPTIMIZER: ===== BEFORE TRANSFORM =====" << std::endl;
                std::cout << "🔧 OPTIMIZER: Current projection_ids: [";
                for (size_t i = 0; i < get_op.projection_ids.size(); i++) {
                    std::cout << get_op.projection_ids[i];
                    if (i < get_op.projection_ids.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                
                auto current_column_ids = get_op.GetColumnIds();
                std::cout << "🔧 OPTIMIZER: Current column_ids: [";
                for (size_t i = 0; i < current_column_ids.size(); i++) {
                    std::cout << current_column_ids[i].GetPrimaryIndex();
                    if (i < current_column_ids.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                
                std::cout << "🔧 OPTIMIZER: Current names: [";
                for (size_t i = 0; i < get_op.names.size(); i++) {
                    std::cout << "\"" << get_op.names[i] << "\"";
                    if (i < get_op.names.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                
                std::cout << "🔧 OPTIMIZER: Current returned_types size: " << get_op.returned_types.size() << std::endl;
                
                // Get current column_ids as a vector to work with
                std::vector<idx_t> existing_column_ids;
                for (const auto &col_id : current_column_ids) {
                    existing_column_ids.push_back(col_id.GetPrimaryIndex());
                }
                
                // Find the maximum column index to determine where to add virtual columns
                idx_t max_column_id = 0;
                if (!existing_column_ids.empty()) {
                    max_column_id = *std::max_element(existing_column_ids.begin(), existing_column_ids.end());
                }
                
                // Add virtual columns to the column_ids array if not already present
                std::set<idx_t> virtual_columns_to_add;
                for (const auto &replacement : replacements) {
                    // Ensure source column is in column_ids
                    if (std::find(existing_column_ids.begin(), existing_column_ids.end(), replacement.original_column_binding) == existing_column_ids.end()) {
                        existing_column_ids.push_back(replacement.original_column_binding);
                        std::cout << "🔧 OPTIMIZER: Added missing source column " << replacement.original_column_binding << std::endl;
                    }
                    
                    // Add virtual column if not already present
                    if (std::find(existing_column_ids.begin(), existing_column_ids.end(), replacement.virtual_column_index) == existing_column_ids.end()) {
                        virtual_columns_to_add.insert(replacement.virtual_column_index);
                    }
                }
                
                // Add virtual columns to the end of existing_column_ids
                for (idx_t virtual_col_id : virtual_columns_to_add) {
                    existing_column_ids.push_back(virtual_col_id);
                    std::cout << "🔧 OPTIMIZER: Added virtual column " << virtual_col_id << " to column_ids" << std::endl;
                }
                
                // Rebuild column_ids with the expanded list
                get_op.ClearColumnIds();
                for (idx_t col_id : existing_column_ids) {
                    get_op.AddColumnId(col_id);
                }
                
                // Replace original column projections with virtual column projections
                for (const auto &replacement : replacements) {
                    // Find the position of the virtual column in our column_ids array
                    auto it = std::find(existing_column_ids.begin(), existing_column_ids.end(), replacement.virtual_column_index);
                    if (it != existing_column_ids.end()) {
                        idx_t virtual_column_position = std::distance(existing_column_ids.begin(), it);
                        std::cout << "🔧 OPTIMIZER: Virtual column " << replacement.virtual_column_index 
                                  << " is at position " << virtual_column_position << " in column_ids" << std::endl;
                        
                        // Find projection positions that reference the original source column and replace them
                        bool found_replacement = false;
                        for (size_t proj_idx = 0; proj_idx < get_op.projection_ids.size(); proj_idx++) {
                            idx_t current_column_pos = get_op.projection_ids[proj_idx];
                            if (current_column_pos < existing_column_ids.size()) {
                                idx_t current_column_id = existing_column_ids[current_column_pos];
                                // If this projection position refers to the source column that was transformed
                                if (current_column_id == replacement.original_column_binding) {
                                    // Replace this projection position with the virtual column position
                                    std::cout << "🔧 OPTIMIZER: Replacing projection_ids[" << proj_idx 
                                              << "] from " << current_column_pos << " (col " << current_column_id 
                                              << ") to " << virtual_column_position 
                                              << " (virtual col " << replacement.virtual_column_index << ")" << std::endl;
                                    get_op.projection_ids[proj_idx] = virtual_column_position;
                                    
                                    // Update the expression binding to point to the virtual column position
                                    if (replacement.expression_ptr) {
                                        replacement.expression_ptr->binding.column_index = virtual_column_position;
                                        std::cout << "🔧 OPTIMIZER: Updated expression binding to column_ids position " << virtual_column_position << std::endl;
                                    }
                                    found_replacement = true;
                                    break; // Only replace the first matching position
                                }
                            }
                        }
                        
                        // If we didn't find a source column to replace, add the virtual column as a new projection
                        if (!found_replacement) {
                            idx_t projection_index = get_op.projection_ids.size();
                            get_op.projection_ids.push_back(virtual_column_position);
                            std::cout << "🔧 OPTIMIZER: Added virtual column position " << virtual_column_position 
                                      << " to projection_ids at index " << projection_index << std::endl;
                            
                            if (replacement.expression_ptr) {
                                replacement.expression_ptr->binding.column_index = virtual_column_position;
                                std::cout << "🔧 OPTIMIZER: Updated expression binding to column_ids position " << virtual_column_position << std::endl;
                            }
                        }
                    }
                }
                
                std::cout << "🔧 OPTIMIZER: ===== AFTER TRANSFORM =====" << std::endl;
                auto final_column_ids = get_op.GetColumnIds();
                std::cout << "🔧 OPTIMIZER: Final column_ids: [";
                for (size_t i = 0; i < final_column_ids.size(); i++) {
                    std::cout << final_column_ids[i].GetPrimaryIndex();
                    if (i < final_column_ids.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                
                std::cout << "🔧 OPTIMIZER: Final projection_ids: [";
                for (size_t i = 0; i < get_op.projection_ids.size(); i++) {
                    std::cout << get_op.projection_ids[i];
                    if (i < get_op.projection_ids.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                
                std::cout << "🔧 OPTIMIZER: Final names: [";
                for (size_t i = 0; i < get_op.names.size(); i++) {
                    std::cout << "\"" << get_op.names[i] << "\"";
                    if (i < get_op.names.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                
                std::cout << "🔧 OPTIMIZER: Final returned_types size: " << get_op.returned_types.size() << std::endl;
            }
        }
        
        // Recursively update children
        for (auto &child : op.children) {
            UpdateVortexScanProjections(*child, replacements);
        }
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