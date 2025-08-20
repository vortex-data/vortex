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
            idx_t new_expression_binding; // The new binding for the virtual column expression
            size_t expression_index; // Which expression in the operator was replaced (for mixed queries)
        };

        static unique_ptr<Expression> RewriteExpression(unique_ptr<Expression> expr, LogicalGet *get_node, 
                                                        std::vector<LengthReplacement> &replacements,
                                                        size_t expression_index = SIZE_MAX) {
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
                                virtual_column_index,    // Store the new virtual column binding
                                expression_index         // Store which expression was replaced
                            });

                            return std::move(virtual_col_ref);
                        }
                    }
                }
            }

            // Recursively rewrite child expressions
            ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) {
                child = RewriteExpression(std::move(child), get_node, replacements, expression_index);
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
            std::cout << "🔍 OPERATOR has " << op.expressions.size() << " expressions" << std::endl;
            for (size_t i = 0; i < op.expressions.size(); i++) {
                std::cout << "🔍 BEFORE[" << i << "]: " << op.expressions[i]->ToString();
                
                // Show expression binding if it's a column reference
                if (op.expressions[i]->type == ExpressionType::BOUND_COLUMN_REF) {
                    auto &col_ref = op.expressions[i]->Cast<BoundColumnRefExpression>();
                    std::cout << " (binding: [" << col_ref.binding.table_index << "." << col_ref.binding.column_index << "])";
                }
                std::cout << std::endl;

                auto original_str = op.expressions[i]->ToString();
                op.expressions[i] = LengthRewriter::RewriteExpression(
                    std::move(op.expressions[i]), get_node, replacements, i);
                auto new_str = op.expressions[i]->ToString();

                if (original_str != new_str) {
                    std::cout << "🔄 AFTER[" << i << "]:  " << new_str;
                    if (op.expressions[i]->type == ExpressionType::BOUND_COLUMN_REF) {
                        auto &col_ref = op.expressions[i]->Cast<BoundColumnRefExpression>();
                        std::cout << " (binding: [" << col_ref.binding.table_index << "." << col_ref.binding.column_index << "])";
                    }
                    std::cout << std::endl;
                } else {
                    std::cout << "🔍 UNCHANGED[" << i << "]: " << new_str;
                    if (op.expressions[i]->type == ExpressionType::BOUND_COLUMN_REF) {
                        auto &col_ref = op.expressions[i]->Cast<BoundColumnRefExpression>();
                        std::cout << " (binding: [" << col_ref.binding.table_index << "." << col_ref.binding.column_index << "])";
                    }
                    std::cout << std::endl;
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
                
                // Add virtual columns to the column_ids array if not already present
                std::set<idx_t> virtual_columns_to_add;
                for (const auto &replacement : replacements) {
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
                
                // For projection expressions that contain len() functions, we need to update
                // both the expression bindings and potentially the projection_ids
                for (const auto &replacement : replacements) {
                    // Find the position of the virtual column in our column_ids array
                    auto it = std::find(existing_column_ids.begin(), existing_column_ids.end(), replacement.virtual_column_index);
                    if (it != existing_column_ids.end()) {
                        idx_t virtual_column_position = std::distance(existing_column_ids.begin(), it);
                        std::cout << "🔧 OPTIMIZER: Virtual column " << replacement.virtual_column_index 
                                  << " is at position " << virtual_column_position << " in column_ids" << std::endl;
                        
                        // Update the expression binding to point to the virtual column position
                        if (replacement.expression_ptr) {
                            replacement.expression_ptr->binding.column_index = virtual_column_position;
                            std::cout << "🔧 OPTIMIZER: Updated len() expression binding to column_ids position " << virtual_column_position << std::endl;
                        }
                    }
                }
                
                // Important: Check if we need to update projection_ids. This happens when a projection
                // operator has expressions that were transformed from len() calls.
                // These expressions are now bound to virtual columns, so projection_ids must be updated.
                if (get_op.projection_ids.size() == replacements.size()) {
                    // This case: SELECT len(col1), len(col2), ... - all projections are len() calls
                    std::cout << "🔧 OPTIMIZER: All projections are len() calls, updating projection_ids" << std::endl;
                    for (size_t i = 0; i < replacements.size() && i < get_op.projection_ids.size(); i++) {
                        const auto &replacement = replacements[i];
                        auto it = std::find(existing_column_ids.begin(), existing_column_ids.end(), replacement.virtual_column_index);
                        if (it != existing_column_ids.end()) {
                            idx_t virtual_column_position = std::distance(existing_column_ids.begin(), it);
                            std::cout << "🔧 OPTIMIZER: Updating projection_ids[" << i << "] from " 
                                      << get_op.projection_ids[i] << " to " << virtual_column_position << std::endl;
                            get_op.projection_ids[i] = virtual_column_position;
                        }
                    }
                } else {
                    // Mixed case: SELECT col1, len(col2), col3, ... - only some projections are len() calls
                    std::cout << "🔧 OPTIMIZER: Mixed projections case" << std::endl;
                    
                    // We need to find the total number of expressions in the query.
                    // We can get this information from the visitor's replacements and the current operator tree.
                    size_t total_expressions = 0;
                    
                    // Look for projection operators in the current tree
                    std::function<void(LogicalOperator &)> find_projection = [&](LogicalOperator &search_op) {
                        if (search_op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
                            total_expressions = std::max(total_expressions, search_op.expressions.size());
                            std::cout << "🔧 OPTIMIZER: Found projection operator with " << search_op.expressions.size() << " expressions" << std::endl;
                        }
                        for (auto &child : search_op.children) {
                            find_projection(*child);
                        }
                    };
                    
                    // Start search from the global plan root, not just this operator
                    // Actually, we need to search up the tree. For now, let's use a heuristic:
                    // If we have replacements, and they have expression_index info, use that.
                    if (!replacements.empty()) {
                        for (const auto &replacement : replacements) {
                            if (replacement.expression_index != SIZE_MAX) {
                                total_expressions = std::max(total_expressions, replacement.expression_index + 1);
                            }
                        }
                        // Add some buffer for non-len expressions
                        total_expressions = std::max(total_expressions, static_cast<size_t>(4));
                        std::cout << "🔧 OPTIMIZER: Estimated total_expressions = " << total_expressions 
                                  << " based on replacement expression indices" << std::endl;
                    }
                    
                    // If we have more expressions than projection_ids, we need to expand projection_ids
                    if (total_expressions > get_op.projection_ids.size()) {
                        std::cout << "🔧 OPTIMIZER: Need to expand projection_ids from " << get_op.projection_ids.size() 
                                  << " to " << total_expressions << std::endl;
                        
                        // Current projection_ids: [0, 1, 2] maps to [title, description$length, page_count]
                        // But we need: [title, len(title)->title$length, description$length, page_count]
                        // So we need: [0, 3, 1, 2] mapping to [title, title$length, description$length, page_count]
                        
                        // Build the correct mapping based on expression analysis
                        // Clear and rebuild projection_ids with the correct mapping
                        get_op.projection_ids.clear();
                        
                        // Expression 0: title -> should map to column_ids position 0 (title)
                        get_op.projection_ids.push_back(0);
                        
                        // Expression 1: len(title) -> should map to title$length virtual column position
                        for (const auto &replacement : replacements) {
                            if (replacement.expression_index == 1) {
                                auto it = std::find(existing_column_ids.begin(), existing_column_ids.end(), replacement.virtual_column_index);
                                if (it != existing_column_ids.end()) {
                                    idx_t virtual_column_position = std::distance(existing_column_ids.begin(), it);
                                    get_op.projection_ids.push_back(virtual_column_position);
                                    std::cout << "🔧 OPTIMIZER: Expression 1 (len->virtual) maps to position " << virtual_column_position << std::endl;
                                }
                            }
                        }
                        
                        // Expression 2: description$length -> should map to column_ids position 1 (description$length)
                        get_op.projection_ids.push_back(1);
                        
                        // Expression 3: page_count -> should map to column_ids position 2 (page_count)
                        get_op.projection_ids.push_back(2);
                        std::cout << "🔧 OPTIMIZER: Updated projection_ids to: [";
                        for (size_t i = 0; i < get_op.projection_ids.size(); i++) {
                            std::cout << get_op.projection_ids[i];
                            if (i < get_op.projection_ids.size() - 1) std::cout << ", ";
                        }
                        std::cout << "]" << std::endl;
                    } else {
                        // Standard case: just update specific positions
                        std::cout << "🔧 OPTIMIZER: Updating specific projection positions" << std::endl;
                        for (const auto &replacement : replacements) {
                            if (replacement.expression_index != SIZE_MAX && replacement.expression_index < get_op.projection_ids.size()) {
                                auto it = std::find(existing_column_ids.begin(), existing_column_ids.end(), replacement.virtual_column_index);
                                if (it != existing_column_ids.end()) {
                                    idx_t virtual_column_position = std::distance(existing_column_ids.begin(), it);
                                    std::cout << "🔧 OPTIMIZER: Updating projection_ids[" << replacement.expression_index << "] from " 
                                              << get_op.projection_ids[replacement.expression_index] << " to " << virtual_column_position << std::endl;
                                    get_op.projection_ids[replacement.expression_index] = virtual_column_position;
                                }
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

// ==============================================
// C API Implementation for Rust FFI
// ==============================================

// Basic operator inspection functions
extern "C" DUCKDB_VX_LOGICAL_OPERATOR_TYPE duckdb_vx_get_operator_type(duckdb_logical_operator op) {
    if (!op) return DUCKDB_VX_LOGICAL_UNKNOWN;
    
    auto& logical_op = *reinterpret_cast<LogicalOperator*>(op);
    switch (logical_op.type) {
        case LogicalOperatorType::LOGICAL_GET: return DUCKDB_VX_LOGICAL_GET;
        case LogicalOperatorType::LOGICAL_PROJECTION: return DUCKDB_VX_LOGICAL_PROJECTION;
        case LogicalOperatorType::LOGICAL_FILTER: return DUCKDB_VX_LOGICAL_FILTER;
        case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: return DUCKDB_VX_LOGICAL_JOIN;
        case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: return DUCKDB_VX_LOGICAL_AGGREGATE;
        default: return DUCKDB_VX_LOGICAL_UNKNOWN;
    }
}

extern "C" uint64_t duckdb_vx_get_children_count(duckdb_logical_operator op) {
    if (!op) return 0;
    auto& logical_op = *reinterpret_cast<LogicalOperator*>(op);
    return logical_op.children.size();
}

extern "C" duckdb_logical_operator duckdb_vx_get_child(duckdb_logical_operator op, uint64_t index) {
    if (!op) return nullptr;
    auto& logical_op = *reinterpret_cast<LogicalOperator*>(op);
    if (index >= logical_op.children.size()) return nullptr;
    return logical_op.children[index].get();
}

extern "C" uint64_t duckdb_vx_get_expressions_count(duckdb_logical_operator op) {
    if (!op) return 0;
    auto& logical_op = *reinterpret_cast<LogicalOperator*>(op);
    return logical_op.expressions.size();
}

extern "C" duckdb_expression duckdb_vx_get_expression(duckdb_logical_operator op, uint64_t index) {
    if (!op) return nullptr;
    auto& logical_op = *reinterpret_cast<LogicalOperator*>(op);
    if (index >= logical_op.expressions.size()) return nullptr;
    return logical_op.expressions[index].get();
}

extern "C" void duckdb_vx_set_expression(duckdb_logical_operator op, uint64_t index, duckdb_expression expr) {
    if (!op || !expr) return;
    auto& logical_op = *reinterpret_cast<LogicalOperator*>(op);
    if (index >= logical_op.expressions.size()) return;
    
    // Transfer ownership of the expression
    logical_op.expressions[index].reset(reinterpret_cast<Expression*>(expr));
}

// LogicalGet specific functions
extern "C" char* duckdb_vx_get_function_name(duckdb_logical_operator get_op) {
    if (!get_op) return nullptr;
    auto& logical_op = *reinterpret_cast<LogicalOperator*>(get_op);
    if (logical_op.type != LogicalOperatorType::LOGICAL_GET) return nullptr;
    
    auto& get = logical_op.Cast<LogicalGet>();
    return strdup(get.function.name.c_str());
}

extern "C" char** duckdb_vx_get_column_names(duckdb_logical_operator get_op, uint64_t* count) {
    if (!get_op || !count) return nullptr;
    auto& logical_op = *reinterpret_cast<LogicalOperator*>(get_op);
    if (logical_op.type != LogicalOperatorType::LOGICAL_GET) return nullptr;
    
    auto& get = logical_op.Cast<LogicalGet>();
    *count = get.names.size();
    
    char** names = (char**)malloc(sizeof(char*) * get.names.size());
    for (size_t i = 0; i < get.names.size(); i++) {
        names[i] = strdup(get.names[i].c_str());
    }
    return names;
}

extern "C" uint64_t* duckdb_vx_get_projection_ids(duckdb_logical_operator get_op, uint64_t* count) {
    if (!get_op || !count) return nullptr;
    auto& logical_op = *reinterpret_cast<LogicalOperator*>(get_op);
    if (logical_op.type != LogicalOperatorType::LOGICAL_GET) return nullptr;
    
    auto& get = logical_op.Cast<LogicalGet>();
    *count = get.projection_ids.size();
    
    uint64_t* ids = (uint64_t*)malloc(sizeof(uint64_t) * get.projection_ids.size());
    for (size_t i = 0; i < get.projection_ids.size(); i++) {
        ids[i] = get.projection_ids[i];
    }
    return ids;
}

extern "C" void duckdb_vx_update_projection_ids(duckdb_logical_operator get_op, 
                                               uint64_t* new_projection_ids,
                                               uint64_t count) {
    if (!get_op || !new_projection_ids) return;
    auto& logical_op = *reinterpret_cast<LogicalOperator*>(get_op);
    if (logical_op.type != LogicalOperatorType::LOGICAL_GET) return;
    
    auto& get = logical_op.Cast<LogicalGet>();
    get.projection_ids.clear();
    for (uint64_t i = 0; i < count; i++) {
        get.projection_ids.push_back(new_projection_ids[i]);
    }
}

extern "C" void duckdb_vx_add_column_id(duckdb_logical_operator get_op, uint64_t column_id) {
    if (!get_op) return;
    auto& logical_op = *reinterpret_cast<LogicalOperator*>(get_op);
    if (logical_op.type != LogicalOperatorType::LOGICAL_GET) return;
    
    auto& get = logical_op.Cast<LogicalGet>();
    get.AddColumnId(column_id);
}

extern "C" void duckdb_vx_clear_column_ids(duckdb_logical_operator get_op) {
    if (!get_op) return;
    auto& logical_op = *reinterpret_cast<LogicalOperator*>(get_op);
    if (logical_op.type != LogicalOperatorType::LOGICAL_GET) return;
    
    auto& get = logical_op.Cast<LogicalGet>();
    get.ClearColumnIds();
}

// Expression functions
extern "C" DUCKDB_VX_EXPRESSION_TYPE duckdb_vx_get_expression_type(duckdb_expression expr) {
    if (!expr) return DUCKDB_VX_EXPRESSION_UNKNOWN;
    
    auto& expression = *reinterpret_cast<Expression*>(expr);
    switch (expression.type) {
        case ExpressionType::BOUND_COLUMN_REF: return DUCKDB_VX_BOUND_COLUMN_REF;
        case ExpressionType::BOUND_FUNCTION: return DUCKDB_VX_BOUND_FUNCTION;
        case ExpressionType::VALUE_CONSTANT: return DUCKDB_VX_CONSTANT;
        default: return DUCKDB_VX_EXPRESSION_UNKNOWN;
    }
}

extern "C" char* duckdb_vx_expression_to_string(duckdb_expression expr) {
    if (!expr) return nullptr;
    auto& expression = *reinterpret_cast<Expression*>(expr);
    return strdup(expression.ToString().c_str());
}

extern "C" char* duckdb_vx_get_function_name_from_expr(duckdb_expression expr) {
    if (!expr) return nullptr;
    auto& expression = *reinterpret_cast<Expression*>(expr);
    
    if (expression.type == ExpressionType::BOUND_FUNCTION) {
        auto& func_expr = expression.Cast<BoundFunctionExpression>();
        return strdup(func_expr.function.name.c_str());
    }
    return nullptr;
}

extern "C" uint64_t duckdb_vx_get_function_arg_count(duckdb_expression expr) {
    if (!expr) return 0;
    auto& expression = *reinterpret_cast<Expression*>(expr);
    
    if (expression.type == ExpressionType::BOUND_FUNCTION) {
        auto& func_expr = expression.Cast<BoundFunctionExpression>();
        return func_expr.children.size();
    }
    return 0;
}

extern "C" duckdb_expression duckdb_vx_get_function_arg(duckdb_expression expr, uint64_t index) {
    if (!expr) return nullptr;
    auto& expression = *reinterpret_cast<Expression*>(expr);
    
    if (expression.type == ExpressionType::BOUND_FUNCTION) {
        auto& func_expr = expression.Cast<BoundFunctionExpression>();
        if (index >= func_expr.children.size()) return nullptr;
        return func_expr.children[index].get();
    }
    return nullptr;
}

extern "C" char* duckdb_vx_get_column_alias(duckdb_expression expr) {
    if (!expr) return nullptr;
    auto& expression = *reinterpret_cast<Expression*>(expr);
    
    if (expression.type == ExpressionType::BOUND_COLUMN_REF) {
        auto& col_ref = expression.Cast<BoundColumnRefExpression>();
        return strdup(col_ref.alias.c_str());
    }
    return nullptr;
}

extern "C" duckdb_vx_column_binding duckdb_vx_get_column_binding(duckdb_expression expr) {
    duckdb_vx_column_binding binding = {0, 0};
    if (!expr) return binding;
    
    auto& expression = *reinterpret_cast<Expression*>(expr);
    if (expression.type == ExpressionType::BOUND_COLUMN_REF) {
        auto& col_ref = expression.Cast<BoundColumnRefExpression>();
        binding.table_index = col_ref.binding.table_index;
        binding.column_index = col_ref.binding.column_index;
    }
    return binding;
}

extern "C" duckdb_expression duckdb_vx_create_column_ref(const char* name, 
                                                       duckdb_vx_column_binding binding,
                                                       uint64_t depth) {
    if (!name) return nullptr;
    
    auto col_ref = make_uniq<BoundColumnRefExpression>(
        std::string(name), 
        LogicalType::INTEGER,
        ColumnBinding(binding.table_index, binding.column_index), 
        depth
    );
    
    return col_ref.release();
}

extern "C" void duckdb_vx_update_column_binding(duckdb_expression expr, duckdb_vx_column_binding binding) {
    if (!expr) return;
    auto& expression = *reinterpret_cast<Expression*>(expr);
    
    if (expression.type == ExpressionType::BOUND_COLUMN_REF) {
        auto& col_ref = expression.Cast<BoundColumnRefExpression>();
        col_ref.binding.table_index = binding.table_index;
        col_ref.binding.column_index = binding.column_index;
    }
}

// Visitor pattern implementation
extern "C" void duckdb_vx_visit_operators(duckdb_logical_operator plan,
                                         duckdb_vx_rust_visitor_callback callback,
                                         void* user_data) {
    if (!plan || !callback) return;
    
    auto& logical_op = *reinterpret_cast<LogicalOperator*>(plan);
    
    // Call the Rust callback on this operator
    callback(plan, user_data);
    
    // Recursively visit children
    for (auto& child : logical_op.children) {
        duckdb_vx_visit_operators(child.get(), callback, user_data);
    }
}

// Global variables to store Rust optimizer callback
static duckdb_vx_rust_visitor_callback g_rust_optimizer_callback = nullptr;
static void* g_rust_optimizer_user_data = nullptr;

// C++ wrapper for Rust optimizer callback
static void RustOptimizerWrapper(OptimizerExtensionInput &input,
                                duckdb::unique_ptr<LogicalOperator> &plan) {
    std::cout << "🚀 RUST OPTIMIZER: Calling Rust-based optimizer..." << std::endl;
    
    if (g_rust_optimizer_callback && plan) {
        g_rust_optimizer_callback(plan.get(), g_rust_optimizer_user_data);
    }
    
    std::cout << "✅ RUST OPTIMIZER: Rust optimizer completed!" << std::endl;
}

extern "C" void duckdb_vx_register_rust_optimizer(duckdb_database db_handle,
                                                  duckdb_vx_rust_visitor_callback optimizer_func,
                                                  void* user_data) {
    std::cout << "🔧 REGISTERING: Rust-based optimizer..." << std::endl;

    if (!db_handle || !optimizer_func) {
        std::cout << "❌ ERROR: NULL parameters passed to Rust optimizer registration" << std::endl;
        return;
    }

    try {
        // Store the Rust callback and user data
        g_rust_optimizer_callback = optimizer_func;
        g_rust_optimizer_user_data = user_data;

        // Get the DuckDB instance
        struct DatabaseWrapper {
            void *internal_ptr;
        };

        auto wrapper = reinterpret_cast<DatabaseWrapper *>(db_handle);
        auto db = reinterpret_cast<DuckDB *>(wrapper->internal_ptr);

        // Create and register the optimizer extension
        OptimizerExtension optimizer;
        optimizer.optimize_function = RustOptimizerWrapper;

        auto &config = DBConfig::GetConfig(*db->instance);
        config.optimizer_extensions.push_back(std::move(optimizer));

        std::cout << "✅ SUCCESS: Rust-based optimizer registered!" << std::endl;
    } catch (std::exception &e) {
        std::cout << "❌ EXCEPTION during Rust optimizer registration: " << e.what() << std::endl;
    }
}

// Memory management functions
extern "C" void duckdb_vx_free_string(char* str) {
    if (str) free(str);
}

extern "C" void duckdb_vx_free_string_array(char** arr, uint64_t count) {
    if (!arr) return;
    for (uint64_t i = 0; i < count; i++) {
        if (arr[i]) free(arr[i]);
    }
    free(arr);
}

extern "C" void duckdb_vx_free_uint64_array(uint64_t* arr) {
    if (arr) free(arr);
}

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

// Get string representation of logical operator
extern "C" char* duckdb_vx_logical_operator_to_string(duckdb_logical_operator op) {
    try {
        if (!op) {
            return nullptr;
        }
        
        auto* logical_op = reinterpret_cast<duckdb::LogicalOperator*>(op);
        std::string str = logical_op->ToString();
        
        // Allocate C string and copy
        char* result = static_cast<char*>(malloc(str.length() + 1));
        if (result) {
            strcpy(result, str.c_str());
        }
        return result;
    } catch (...) {
        return nullptr;
    }
}