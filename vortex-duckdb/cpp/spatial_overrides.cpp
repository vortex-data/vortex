// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "spatial_overrides.hpp"

#include "expr.h"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/logging/logger.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/transaction/meta_transaction.hpp"

#include <algorithm>
#include <exception>

/// A spatial function Vortex shadows so that its filters can push into Vortex scans.
struct SpatialOverride {
    const char *name;
    idx_t arity;
    void (*tweak)(ScalarFunction &);
};

static constexpr SpatialOverride SPATIAL_OVERRIDES[] = {
    // Drop spatial's bind so the filter keeps the radius visible as `children[2]`.
    {"st_dwithin",
     3,
     [](ScalarFunction &fn) {
         fn.bind = nullptr;
     }},
    // Clear the error mode so the filter pushes through view projections.
    {"st_intersects",
     2,
     [](ScalarFunction &fn) {
         fn.SetErrorMode(FunctionErrors::CANNOT_ERROR);
     }},
};

/// Apply one override, later calls to the function bind to the pushable copy instead of
/// spatial's original.
static void RegisterSpatialOverride(ClientContext &context, const SpatialOverride &fn_override) {
    auto &system = Catalog::GetSystemCatalog(context);
    auto entry = system.GetEntry<ScalarFunctionCatalogEntry>(context,
                                                             DEFAULT_SCHEMA,
                                                             fn_override.name,
                                                             OnEntryNotFound::RETURN_NULL);
    if (!entry) {
        return;
    }
    ScalarFunctionSet set(fn_override.name);
    for (const auto &overload : entry->functions.functions) {
        ScalarFunction copy = overload;
        fn_override.tweak(copy);
        set.AddFunction(copy);
    }
    CreateScalarFunctionInfo info(std::move(set));
    info.on_conflict = OnCreateConflict::REPLACE_ON_CONFLICT;

    info.internal = false;
    // Register in the default database's catalog: unqualified calls resolve there before the
    // system catalog, so the copy shadows spatial's entry.
    auto &catalog = Catalog::GetCatalog(context, DatabaseManager::GetDefaultDatabase(context));
    // A file-backed catalog rejects writes unless the database is marked modified; the mark is
    // harmless here because function entries are never persisted to disk.
    MetaTransaction::Get(context).ModifyDatabase(catalog.GetAttached(), DatabaseModificationType());
    catalog.CreateFunction(context, info);
}

/// Apply every override in `SPATIAL_OVERRIDES` in one transaction.
extern "C" duckdb_state duckdb_vx_register_spatial_overrides(duckdb_database ffi_db) {
    if (!ffi_db) {
        return DuckDBError;
    }
    const DatabaseWrapper &wrapper = *reinterpret_cast<DatabaseWrapper *>(ffi_db);
    DatabaseInstance &db = *wrapper.database->instance;
    try {
        Connection conn(db);
        ClientContext &context = *conn.context;
        context.RunFunctionInTransaction([&]() {
            for (const auto &fn_override : SPATIAL_OVERRIDES) {
                RegisterSpatialOverride(context, fn_override);
            }
        });
    } catch (const std::exception &e) {
        ErrorData data(e);
        DUCKDB_LOG_ERROR(db, "Failed to register the spatial overrides:\t" + data.Message());
        return DuckDBError;
    }
    return DuckDBSuccess;
}

namespace {

// Rebinds overridden spatial calls in join conditions back to spatial's original, so spatial's
// own machinery handles joins. Filters are left untouched: they keep the override and push to
// Vortex scans.
class SpatialOverrideRestore final : public LogicalOperatorVisitor {
public:
    explicit SpatialOverrideRestore(ClientContext &context) : context(context) {
    }

    void VisitOperator(LogicalOperator &op) override {
        using enum LogicalOperatorType;
        switch (op.type) {
        case LOGICAL_COMPARISON_JOIN:
        case LOGICAL_ANY_JOIN:
        case LOGICAL_DELIM_JOIN:
        case LOGICAL_ASOF_JOIN:
            VisitOperatorExpressions(op);
            break;
        default:
            break;
        }
        VisitOperatorChildren(op);
    }

    unique_ptr<Expression> VisitReplace(BoundFunctionExpression &expr, unique_ptr<Expression> *) override {
        const bool overridden =
            std::any_of(std::begin(SPATIAL_OVERRIDES),
                        std::end(SPATIAL_OVERRIDES),
                        [&](const SpatialOverride &o) {
                            return expr.function.name == o.name && expr.children.size() == o.arity;
                        });
        if (!overridden) {
            return nullptr; // Not an overridden call: leave it as is.
        }
        // Spatial's original lives in the system catalog, where the override cannot shadow it.
        auto original = Catalog::GetSystemCatalog(context).GetEntry<ScalarFunctionCatalogEntry>(
            context,
            DEFAULT_SCHEMA,
            expr.function.name,
            OnEntryNotFound::RETURN_NULL);
        if (!original) {
            return nullptr;
        }
        // Rebind a copy of the call's arguments through the original function.
        vector<unique_ptr<Expression>> children;
        children.reserve(expr.children.size());
        for (const auto &child : expr.children) {
            children.push_back(child->Copy());
        }
        ErrorData error;
        FunctionBinder binder(context);
        auto bound = binder.BindScalarFunction(*original, std::move(children), error);
        if (!bound) {
            return nullptr; // No matching overload: the override call still executes, keep it.
        }
        return bound;
    }

private:
    ClientContext &context;
};

} // namespace

void RestoreSpatialOverrides(ClientContext &context, LogicalOperator &plan) {
    SpatialOverrideRestore(context).VisitOperator(plan);
}
