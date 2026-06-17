// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb_vx/optimizer.h"
#include "scalar_fn_pushdown.hpp"
#include "aggregate_fn_pushdown.hpp"
#include "duckdb_vx/table_function.h"
#include "vortex.h"

using namespace duckdb;

static void VortexOptimizeFunction(OptimizerExtensionInput &input, LogicalOperatorPtr &plan) {
    plan = TryPushdownScalarFunctions(input.context, std::move(plan));
    plan = TryPushdownAggregateFunctions(input.context, std::move(plan));
}

struct VortexOptimizerExtension final : OptimizerExtension {
    inline VortexOptimizerExtension() : OptimizerExtension(VortexOptimizeFunction, nullptr, {}) {
    }
};

extern "C" duckdb_state duckdb_vx_optimizer_extension_register(duckdb_database ffi_db) {
    D_ASSERT(ffi_db);
    const DatabaseWrapper &wrapper = *reinterpret_cast<DatabaseWrapper *>(ffi_db);
    DatabaseInstance &db = *wrapper.database->instance;
    try {
        DBConfig::GetConfig(db).GetCallbackManager().Register(VortexOptimizerExtension());
    } catch (const std::exception &e) {
        ErrorData data(e);
        DUCKDB_LOG_ERROR(db, "Failed to create Vortex optimizer extension:\t" + data.Message());
        return DuckDBError;
    }
    return DuckDBSuccess;
}
