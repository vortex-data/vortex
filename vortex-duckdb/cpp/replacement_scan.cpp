// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx/duckdb_diagnostics.h"

DUCKDB_INCLUDES_BEGIN
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
DUCKDB_INCLUDES_END

#include "duckdb_vx.h"

namespace vortex {

static duckdb::unique_ptr<duckdb::TableRef>
VortexScanReplacement(duckdb::ClientContext &context,
                      duckdb::ReplacementScanInput &input,
                      duckdb::optional_ptr<duckdb::ReplacementScanData> /*data*/) {
    auto table_name = duckdb::ReplacementScan::GetFullPath(input);
    if (!duckdb::ReplacementScan::CanReplace(table_name, {"vortex"})) {
        return nullptr;
    }
    auto table_function = duckdb::make_uniq<duckdb::TableFunctionRef>();

    duckdb::vector<duckdb::unique_ptr<duckdb::ParsedExpression>> children(1);
    children[0] = duckdb::make_uniq<duckdb::ConstantExpression>(duckdb::Value(table_name));
    table_function->function =
        duckdb::make_uniq<duckdb::FunctionExpression>("read_vortex", std::move(children));

    if (!duckdb::FileSystem::HasGlob(table_name)) {
        auto &fs = duckdb::FileSystem::GetFileSystem(context);
        table_function->alias = fs.ExtractBaseName(table_name);
    }

    return table_function;
}

} // namespace vortex

extern "C" duckdb_state duckdb_vx_register_scan_replacement(duckdb_database duckdb_database) {
    if (!duckdb_database) {
        return DuckDBError;
    }

    auto wrapper = reinterpret_cast<duckdb::DatabaseWrapper *>(duckdb_database);
    if (!wrapper) {
        return DuckDBError;
    }

    auto &config = duckdb::DBConfig::GetConfig(*wrapper->database->instance);
    config.replacement_scans.emplace_back(vortex::VortexScanReplacement);

    return DuckDBSuccess;
}
