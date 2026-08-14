// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "data.hpp"
#include "error.hpp"
#include "expr.h"
#include "multi_file_reader.hpp"
#include "table_function.h"
#include "table_function.hpp"
#include "vortex.h"
#include "vortex_duckdb.h"

#include "duckdb.h"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/insertion_order_preserving_map.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

using namespace std::string_literals;
constexpr column_t COLUMN_IDENTIFIER_FILE_INDEX = MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX;
constexpr column_t COLUMN_IDENTIFIER_FILE_ROW_NUMBER = MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;

// This is a flaw of Duckdb API which doesn't allow passing non-const
// expressions. We never modify the value on Rust side.
static duckdb_vx_expr get_ffi_expr(const Expression &expr) {
    return reinterpret_cast<duckdb_vx_expr>(const_cast<Expression *>(&expr));
}

static void *get_ffi_bind(const FunctionData *bind_data) {
    return bind_data->Cast<MultiFileBindData>().bind_data->Cast<VortexBindData>().ffi_bind_data->DataPtr();
}

bool projection_expression_pushdown(ClientContext &, const TableFunctionProjectionExpressionInput &input) {
    duckdb_vx_expr ffi_expr = get_ffi_expr(input.expression);
    void *const ffi_bind = get_ffi_bind(input.get.bind_data.get());
    duckdb_vx_error error_out = nullptr;

    const bool ret = duckdb_table_function_pushdown_projection_expression( //
        ffi_bind,
        ffi_expr,
        input.projection_idx,
        &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }
    return ret;
}

extern "C" {
idx_t duckdb_vx_aggregate_len(duckdb_vx_agg_input ffi_input) {
    return reinterpret_cast<const TableFunctionUngroupedAggregateInput *>(ffi_input)->projections.size();
}

duckdb_vx_expr duckdb_vx_aggregate_at(duckdb_vx_agg_input ffi_input, idx_t i, idx_t *proj_idx) {
    const auto &input = *reinterpret_cast<const TableFunctionUngroupedAggregateInput *>(ffi_input);
    const auto &[scan_index, expr] = input.projections[i];
    *proj_idx = scan_index == COUNT_STAR_PROJ_IDX ? scan_index
                                                  : input.get.GetColumnIds()[scan_index].GetPrimaryIndex();
    return get_ffi_expr(expr);
}
}

bool aggregate_pushdown(ClientContext &, const TableFunctionUngroupedAggregateInput &input) {
    void *const ffi_bind = get_ffi_bind(input.get.bind_data.get());
    duckdb_vx_error error_out = nullptr;
    const auto ffi_input =
        reinterpret_cast<duckdb_vx_agg_input>(const_cast<TableFunctionUngroupedAggregateInput *>(&input));
    const bool res = duckdb_table_function_pushdown_projection_aggregates(ffi_bind, ffi_input, &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }
    return res;
}

using FilterVec = vector<unique_ptr<Expression>>;

void pushdown_complex_filter(const FunctionData &bind_data, FilterVec &filters) {
    void *const ffi_bind = get_ffi_bind(&bind_data);
    duckdb_vx_error error_out = nullptr;

    for (auto iter = filters.begin(); iter != filters.end();) {
        duckdb_vx_expr ffi_expr = reinterpret_cast<duckdb_vx_expr>(iter->get());

        const bool pushed = duckdb_table_function_pushdown_complex_filter(ffi_bind, ffi_expr, &error_out);
        if (error_out) {
            throw BinderException(IntoErrString(error_out));
        }
        iter = pushed ? filters.erase(iter) : std::next(iter);
    }
}

extern "C" void duckdb_vx_tfunc_bind_result_add_column(duckdb_bind_result ffi_result,
                                                       const char *name_str,
                                                       size_t name_len,
                                                       duckdb_logical_type ffi_type) {
    D_ASSERT(ffi_result);
    D_ASSERT(name_str);
    D_ASSERT(ffi_type);
    VortexBindResult &result = *reinterpret_cast<VortexBindResult *>(ffi_result);
    const LogicalType logical_type = *reinterpret_cast<LogicalType *>(ffi_type);

    result.names.emplace_back(name_str, name_len);
    result.return_types.emplace_back(logical_type);
}

extern "C" void duckdb_vx_string_map_insert(duckdb_vx_string_map map, const char *key, const char *value) {
    D_ASSERT(map);
    D_ASSERT(key);
    D_ASSERT(value);
    reinterpret_cast<InsertionOrderPreservingMap<string> *>(map)->insert(key, value);
}

InsertionOrderPreservingMap<string> to_string(TableFunctionToStringInput &input) {
    InsertionOrderPreservingMap<string> result;
    duckdb_vx_string_map ffi_map = reinterpret_cast<duckdb_vx_string_map>(&result);
    const void *const ffi_bind = get_ffi_bind(input.bind_data.get());
    duckdb_table_function_to_string(ffi_bind, ffi_map);
    return result;
}

bool is_vortex_scan(const TableFunction &function) {
    return function.bind == MultiFileFunction<VortexReaderInterface>::MultiFileBind;
}

unique_ptr<MultiFileReader> get_multi_file_reader(const TableFunction &) {
    return make_uniq<VortexMultiFileReader>();
}

duckdb_state register_table_function(DatabaseInstance &db, LogicalType parameter, const std::string &name) {
    MultiFileFunction<VortexReaderInterface> fn(name);
    fn.arguments[0] = parameter;
    // We neither support UNION BY NAME nor hive partitioning as for now
    fn.named_parameters = {};

    fn.filter_pushdown = true;
    fn.filter_prune = true;

    fn.pushdown_expression = [](auto &, const auto &, Expression &expression) {
        return duckdb_table_function_pushdown_expression(reinterpret_cast<duckdb_vx_expr>(&expression));
    };
    fn.pushdown_complex_filter = [](auto &, auto &, FunctionData *bind_data, FilterVec &filters) {
        pushdown_complex_filter(*bind_data, filters);
    };
    fn.to_string = to_string;

    fn.late_materialization = true;
    // Columns that uniquely identify a row for deferred re-fetch in a multi
    // file scan: (file index, row number in file).
    fn.get_row_id_columns = [](auto &, auto) -> vector<column_t> {
        return {COLUMN_IDENTIFIER_FILE_INDEX, COLUMN_IDENTIFIER_FILE_ROW_NUMBER};
    };

    fn.statistics = MultiFileFunction<VortexReaderInterface>::MultiFileScanStats;
    fn.get_multi_file_reader = get_multi_file_reader;

    try {
        auto &system_catalog = Catalog::GetSystemCatalog(db);
        auto data = CatalogTransaction::GetSystemTransaction(db);
        CreateTableFunctionInfo tf_info(fn);
        tf_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        system_catalog.CreateFunction(data, tf_info);
    } catch (const std::exception &e) {
        ErrorData data(e);
        DUCKDB_LOG_ERROR(db, "Failed to create Vortex table function:\t" + data.Message());
        return DuckDBError;
    }
    return DuckDBSuccess;
}

extern "C" duckdb_state duckdb_vx_register_table_functions(duckdb_database ffi_db) {
    D_ASSERT(ffi_db);
    const DatabaseWrapper &wrapper = *reinterpret_cast<DatabaseWrapper *>(ffi_db);
    DatabaseInstance &db = *wrapper.database->instance;

    for (LogicalType type : {LogicalType(LogicalType::VARCHAR), LogicalType::LIST(LogicalType::VARCHAR)}) {
        for (const std::string &name : {"read_vortex"s, "vortex_scan"s}) {
            if (register_table_function(db, type, name) == DuckDBError) {
                return DuckDBError;
            }
        }
    }
    return DuckDBSuccess;
}
