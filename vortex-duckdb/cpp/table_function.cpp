// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx/data.hpp"
#include "duckdb_vx/error.hpp"
#include "duckdb_vx/table_function.h"
#include "vortex.h"

#include "duckdb.h"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/insertion_order_preserving_map.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"

using namespace std::string_literals;
using namespace duckdb;
using vortex::CData;
using vortex::IntoErrString;
constexpr column_t COLUMN_IDENTIFIER_FILE_INDEX = MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX;
constexpr column_t COLUMN_IDENTIFIER_FILE_ROW_NUMBER = MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;

struct CTableBindData final : FunctionData {
    CTableBindData(unique_ptr<CData> ffi_data_p, const vector<LogicalType> &types)
        : ffi_data(std::move(ffi_data_p)), types(types) {
    }
    unique_ptr<FunctionData> Copy() const override;
    bool Equals(const FunctionData &other_base) const override;

    unique_ptr<CData> ffi_data;
    vector<LogicalType> types;
};

unique_ptr<FunctionData> CTableBindData::Copy() const {
    duckdb_vx_error error_out = nullptr;
    const auto copied_ffi_data = duckdb_table_function_bind_data_clone(ffi_data->DataPtr(), &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }

    auto ffi_data_p = unique_ptr<CData>(reinterpret_cast<CData *>(copied_ffi_data));
    return make_uniq<CTableBindData>(std::move(ffi_data_p), types);
}

bool CTableBindData::Equals(const FunctionData &other_base) const {
    const CTableBindData &other = other_base.Cast<CTableBindData>();
    // if "types" are different, "ffi_data" would also be different as it
    // contains types inside, so omit "types" from comparison.
    return ffi_data.get() == other.ffi_data.get();
}

struct CTableGlobalData final : GlobalTableFunctionState {
    explicit CTableGlobalData(unique_ptr<CData> ffi_data) : ffi_data(std::move(ffi_data)) {
    }

    idx_t MaxThreads() const override {
        return GlobalTableFunctionState::MAX_THREADS;
    }

    unique_ptr<CData> ffi_data;
};

struct CTableLocalData final : LocalTableFunctionState {
    explicit CTableLocalData(unique_ptr<CData> ffi_data) : ffi_data(std::move(ffi_data)) {
    }

    unique_ptr<CData> ffi_data;
};

double
table_scan_progress(ClientContext &, const FunctionData *, const GlobalTableFunctionState *global_state) {
    void *const c_global_state = global_state->Cast<CTableGlobalData>().ffi_data->DataPtr();
    return duckdb_table_function_scan_progress(c_global_state);
}

static Value &UnwrapValue(duckdb_value value) {
    return *(reinterpret_cast<Value *>(value));
}

unique_ptr<BaseStatistics> numeric_stats(duckdb_column_statistics &stats, LogicalType type) {
    BaseStatistics out = StringStats::CreateUnknown(type);
    if (stats.min) {
        NumericStats::SetMin(out, UnwrapValue(stats.min));
        duckdb_destroy_value(&stats.min);
    }
    if (stats.max) {
        NumericStats::SetMax(out, UnwrapValue(stats.max));
        duckdb_destroy_value(&stats.max);
    }
    if (!stats.has_null) {
        out.Set(StatsInfo::CANNOT_HAVE_NULL_VALUES);
    }
    return out.ToUnique();
}

unique_ptr<BaseStatistics> string_stats(duckdb_column_statistics &stats, LogicalType type) {
    BaseStatistics out = StringStats::CreateUnknown(type);
    if (stats.min) {
        StringStats::SetMin(out, StringValue::Get(UnwrapValue(stats.min)));
        duckdb_destroy_value(&stats.min);
    }
    if (stats.max) {
        StringStats::SetMax(out, StringValue::Get(UnwrapValue(stats.max)));
        duckdb_destroy_value(&stats.max);
    }
    if (stats.max_string_length >> 63) {
        StringStats::SetMaxStringLength(out, uint32_t(stats.max_string_length));
    }
    if (!stats.has_null) {
        out.Set(StatsInfo::CANNOT_HAVE_NULL_VALUES);
    }

    return out.ToUnique();
}

unique_ptr<BaseStatistics> base_stats(duckdb_column_statistics &stats, LogicalType type) {
    BaseStatistics out = StringStats::CreateUnknown(type);
    if (!stats.has_null) {
        out.Set(StatsInfo::CANNOT_HAVE_NULL_VALUES);
    }
    return out.ToUnique();
}

unique_ptr<BaseStatistics> statistics(ClientContext &, const FunctionData *bind_data, column_t column_index) {
    if (IsVirtualColumn(column_index)) {
        return {};
    }

    const auto &bind = bind_data->Cast<CTableBindData>();
    void *const ffi_bind = bind.ffi_data->DataPtr();

    duckdb_column_statistics statistics = {};
    if (!duckdb_table_function_statistics(ffi_bind, column_index, &statistics)) {
        return {};
    }

    const LogicalType type = bind.types[column_index];

    switch (type.id()) {
    case LogicalTypeId::BOOLEAN:
    case LogicalTypeId::TINYINT:
    case LogicalTypeId::SMALLINT:
    case LogicalTypeId::INTEGER:
    case LogicalTypeId::BIGINT:
    case LogicalTypeId::FLOAT:
    case LogicalTypeId::DOUBLE:
    case LogicalTypeId::UTINYINT:
    case LogicalTypeId::USMALLINT:
    case LogicalTypeId::UINTEGER:
    case LogicalTypeId::UBIGINT:
    case LogicalTypeId::UHUGEINT:
    case LogicalTypeId::HUGEINT: {
        return numeric_stats(statistics, type);
    }
    case LogicalTypeId::VARCHAR:
    case LogicalTypeId::BLOB: {
        return string_stats(statistics, type);
    }
    case LogicalTypeId::STRUCT: {
        // TODO(myrrc)
        // Duckdb's has_null has a different semantics for structs.
        // If we propagate our has_null, this breaks Duckdb optimizer.
        // You can reproduce it in struct.slt test in vortex-sqllogictests:
        return {};
    }
    default:
        return base_stats(statistics, type);
    }
}

struct CTableBindResult {
    vector<LogicalType> &return_types;
    vector<string> &names;
};

/**
 * Called for every new query. For example, if there is a VIEW over *.vortex,
 * and after a query another file is added matching the glob, for second query
 * bind() will be called again.
 */
unique_ptr<FunctionData> c_bind(ClientContext &context,
                                TableFunctionBindInput &input,
                                vector<LogicalType> &return_types,
                                vector<string> &names) {
    CTableBindResult result = {return_types, names};

    duckdb_vx_error error_out = nullptr;
    auto ctx = reinterpret_cast<duckdb_client_context>(&context);
    auto ffi_bind_data = duckdb_table_function_bind(ctx,
                                                    reinterpret_cast<duckdb_vx_tfunc_bind_input>(&input),
                                                    reinterpret_cast<duckdb_vx_tfunc_bind_result>(&result),
                                                    &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }

    auto cdata = unique_ptr<CData>(reinterpret_cast<CData *>(ffi_bind_data));
    return make_uniq<CTableBindData>(std::move(cdata), return_types);
}

unique_ptr<GlobalTableFunctionState> c_init_global(ClientContext &context, TableFunctionInitInput &input) {
    const CTableBindData &bind = input.bind_data->Cast<CTableBindData>();

    duckdb_vx_tfunc_init_input ffi_input = {
        .bind_data = bind.ffi_data->DataPtr(),
        .column_ids = input.column_ids.data(),
        .column_ids_count = input.column_ids.size(),
        .projection_ids = input.projection_ids.data(),
        .projection_ids_count = input.projection_ids.size(),
        .filters = reinterpret_cast<duckdb_vx_table_filter_set>(input.filters.get()),
        .client_context = reinterpret_cast<duckdb_client_context>(&context),
    };

    duckdb_vx_error error_out = nullptr;
    duckdb_vx_data ffi_global_data = duckdb_table_function_init_global(&ffi_input, &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }

    auto cdata = unique_ptr<CData>(reinterpret_cast<CData *>(ffi_global_data));
    return make_uniq<CTableGlobalData>(std::move(cdata));
}

unique_ptr<LocalTableFunctionState>
init_local(ExecutionContext &, TableFunctionInitInput &, GlobalTableFunctionState *global_state) {
    void *const ffi_global = global_state->Cast<CTableGlobalData>().ffi_data->DataPtr();

    duckdb_vx_data ffi_local_data = duckdb_table_function_init_local(ffi_global);
    auto cdata = unique_ptr<CData>(reinterpret_cast<CData *>(ffi_local_data));
    return make_uniq<CTableLocalData>(std::move(cdata));
}

void function(ClientContext &, TableFunctionInput &input, DataChunk &output) {
    void *const ffi_global = input.global_state->Cast<CTableGlobalData>().ffi_data->DataPtr();
    void *const ffi_local = input.local_state->Cast<CTableLocalData>().ffi_data->DataPtr();

    duckdb_data_chunk chunk = reinterpret_cast<duckdb_data_chunk>(&output);
    duckdb_vx_error error_out = nullptr;
    duckdb_table_function_scan(ffi_global, ffi_local, chunk, &error_out);
    if (error_out) {
        throw InvalidInputException(IntoErrString(error_out));
    }
}

using FilterVec = vector<unique_ptr<Expression>>;

/*
 * Table filter pushdown is used for two tasks in duckdb:
 *
 * 1. Prune files based on filename or hive partitioning, see Parquet
 * filter pushdown. We don't use this because we do own file-level pruning in
 * FileStatsLayoutReader, and we don't support hive partitioning yet.
 *
 * 2. Avoid reading unused file data. Filter expressions are pushed to Vortex,
 * converted to Vortex expressions and used during the scan.
 * Duckdb pushes a subset of expressions i.e. equality operators, and also
 * expressions which return true in pushdown_expression.
 */
void pushdown_complex_filter(const FunctionData &bind_data, FilterVec &filters) {
    const auto &bind = bind_data.Cast<CTableBindData>();
    void *const ffi_bind = bind.ffi_data->DataPtr();
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

unique_ptr<NodeStatistics> c_cardinality(ClientContext &, const FunctionData *bind_data) {
    auto &bind = bind_data->Cast<CTableBindData>();

    duckdb_vx_node_statistics stats = {};
    duckdb_table_function_cardinality(bind.ffi_data->DataPtr(), &stats);

    auto out = make_uniq<NodeStatistics>();
    out->has_estimated_cardinality = stats.has_estimated_cardinality;
    out->estimated_cardinality = stats.estimated_cardinality;
    out->has_max_cardinality = false;

    return out;
}

extern "C" duckdb_value duckdb_vx_tfunc_bind_input_get_parameter(duckdb_vx_tfunc_bind_input ffi_input,
                                                                 size_t index) {
    D_ASSERT(ffi_input);
    const TableFunctionBindInput &input = *reinterpret_cast<TableFunctionBindInput *>(ffi_input);
    return reinterpret_cast<duckdb_value>(new Value(input.inputs[index]));
}

extern "C" void duckdb_vx_tfunc_bind_result_add_column(duckdb_vx_tfunc_bind_result ffi_result,
                                                       const char *name_str,
                                                       size_t name_len,
                                                       duckdb_logical_type ffi_type) {
    D_ASSERT(ffi_result);
    D_ASSERT(name_str);
    D_ASSERT(ffi_type);
    const CTableBindResult &result = *reinterpret_cast<CTableBindResult *>(ffi_result);
    const LogicalType logical_type = *reinterpret_cast<LogicalType *>(ffi_type);

    result.names.emplace_back(name_str, name_len);
    result.return_types.emplace_back(logical_type);
}

/**
 * Called at planning time to determine whether data is partitioned by a
 * given set of columns. Requested columns are GROUP BY parameters i.e. columns
 * over which the query aggregates.
 */
TablePartitionInfo get_partition_info(ClientContext &, TableFunctionPartitionInput &input) {
    const vector<column_t> &ids = input.partition_ids;
    // Our data is partitioned by array exporters. Each exporter processes a
    // single Array which belongs to a single file. If data is partitioned only
    // by file_index, there is one unique value for an Array. Otherwise there
    // may be multiple values.
    return (ids.size() == 1 && ids[0] == COLUMN_IDENTIFIER_FILE_INDEX)
               ? TablePartitionInfo::SINGLE_VALUE_PARTITIONS
               : TablePartitionInfo::NOT_PARTITIONED;
}

/**
 * Duckdb requests this function after exporting the chunk. We answer with
 * partition_index we have exported as well as information about constant
 * columns in this partition. As data is partitioned by array exporters, in
 * each partition ~ exported array file_index is constant.
 */
OperatorPartitionData get_partition_data(ClientContext &, TableFunctionGetPartitionInput &input) {
    void *const ffi_global = input.global_state->Cast<CTableGlobalData>().ffi_data->DataPtr();
    void *const ffi_local = input.local_state->Cast<CTableLocalData>().ffi_data->DataPtr();
    duckdb_vx_partition_data partition_data;
    duckdb_table_function_get_partition_data(ffi_global, ffi_local, &partition_data);

    OperatorPartitionData out(partition_data.partition_index);

    // file_index_column_pos may be INVALID_IDX, but column_index will never
    // be INVALID_IDX, so we can compare directly
    for (const column_t column_index : input.partition_info.partition_columns) {
        if (column_index == partition_data.file_index_column_pos) {
            out.partition_data.emplace_back(Value::UBIGINT(partition_data.file_index));
        } else {
            throw InternalException(StringUtil::Format(
                "get_partition_data: requested column_index %d is not constant for given partition",
                column_index));
        }
    }
    return out;
}

extern "C" void duckdb_vx_string_map_insert(duckdb_vx_string_map map, const char *key, const char *value) {
    D_ASSERT(map);
    D_ASSERT(key);
    D_ASSERT(value);
    reinterpret_cast<InsertionOrderPreservingMap<string> *>(map)->insert(key, value);
}

InsertionOrderPreservingMap<string> c_to_string(TableFunctionToStringInput &input) {
    InsertionOrderPreservingMap<string> result;
    duckdb_vx_string_map ffi_map = reinterpret_cast<duckdb_vx_string_map>(&result);
    void *const ffi_bind = input.bind_data->Cast<CTableBindData>().ffi_data->DataPtr();
    duckdb_table_function_to_string(ffi_bind, ffi_map);
    return result;
}

duckdb_state register_table_function(DatabaseInstance &db, LogicalType parameter, const std::string &name) {
    TableFunction tf(name, {}, function, c_bind, c_init_global, init_local);

    tf.projection_pushdown = true;
    tf.filter_pushdown = true;
    tf.filter_prune = true;
    tf.sampling_pushdown = false;

    tf.pushdown_expression = [](auto &, const auto &, Expression &expression) {
        return duckdb_table_function_pushdown_expression(reinterpret_cast<duckdb_vx_expr>(&expression));
    };
    tf.pushdown_complex_filter = [](auto &, auto &, FunctionData *bind_data, FilterVec &filters) {
        pushdown_complex_filter(*bind_data, filters);
    };
    tf.cardinality = c_cardinality;
    tf.get_partition_info = get_partition_info;
    tf.get_partition_data = get_partition_data;
    tf.to_string = c_to_string;
    tf.table_scan_progress = table_scan_progress;
    tf.statistics = statistics;

    tf.late_materialization = true;
    // Columns that uniquely identify a row for deferred re-fetch in a multi
    // file scan: (file index, row number in file).
    tf.get_row_id_columns = [](auto &, auto) -> vector<column_t> {
        return {COLUMN_IDENTIFIER_FILE_INDEX, COLUMN_IDENTIFIER_FILE_ROW_NUMBER};
    };

    tf.get_virtual_columns = [](auto &, auto) -> virtual_column_map_t {
        return {
            {COLUMN_IDENTIFIER_EMPTY, {"", LogicalTypeId::BOOLEAN}},
            {COLUMN_IDENTIFIER_FILE_INDEX, {"file_index", LogicalType::UBIGINT}},
            // MultiFileReader's file_row_number column is BIGINT.
            // row_idx() is UBIGINT. Use UBIGINT since there's no difference to
            // Duckdb what to compare.
            {COLUMN_IDENTIFIER_FILE_ROW_NUMBER, {"file_row_number", LogicalType::UBIGINT}},
        };
    };

    tf.arguments.resize(1);
    tf.arguments[0] = parameter;

    try {
        auto &system_catalog = Catalog::GetSystemCatalog(db);
        auto data = CatalogTransaction::GetSystemTransaction(db);
        CreateTableFunctionInfo tf_info(tf);
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
