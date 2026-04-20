// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx/data.hpp"
#include "duckdb_vx/duckdb_diagnostics.h"
#include "duckdb_vx/error.hpp"
#include "duckdb_vx/table_function.h"

DUCKDB_INCLUDES_BEGIN
#include "duckdb.h"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/insertion_order_preserving_map.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
DUCKDB_INCLUDES_END

using namespace duckdb;
using vortex::CData;
using vortex::IntoErrString;

struct CTableFunctionInfo final : TableFunctionInfo {
    explicit CTableFunctionInfo(const duckdb_vx_tfunc_vtab_t &vtab) : vtab(vtab) {
    }

    duckdb_vx_tfunc_vtab_t vtab;
};

struct CTableBindData final : TableFunctionData {
    CTableBindData(unique_ptr<CTableFunctionInfo> info_p,
                   unique_ptr<CData> ffi_data_p,
                   const vector<LogicalType> &types)
        : info(std::move(info_p)), ffi_data(std::move(ffi_data_p)), types(types) {
    }

    unique_ptr<FunctionData> Copy() const override {
        D_ASSERT(info->vtab.bind_data_clone);

        duckdb_vx_error error_out = nullptr;
        const auto copied_ffi_data = info->vtab.bind_data_clone(ffi_data->DataPtr(), &error_out);
        if (error_out) {
            throw BinderException(IntoErrString(error_out));
        }

        auto info_p = make_uniq<CTableFunctionInfo>(info->vtab);
        auto ffi_data_p = unique_ptr<CData>(reinterpret_cast<CData *>(copied_ffi_data));
        return make_uniq<CTableBindData>(std::move(info_p), std::move(ffi_data_p), types);
    }

    unique_ptr<CTableFunctionInfo> info;
    unique_ptr<CData> ffi_data;
    vector<LogicalType> types;
};

struct CTableGlobalData final : GlobalTableFunctionState {
    explicit CTableGlobalData(unique_ptr<CData> ffi_data_p) : ffi_data(std::move(ffi_data_p)) {
    }

    idx_t MaxThreads() const override {
        return GlobalTableFunctionState::MAX_THREADS;
    }

    unique_ptr<CData> ffi_data;
};

struct CTableLocalData final : LocalTableFunctionState {
    explicit CTableLocalData(unique_ptr<CData> ffi_data_p) : ffi_data(std::move(ffi_data_p)) {
    }

    unique_ptr<CData> ffi_data;
};

/**
 * Result of the bind function encapsulates the output schema.
 */
struct CTableBindResult {
    vector<LogicalType> &return_types;
    vector<string> &names;
};

double c_table_scan_progress(ClientContext &context,
                             const FunctionData *bind_data,
                             const GlobalTableFunctionState *global_state) {
    auto &bind = bind_data->Cast<CTableBindData>();
    duckdb_client_context c_ctx = reinterpret_cast<duckdb_client_context>(&context);
    void *const c_bind_data = bind.ffi_data->DataPtr();
    void *const c_global_state = global_state->Cast<CTableGlobalData>().ffi_data->DataPtr();
    return bind.info->vtab.table_scan_progress(c_ctx, c_bind_data, c_global_state);
}

unique_ptr<FunctionData> c_bind(ClientContext &context,
                                TableFunctionBindInput &input,
                                vector<LogicalType> &return_types,
                                vector<string> &names) {
    const auto &info = input.table_function.function_info->Cast<CTableFunctionInfo>();

    // Setup bind info to pass into the callback.
    CTableBindResult result = {
        return_types,
        names,
    };

    duckdb_vx_error error_out = nullptr;
    auto ctx = reinterpret_cast<duckdb_client_context>(&context);
    auto ffi_bind_data = info.vtab.bind(ctx,
                                        reinterpret_cast<duckdb_vx_tfunc_bind_input>(&input),
                                        reinterpret_cast<duckdb_vx_tfunc_bind_result>(&result),
                                        &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }

    return make_uniq<CTableBindData>(make_uniq<CTableFunctionInfo>(info.vtab),
                                     unique_ptr<CData>(reinterpret_cast<CData *>(ffi_bind_data)),
                                     return_types);
}

unique_ptr<GlobalTableFunctionState> c_init_global(ClientContext &context, TableFunctionInitInput &input) {
    const auto &bind = input.bind_data->Cast<CTableBindData>();

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
    auto ffi_global_data = bind.info->vtab.init_global(&ffi_input, &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }

    auto cdata = unique_ptr<CData>(reinterpret_cast<CData *>(ffi_global_data));
    return make_uniq<CTableGlobalData>(std::move(cdata));
}

unique_ptr<LocalTableFunctionState> c_init_local(ExecutionContext &context,
                                                 TableFunctionInitInput &input,
                                                 GlobalTableFunctionState *global_state) {
    const auto &bind = input.bind_data->Cast<CTableBindData>();
    auto global_data = global_state->Cast<CTableGlobalData>().ffi_data->DataPtr();

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
    auto ffi_local_data = bind.info->vtab.init_local(&ffi_input, global_data, &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }

    return make_uniq<CTableLocalData>(unique_ptr<CData>(reinterpret_cast<CData *>(ffi_local_data)));
}

void c_function(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
    const auto &bind = input.bind_data->Cast<CTableBindData>();

    auto ctx = reinterpret_cast<duckdb_client_context>(&context);
    const auto bind_data = bind.ffi_data->DataPtr();
    auto global_data = input.global_state->Cast<CTableGlobalData>().ffi_data->DataPtr();
    auto local_data = input.local_state->Cast<CTableLocalData>().ffi_data->DataPtr();

    duckdb_data_chunk chunk = reinterpret_cast<duckdb_data_chunk>(&output);
    duckdb_vx_error error_out = nullptr;
    bind.info->vtab.function(ctx, bind_data, global_data, local_data, chunk, &error_out);
    if (error_out) {
        throw InvalidInputException(IntoErrString(error_out));
    }
}

void c_pushdown_complex_filter(ClientContext &,
                               LogicalGet &,
                               FunctionData *bind_data,
                               vector<unique_ptr<Expression>> &filters) {
    auto &bind = bind_data->Cast<CTableBindData>();
    void *const ffi_bind = bind.ffi_data->DataPtr();

    for (auto iter = filters.begin(); iter != filters.end();) {
        duckdb_vx_error error_out = nullptr;
        duckdb_vx_expr ffi_expr = reinterpret_cast<duckdb_vx_expr>(iter->get());

        const bool pushed = bind.info->vtab.pushdown_complex_filter(ffi_bind, ffi_expr, &error_out);
        if (error_out) {
            throw BinderException(IntoErrString(error_out));
        }

        // If the pushdown complex filter returns true, we can remove the filter from the list.
        iter = pushed ? filters.erase(iter) : std::next(iter);
    }
}

unique_ptr<NodeStatistics> c_cardinality(ClientContext &, const FunctionData *bind_data) {
    auto &bind = bind_data->Cast<CTableBindData>();

    duckdb_vx_node_statistics stats = {};
    bind.info->vtab.cardinality(bind.ffi_data->DataPtr(), &stats);

    auto out = make_uniq<NodeStatistics>();
    out->has_estimated_cardinality = stats.has_estimated_cardinality;
    out->estimated_cardinality = stats.estimated_cardinality;
    out->has_max_cardinality = stats.has_max_cardinality;
    out->max_cardinality = stats.max_cardinality;

    return out;
}

extern "C" duckdb_value duckdb_vx_tfunc_bind_input_get_parameter(duckdb_vx_tfunc_bind_input ffi_input,
                                                                 size_t index) {
    if (!ffi_input) {
        return nullptr;
    }

    const TableFunctionBindInput &input = *reinterpret_cast<TableFunctionBindInput *>(ffi_input);
    if (index >= input.inputs.size()) {
        return nullptr;
    }
    return reinterpret_cast<duckdb_value>(new Value(input.inputs[index]));
}

extern "C" void duckdb_vx_tfunc_bind_result_add_column(duckdb_vx_tfunc_bind_result ffi_result,
                                                       const char *name_str,
                                                       size_t name_len,
                                                       duckdb_logical_type ffi_type) {
    if (!name_str || !ffi_type) {
        return;
    }
    const auto result = reinterpret_cast<CTableBindResult *>(ffi_result);
    const auto logical_type = reinterpret_cast<LogicalType *>(ffi_type);

    result->names.emplace_back(name_str, name_len);
    result->return_types.emplace_back(*logical_type);
}

static Value &UnwrapValue(duckdb_value value) {
    return *(reinterpret_cast<Value *>(value));
}

using BaseStatisticsPtr = unique_ptr<BaseStatistics>;

BaseStatisticsPtr numeric_stats(duckdb_vx_column_statistics &stats, LogicalType type) {
    BaseStatistics out = StringStats::CreateUnknown(type);
    if (stats.min) {
        NumericStats::SetMin(out, UnwrapValue(stats.min));
        duckdb_destroy_value(&stats.min);
    }
    if (stats.max) {
        NumericStats::SetMax(out, UnwrapValue(stats.max));
        duckdb_destroy_value(&stats.max);
    }
    if (!(stats.flags & ColumnStatisticsFlags::HasNull)) {
        out.Set(StatsInfo::CANNOT_HAVE_NULL_VALUES);
    }
    return out.ToUnique();
}

BaseStatisticsPtr string_stats(duckdb_vx_column_statistics &stats, LogicalType type) {
    BaseStatistics out = StringStats::CreateUnknown(type);
    if (stats.min) {
        StringStats::SetMin(out, StringValue::Get(UnwrapValue(stats.min)));
        duckdb_destroy_value(&stats.min);
    }
    if (stats.max) {
        StringStats::SetMax(out, StringValue::Get(UnwrapValue(stats.max)));
        duckdb_destroy_value(&stats.max);
    }
    if (stats.flags & ColumnStatisticsFlags::HasMaxStringLength) {
        StringStats::SetMaxStringLength(out, stats.max_string_length);
    }
    if (!(stats.flags & ColumnStatisticsFlags::HasNull)) {
        out.Set(StatsInfo::CANNOT_HAVE_NULL_VALUES);
    }
    return out.ToUnique();
}

BaseStatisticsPtr base_stats(duckdb_vx_column_statistics &stats, LogicalType type) {
    BaseStatistics out = StringStats::CreateUnknown(type);
    if (!(stats.flags & ColumnStatisticsFlags::HasNull)) {
        out.Set(StatsInfo::CANNOT_HAVE_NULL_VALUES);
    }
    return out.ToUnique();
}

BaseStatisticsPtr to_duckdb_statistics(duckdb_vx_column_statistics &statistics, LogicalType type) {
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

struct VortexRowGroup final : PartitionRowGroup {
    VortexRowGroup(const CTableBindData &bind_data, idx_t file_idx)
        : bind_data(bind_data), file_idx(file_idx) {
    }
    ~VortexRowGroup() override = default;

    BaseStatisticsPtr GetColumnStatistics(const StorageIndex &storage_index) override {
        const idx_t column_idx = storage_index.GetPrimaryIndex();
        D_ASSERT(!IsVirtualColumn(column_idx));
        D_ASSERT(column_idx < bind_data.types.size());

        void *const bind_ffi = bind_data.ffi_data->DataPtr();
        if (!bind_data.info->vtab.get_column_stats(bind_ffi, file_idx, column_idx, &statistics)) {
            return {};
        }
        return to_duckdb_statistics(statistics, bind_data.types[column_idx]);
    }

    bool MinMaxIsExact(const BaseStatistics &, const StorageIndex &) override {
        // This function is called after GetColumnStatistics with same storage
        // index, thus we can reuse calculated statistics
        // duckdb/src/optimizer/statistics/operator/propagate_aggregate.cpp:84
        return statistics.flags & ColumnStatisticsFlags::MinMaxIsExact;
    }

    const CTableBindData &bind_data;
    const idx_t file_idx;
    duckdb_vx_column_statistics statistics;
};

vector<PartitionStatistics> get_partition_stats(ClientContext &, GetPartitionStatsInput &input) {
    const CTableBindData &bind_data = input.bind_data->Cast<CTableBindData>();
    void *const ffi_bind = bind_data.ffi_data->DataPtr();

    const size_t partitions_len = bind_data.info->vtab.get_partition_count(ffi_bind);
    vector<idx_t> row_counts(partitions_len);
    if (!bind_data.info->vtab.get_partition_row_counts(ffi_bind, row_counts.data(), partitions_len)) {
        return {};
    }

    idx_t offset = 0;
    vector<PartitionStatistics> out(partitions_len);
    for (idx_t i = 0; i < partitions_len; ++i) {
        out[i].row_start = offset;
        out[i].count = row_counts[i];
        out[i].count_type = CountType::COUNT_EXACT;
        out[i].partition_row_group = make_shared_ptr<VortexRowGroup>(bind_data, i);
        offset += row_counts[i];
    }
    return out;
}

BaseStatisticsPtr statistics(ClientContext &, const FunctionData *bind_data, column_t column_idx) {
    if (IsVirtualColumn(column_idx)) {
        return {};
    }
    const auto &bind = bind_data->Cast<CTableBindData>();
    duckdb_vx_column_statistics statistics;
    void *const ffi_bind = bind.ffi_data->DataPtr();
    if (!bind.info->vtab.get_column_stats(ffi_bind, -1, column_idx, &statistics)) {
        return {};
    }
    return to_duckdb_statistics(statistics, bind.types[column_idx]);
}

OperatorPartitionData get_partition_data(ClientContext &, TableFunctionGetPartitionInput &input) {
    if (input.partition_info.RequiresPartitionColumns()) {
        throw InternalException("TableScan::GetPartitionData: partition columns not supported");
    }
    const auto &bind = input.bind_data->Cast<CTableBindData>();
    void *const ffi_local = input.local_state->Cast<CTableLocalData>().ffi_data->DataPtr();
    const idx_t batch_index = bind.info->vtab.get_partition_data(ffi_local);
    return OperatorPartitionData(batch_index);
}

InsertionOrderPreservingMap<string> c_to_string(TableFunctionToStringInput &input) {
    InsertionOrderPreservingMap<string> result;
    duckdb_vx_string_map ffi_map = reinterpret_cast<duckdb_vx_string_map>(&result);
    void *const ffi_bind = input.bind_data->Cast<CTableBindData>().ffi_data->DataPtr();
    static_cast<CTableFunctionInfo &>(*input.table_function.function_info).vtab.to_string(ffi_bind, ffi_map);
    return result;
}

extern "C" duckdb_state duckdb_vx_tfunc_register(duckdb_database ffi_db, const duckdb_vx_tfunc_vtab_t *vtab) {
    if (!ffi_db || !vtab) {
        return DuckDBError;
    }

    auto wrapper = reinterpret_cast<duckdb::DatabaseWrapper *>(ffi_db);
    auto db = wrapper->database->instance;
    auto tf = TableFunction(vtab->name, {}, c_function, c_bind, c_init_global, c_init_local);

    tf.projection_pushdown = true;
    tf.filter_pushdown = true;
    // We can prune out filter columns that are unused in the remainder of the query plan.
    // e.g. in "SELECT i FROM tbl WHERE j = 42" j does not leave Vortex table function.
    tf.filter_prune = true;
    tf.sampling_pushdown = false;
    tf.late_materialization = false;

    tf.pushdown_complex_filter = c_pushdown_complex_filter;
    tf.cardinality = c_cardinality;
    tf.to_string = c_to_string;
    tf.table_scan_progress = c_table_scan_progress;
    tf.get_partition_data = get_partition_data;
    tf.get_partition_stats = get_partition_stats;
    tf.statistics = statistics;

    tf.get_virtual_columns = [](auto &, auto) -> virtual_column_map_t {
        return {{COLUMN_IDENTIFIER_EMPTY, TableColumn("", LogicalTypeId::BOOLEAN)}};
    };

    tf.arguments.reserve(vtab->parameter_count);
    for (size_t i = 0; i < vtab->parameter_count; i++) {
        auto logical_type = reinterpret_cast<LogicalType *>(vtab->parameters[i]);
        tf.arguments.emplace_back(*logical_type);
    }

    tf.function_info = make_shared_ptr<CTableFunctionInfo>(*vtab);

    try {
        auto &system_catalog = Catalog::GetSystemCatalog(*db);
        auto data = CatalogTransaction::GetSystemTransaction(*db);
        CreateTableFunctionInfo tf_info(tf);
        // Allow registering multiple overloads with the same name but different parameter types.
        tf_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        system_catalog.CreateFunction(data, tf_info);
    } catch (...) {
        return DuckDBError;
    }
    return DuckDBSuccess;
}
