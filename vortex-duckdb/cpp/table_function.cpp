// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "data.hpp"
#include "error.hpp"
#include "table_function.hpp"
#include "multi_file_reader.hpp"
#include "expr.h"
#include "vortex_duckdb.h"
#include "table_function.h"
#include "vortex.h"

#include "duckdb.h"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/default/default_functions.hpp"
#include "duckdb/common/enums/file_glob_options.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/identifier.hpp"
#include "duckdb/common/insertion_order_preserving_map.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/function/partition_stats.hpp"
#include "duckdb/function/scalar_macro_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/logging/logger.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/parser/keyword_helper.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/storage_index.hpp"
#include <limits>

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
    const MultiFileBindData &bind_data = input.get.bind_data->Cast<MultiFileBindData>();
    const optional_idx filename_idx = bind_data.reader_bind.filename_idx;
    const auto &hive_indices = bind_data.reader_bind.hive_partitioning_indexes;
    for (const auto &projection : input.projections) {
        const idx_t scan_index = projection.first;
        if (scan_index == COUNT_STAR_PROJ_IDX) {
            continue;
        }
        const idx_t primary_index = input.get.GetColumnIds()[scan_index].GetPrimaryIndex();
        if (SkipMultiFileReaderColumn(primary_index, filename_idx, hive_indices)) {
            return false;
        }
    }

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

    result.names.emplace_back(string(name_str, name_len));
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

bool is_vortex_scan(const BoundTableFunction &function) {
    return function.bind == MultiFileFunction<VortexReaderInterface>::MultiFileBind;
}

unique_ptr<MultiFileReader> get_multi_file_reader(const BoundTableFunction &) {
    return make_uniq<VortexMultiFileReader>();
}

unique_ptr<BaseStatistics> VortexRowGroup::GetColumnStatistics(const StorageIndex &storage_index) {
    const idx_t idx = storage_index.GetPrimaryIndex();
    if (idx >= columns.size()) {
        return {};
    }
    const MultiFileColumnDefinition &column = columns[idx];

    duckdb_column_statistics statistics = {};
    const void *const ffi_footer_ptr = ffi_footer->DataPtr();
    if (!duckdb_footer_get_statistics(ffi_footer_ptr, column.name.c_str(), column.name.size(), &statistics)) {
        return {};
    }

    unique_ptr<BaseStatistics> stats = to_duckdb_statistics(statistics);
    if (stats && stats->GetType() != column.type) {
        return {};
    }
    return stats;
}

static vector<PartitionStatistics> get_partition_stats(ClientContext &, GetPartitionStatsInput &input) {
    const MultiFileBindData &bind_data = input.bind_data->Cast<MultiFileBindData>();
    void *const ffi_bind = get_ffi_bind(input.bind_data.get());

    if (!duckdb_table_function_can_get_partition_stats(ffi_bind)) {
        return {};
    }

    vector<OpenFileInfo> files = bind_data.file_list->GetAllFiles();
    vector<PartitionStatistics> result(files.size());
    idx_t row_start = 0;
    uint64_t count = 0;
    duckdb_vx_error error = nullptr;
    for (size_t i = 0; i < files.size(); ++i) {
        const std::string_view path = files[i].path;
        duckdb_vx_data raw = duckdb_footer_get_cached(ffi_bind, path.data(), path.size(), &count, &error);
        unique_ptr<CData> cdata(reinterpret_cast<CData *>(raw));
        if (error) {
            throw BinderException(IntoErrString(error));
        }
        if (!cdata) {
            return {};
        }

        PartitionStatistics &stats = result[i];
        stats.row_start = row_start;
        stats.count = count;
        stats.count_type = CountType::COUNT_EXACT;
        row_start += count;
        stats.partition_row_group = make_shared_ptr<VortexRowGroup>(std::move(cdata), bind_data.columns);
    }
    return result;
}

static void serialize(Serializer &serializer,
                      const optional_ptr<FunctionData> base_bind_data,
                      const BoundTableFunction &) {
    if (!dynamic_cast<BinarySerializer *>(&serializer)) {
        throw SerializationException("Vortex table function needs a binary serializer");
    }
    const MultiFileBindData &bind_data = base_bind_data->Cast<MultiFileBindData>();
    vector<OpenFileInfo> open_files = bind_data.file_list->GetAllFiles();
    vector<string> files(open_files.size());
    for (size_t i = 0; i < files.size(); ++i) {
        files[i] = open_files[i].path;
    }

    serializer.WriteProperty(100, "files", files);
    serializer.WriteProperty(101, "types", bind_data.types);
    serializer.WriteProperty(102, "names", bind_data.names);
    serializer.WriteProperty(103, "file_options", bind_data.file_options);

    const uint8_t *state_data = nullptr;
    size_t state_len = 0;
    duckdb_vx_error error = nullptr;
    void *const ffi_bind = get_ffi_bind(base_bind_data.get());
    duckdb_vx_data raw = duckdb_table_function_serialize(ffi_bind, &state_data, &state_len, &error);
    const unique_ptr<CData> state_owner(reinterpret_cast<CData *>(raw));
    if (error) {
        throw SerializationException(IntoErrString(error));
    }
    if (state_len >= std::numeric_limits<uint32_t>::max()) {
        throw SerializationException("State exceeds uint32_t max value");
    }

    const string_t vortex_state = {reinterpret_cast<const char *>(state_data),
                                   static_cast<uint32_t>(state_len)};
    serializer.WritePropertyWithDefault(104, "vortex_state", vortex_state);
}

static unique_ptr<FunctionData> deserialize(Deserializer &deserializer, BoundTableFunction &) {
    if (!dynamic_cast<BinaryDeserializer *>(&deserializer)) {
        throw SerializationException("Vortex table function needs a binary deserializer");
    }
    auto &context = deserializer.Get<ClientContext &>();
    auto files = deserializer.ReadProperty<vector<string>>(100, "files");
    auto types = deserializer.ReadProperty<vector<LogicalType>>(101, "types");
    auto names = StringsToIdentifiers(deserializer.ReadProperty<vector<string>>(102, "names"));
    auto file_options = deserializer.ReadProperty<MultiFileOptions>(103, "file_options");
    string state = deserializer.ReadPropertyWithDefault<string>(104, "vortex_state");

    vector<Value> file_values;
    file_values.reserve(files.size());
    for (const string &path : files) {
        file_values.emplace_back(path);
    }
    FileGlobInput glob_input(FileGlobOptions::FALLBACK_GLOB, "vortex");
    glob_input.allow_empty = true; // We must allow a list reduced to 0 files

    auto multi_file_reader = make_uniq<VortexMultiFileReader>();
    Value list_input = Value::LIST(LogicalType::VARCHAR, std::move(file_values));
    auto file_list = multi_file_reader->CreateFileList(context, list_input, glob_input);
    auto bind_data = MultiFileFunction<VortexReaderInterface>::MultiFileBindInternal( //
        context,
        std::move(multi_file_reader),
        std::move(file_list),
        types,
        names,
        std::move(file_options),
        make_uniq<BaseFileReaderOptions>(),
        make_uniq<VortexReaderInterface>());
    if (state.empty()) {
        return bind_data;
    }

    auto &mf_bind = bind_data->Cast<MultiFileBindData>();
    vector<duckdb_logical_type> ffi_types(mf_bind.types.size());
    for (size_t i = 0; i < ffi_types.size(); ++i) {
        ffi_types[i] = reinterpret_cast<duckdb_logical_type>(&mf_bind.types[i]);
    }
    duckdb_vx_error error = nullptr;
    void *const ffi_bind = get_ffi_bind(bind_data.get());
    duckdb_table_function_deserialize(ffi_bind,
                                      reinterpret_cast<const uint8_t *>(state.data()),
                                      state.size(),
                                      ffi_types.data(),
                                      ffi_types.size(),
                                      &error);
    if (error) {
        throw SerializationException(IntoErrString(error));
    }
    return bind_data;
}

static bool projection_expression_pushdown(ClientContext &,
                                           const TableFunctionProjectionExpressionInput &input) {
    duckdb_vx_expr ffi_expr = get_ffi_expr(input.expr);
    void *const ffi_bind = get_ffi_bind(input.get.bind_data.get());
    duckdb_vx_error error_out = nullptr;

    const idx_t column_id = input.get.GetColumnIds()[input.column_index].GetPrimaryIndex();

    const bool ret =
        duckdb_table_function_pushdown_projection_expression(ffi_bind, ffi_expr, column_id, &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }

    if (ret) {
        MultiFileBindData &bind_data = input.get.bind_data->Cast<MultiFileBindData>();
        bind_data.types[column_id] = input.expr.GetReturnType();
        bind_data.columns[column_id].type = input.expr.GetReturnType();
    }

    return ret;
}

duckdb_state register_table_function(DatabaseInstance &db, LogicalType parameter, const std::string &name) {
    MultiFileFunction<VortexReaderInterface> fn {Identifier(name)};
    FunctionSignature signature;
    signature.AddPositionalOnly("path", std::move(parameter));
    signature.WithTypedKwargs("options", [](TypedKwargs &options) {
        options.Add("filename", LogicalType::ANY)
            .Add("allow_empty", LogicalType::BOOLEAN)
            .Add("hive_partitioning", LogicalType::BOOLEAN);
    });
    fn.GetSignature() = std::move(signature);

    fn.filter_pushdown = true;
    fn.filter_prune = true;

    fn.pushdown_expression = [](auto &, const auto &, Expression &expression) {
        return duckdb_table_function_pushdown_expression(reinterpret_cast<duckdb_vx_expr>(&expression));
    };
    fn.pushdown_complex_filter = [](auto &, auto &, FunctionData *bind_data, FilterVec &filters) {
        pushdown_complex_filter(*bind_data, filters);
    };
    fn.projection_expression_pushdown = projection_expression_pushdown;
    fn.to_string = to_string;

    fn.late_materialization = true;
    // Columns that uniquely identify a row for deferred re-fetch in a multi
    // file scan: (file index, row number in file).
    fn.get_row_id_columns = [](auto &, auto) -> vector<column_t> {
        return {COLUMN_IDENTIFIER_FILE_INDEX, COLUMN_IDENTIFIER_FILE_ROW_NUMBER};
    };

    fn.get_partition_stats = get_partition_stats;
    fn.statistics_extended = MultiFileFunction<VortexReaderInterface>::MultiFileScanStatsExtended;
    fn.get_multi_file_reader = get_multi_file_reader;

    fn.serialize = serialize;
    fn.deserialize = deserialize;

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

extern "C" duckdb_state duckdb_vx_register_version_function(duckdb_database ffi_db, const char *version) {
    D_ASSERT(ffi_db);
    D_ASSERT(version);
    const DatabaseWrapper &wrapper = *reinterpret_cast<DatabaseWrapper *>(ffi_db);
    DatabaseInstance &db = *wrapper.database->instance;

    unique_ptr<ParsedExpression> expr = ConstantExpression::FromValue(version);
    CreateMacroInfo info(CatalogType::MACRO_ENTRY);
    info.macros.emplace_back(make_uniq<ScalarMacroFunction>(std::move(expr)));
    info.SetQualifiedName({{DEFAULT_SCHEMA}, "vortex_version"});
    info.internal = true; // only internals allowed in system catalog

    try {
        auto data = CatalogTransaction::GetSystemTransaction(db);
        Catalog::GetSystemCatalog(db).CreateFunction(data, info);
    } catch (const std::exception &e) {
        ErrorData data(e);
        DUCKDB_LOG_ERROR(db, "Failed to create the vortex_version macro:\t" + data.Message());
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
