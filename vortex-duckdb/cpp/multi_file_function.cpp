// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * C++ adapters that bridge a duckdb_vx_mff_vtab_t to DuckDB's MultiFileFunction<OP>.
 *
 * Layered design:
 *   - VortexBaseFileReaderOptions : BaseFileReaderOptions   - opaque options handle
 *   - VortexFileReader            : BaseFileReader          - per-file scan adapter
 *   - VortexMultiFileReaderInterface : MultiFileReaderInterface - cross-file orchestrator
 *   - VortexMultiFileFunctionOp                              - OP type for MultiFileFunction<OP>
 *
 * Each adapter holds a non-owning pointer to the registered vtab and an
 * extension-owned FFI handle. The FFI handle is freed via the vtab's free_*
 * callback in the destructor.
 */

#include "duckdb_vx/data.hpp"
#include "duckdb_vx/duckdb_diagnostics.h"
#include "duckdb_vx/error.hpp"
#include "duckdb_vx/multi_file_function.h"

#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <utility>

DUCKDB_INCLUDES_BEGIN
#include "duckdb.h"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/multi_file/base_file_reader.hpp"
#include "duckdb/common/multi_file/multi_file_data.hpp"
#include "duckdb/common/multi_file/multi_file_function.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/function/partition_stats.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
DUCKDB_INCLUDES_END

using namespace duckdb;
using vortex::IntoErrString;
constexpr column_t COLUMN_IDENTIFIER_FILE_INDEX = MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX;
constexpr column_t COLUMN_IDENTIFIER_FILE_ROW_NUMBER = MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;

namespace {

/**
 * Internal bind data stored on the catalog-owned function. We keep the vtable
 * here so per-bind/per-file adapters can find it without a separate registry.
 */
struct VortexMultiFileFunctionInfo : TableFunctionInfo {
    explicit VortexMultiFileFunctionInfo(const duckdb_vx_mff_vtab_t &vtab_p) : vtab(vtab_p) {
    }

    const duckdb_vx_mff_vtab_t vtab;
};

class VortexBaseFileReaderOptions : public BaseFileReaderOptions {
public:
    VortexBaseFileReaderOptions(const duckdb_vx_mff_vtab_t &vtab, duckdb_vx_mff_options handle)
        : vtab(vtab), handle(handle) {
    }
    ~VortexBaseFileReaderOptions() override {
        if (handle) {
            vtab.free_options(handle);
        }
    }

    /** Release ownership of the FFI handle to the caller. */
    duckdb_vx_mff_options Release() {
        auto out = handle;
        handle = nullptr;
        return out;
    }

    const duckdb_vx_mff_vtab_t &vtab;

private:
    duckdb_vx_mff_options handle;
};

/**
 * Bind data attached to the MultiFileBindData. Holds the FFI bind-data handle
 * for the lifetime of the prepared statement.
 */
struct VortexMultiFileBindData : public TableFunctionData {
    VortexMultiFileBindData(const duckdb_vx_mff_vtab_t &vtab, duckdb_vx_mff_bind_data handle)
        : vtab(vtab), handle(handle) {
    }
    ~VortexMultiFileBindData() override {
        if (handle) {
            vtab.free_bind_data(handle);
        }
    }

    bool SupportStatementCache() const override {
        return false;
    }

    unique_ptr<FunctionData> Copy() const override {
        duckdb_vx_error error_out = nullptr;
        auto cloned = vtab.clone_bind_data(handle, &error_out);
        if (error_out) {
            throw InternalException(IntoErrString(error_out));
        }
        return make_uniq<VortexMultiFileBindData>(vtab, cloned);
    }

    const duckdb_vx_mff_vtab_t &vtab;
    duckdb_vx_mff_bind_data handle;
};

/**
 * Global state for a single multi-file scan. Distinct from MultiFileGlobalState
 * (which DuckDB owns); this is the *interface*-owned global state slot.
 */
class VortexInterfaceGlobalState : public GlobalTableFunctionState {
public:
    VortexInterfaceGlobalState(const duckdb_vx_mff_vtab_t &vtab,
                               duckdb_vx_mff_global handle,
                               const MultiFileGlobalState &multi_file_state)
        : vtab(vtab), handle(handle), multi_file_state(&multi_file_state) {
    }
    ~VortexInterfaceGlobalState() override {
        if (handle) {
            vtab.free_global(handle);
        }
    }

    const duckdb_vx_mff_vtab_t &vtab;
    duckdb_vx_mff_global handle;
    const MultiFileGlobalState *multi_file_state;
};

class VortexInterfaceLocalState : public LocalTableFunctionState {
public:
    VortexInterfaceLocalState(const duckdb_vx_mff_vtab_t &vtab, duckdb_vx_mff_local handle)
        : vtab(vtab), handle(handle) {
    }
    ~VortexInterfaceLocalState() override {
        if (handle) {
            vtab.free_local(handle);
        }
    }

    const duckdb_vx_mff_vtab_t &vtab;
    duckdb_vx_mff_local handle;
};

static Value &UnwrapValue(duckdb_value value) {
    return *(reinterpret_cast<Value *>(value));
}

void DestroyValues(duckdb_column_statistics &stats) {
    if (stats.min) {
        duckdb_destroy_value(&stats.min);
    }
    if (stats.max) {
        duckdb_destroy_value(&stats.max);
    }
}

unique_ptr<BaseStatistics> NumericStatsFrom(duckdb_column_statistics &stats, const LogicalType &type) {
    BaseStatistics out = BaseStatistics::CreateUnknown(type);
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

unique_ptr<BaseStatistics> StringStatsFrom(duckdb_column_statistics &stats, const LogicalType &type) {
    BaseStatistics out = BaseStatistics::CreateUnknown(type);
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

unique_ptr<BaseStatistics> BaseStatsFrom(duckdb_column_statistics &stats, const LogicalType &type) {
    BaseStatistics out = BaseStatistics::CreateUnknown(type);
    DestroyValues(stats);
    if (!stats.has_null) {
        out.Set(StatsInfo::CANNOT_HAVE_NULL_VALUES);
    }
    return out.ToUnique();
}

unique_ptr<BaseStatistics> ColumnStatsFrom(duckdb_column_statistics &stats, const LogicalType &type) {
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
    case LogicalTypeId::HUGEINT:
        return NumericStatsFrom(stats, type);
    case LogicalTypeId::VARCHAR:
    case LogicalTypeId::BLOB:
        return StringStatsFrom(stats, type);
    case LogicalTypeId::STRUCT:
        DestroyValues(stats);
        return nullptr;
    default:
        return BaseStatsFrom(stats, type);
    }
}

/**
 * Per-file reader adapter. DuckDB's MultiFileFunction<OP> drives one of these
 * per opened file; each Scan call asks the extension for the next chunk.
 */
class VortexFileReader : public BaseFileReader {
public:
    VortexFileReader(OpenFileInfo file_p,
                     const duckdb_vx_mff_vtab_t &vtab_p,
                     duckdb_vx_mff_reader handle_p)
        : BaseFileReader(std::move(file_p)), vtab(vtab_p), handle(handle_p) {
    }
    ~VortexFileReader() override {
        if (handle) {
            vtab.free_reader(handle);
        }
    }

    string GetReaderType() const override {
        return "vortex";
    }

    void AddVirtualColumn(column_t virtual_column_id) override {
        if (columns.empty()) {
            throw InternalException("Vortex reader received virtual column before column registration");
        }
        virtual_column_ids[columns.size() - 1] = virtual_column_id;
    }

    void PrepareReader(ClientContext &, GlobalTableFunctionState &gstate) override {
        // Translate the multi-file column ids into projected column names, then
        // hand DuckDB's TableFilterSet through to Rust as a borrow. The reader
        // stores the resulting projection/filter so it can apply them when the
        // scan starts.
        auto &g = gstate.Cast<VortexInterfaceGlobalState>();
        std::unordered_set<column_t> projected_column_ids;
        if (g.multi_file_state && !g.multi_file_state->projection_ids.empty()) {
            projected_column_ids.reserve(g.multi_file_state->projection_ids.size());
            for (const auto &projection_id : g.multi_file_state->projection_ids) {
                if (projection_id >= g.multi_file_state->column_indexes.size()) {
                    throw InternalException("Vortex projection id out of range");
                }
                projected_column_ids.insert(g.multi_file_state->column_indexes[projection_id].GetPrimaryIndex());
            }
        }

        std::vector<duckdb_vx_mff_column> ffi_proj;
        ffi_proj.reserve(column_ids.size());
        for (idx_t i = 0; i < column_ids.size(); i++) {
            auto local_id = column_ids[MultiFileLocalIndex(i)];
            // `local_id` is an index into our local `columns` schema. Physical
            // columns use their local id directly; non-constant virtual columns
            // are appended to `columns` by DuckDB's mapper and announced via
            // AddVirtualColumn.
            const auto &col = columns[local_id];
            auto virtual_entry = virtual_column_ids.find(local_id.GetId());
            const bool is_virtual = virtual_entry != virtual_column_ids.end();
            const auto column_id = is_virtual ? virtual_entry->second : local_id.GetId();
            const bool is_projected =
                projected_column_ids.empty() || projected_column_ids.find(column_id) != projected_column_ids.end();
            ffi_proj.push_back({col.name.c_str(), col.name.size(), column_id, is_virtual, is_projected});
        }
        auto filter_ptr = reinterpret_cast<duckdb_vx_table_filter_set>(filters.get());
        duckdb_vx_error error_out = nullptr;
        vtab.prepare_reader(handle, ffi_proj.data(), ffi_proj.size(), filter_ptr, &error_out);
        if (error_out) {
            throw IOException(IntoErrString(error_out));
        }
    }

    bool TryInitializeScan(ClientContext &,
                           GlobalTableFunctionState &gstate,
                           LocalTableFunctionState &lstate) override {
        auto &g = gstate.Cast<VortexInterfaceGlobalState>();
        auto &l = lstate.Cast<VortexInterfaceLocalState>();
        duckdb_vx_error error_out = nullptr;
        const bool ok = vtab.try_initialize_scan(handle, g.handle, l.handle, &error_out);
        if (error_out) {
            throw IOException(IntoErrString(error_out));
        }
        return ok;
    }

    void PrepareScan(ClientContext &,
                     GlobalTableFunctionState &gstate,
                     LocalTableFunctionState &lstate) override {
        auto &g = gstate.Cast<VortexInterfaceGlobalState>();
        auto &l = lstate.Cast<VortexInterfaceLocalState>();
        duckdb_vx_error error_out = nullptr;
        vtab.prepare_scan(handle, g.handle, l.handle, &error_out);
        if (error_out) {
            throw IOException(IntoErrString(error_out));
        }
    }

    AsyncResult Scan(ClientContext &,
                     GlobalTableFunctionState &gstate,
                     LocalTableFunctionState &lstate,
                     DataChunk &chunk) override {
        auto &g = gstate.Cast<VortexInterfaceGlobalState>();
        auto &l = lstate.Cast<VortexInterfaceLocalState>();
        duckdb_vx_error error_out = nullptr;
        auto chunk_handle = reinterpret_cast<duckdb_data_chunk>(&chunk);
        const bool ok = vtab.scan(handle, g.handle, l.handle, chunk_handle, &error_out);
        if (!ok || error_out) {
            throw IOException(IntoErrString(error_out));
        }
        // Translate "0 rows" into FINISHED so the multi-file scanner advances
        // to the next file. Otherwise, signal we may have more.
        return chunk.size() == 0 ? AsyncResult(SourceResultType::FINISHED)
                                 : AsyncResult(SourceResultType::HAVE_MORE_OUTPUT);
    }

    unique_ptr<BaseStatistics> GetStatistics(ClientContext &, const string &name) override {
        for (auto &col : columns) {
            if (col.name != name) {
                continue;
            }
            duckdb_column_statistics stats = {};
            if (!vtab.get_statistics(handle, name.c_str(), name.size(), &stats)) {
                return nullptr;
            }
            return ColumnStatsFrom(stats, col.type);
        }
        return nullptr;
    }

    double GetProgressInFile(ClientContext &) override {
        return vtab.progress_in_file(handle);
    }

private:
    const duckdb_vx_mff_vtab_t &vtab;
    duckdb_vx_mff_reader handle;
    std::unordered_map<idx_t, column_t> virtual_column_ids;
};

/**
 * Cross-file orchestrator. Implements only the methods the basic scan pipeline
 * needs; everything else (hive partitioning, COPY, union-by-name, virtual cols)
 * defaults to the base-class behaviour.
 */
class VortexMultiFileReaderInterface : public MultiFileReaderInterface {
public:
    VortexMultiFileReaderInterface() = default;

    unique_ptr<BaseFileReaderOptions> InitializeOptions(ClientContext &context,
                                                        optional_ptr<TableFunctionInfo> info) override {
        if (!info) {
            throw BinderException("Vortex multi-file function requires TableFunctionInfo");
        }
        vtab = &info->Cast<VortexMultiFileFunctionInfo>().vtab;
        auto &vtab = Vtab();
        duckdb_vx_error error_out = nullptr;
        auto ctx = reinterpret_cast<duckdb_client_context>(&context);
        auto handle = vtab.create_options(ctx, &error_out);
        if (error_out) {
            throw BinderException(IntoErrString(error_out));
        }
        return make_uniq<VortexBaseFileReaderOptions>(vtab, handle);
    }

    bool ParseCopyOption(ClientContext &, const string &, const vector<Value> &,
                         BaseFileReaderOptions &, vector<string> &, vector<LogicalType> &) override {
        return false;
    }

    bool ParseOption(ClientContext &, const string &, const Value &, MultiFileOptions &,
                     BaseFileReaderOptions &) override {
        return false;
    }

    unique_ptr<TableFunctionData> InitializeBindData(MultiFileBindData &,
                                                     unique_ptr<BaseFileReaderOptions> options) override {
        auto &vtab = Vtab();
        auto &vortex_options = options->Cast<VortexBaseFileReaderOptions>();
        // Take ownership of the options handle and pass it to the FFI.
        duckdb_vx_error error_out = nullptr;
        auto bind_handle = vtab.initialize_bind_data(vortex_options.Release(), &error_out);
        if (error_out) {
            throw BinderException(IntoErrString(error_out));
        }
        return make_uniq<VortexMultiFileBindData>(vtab, bind_handle);
    }

    void BindReader(ClientContext &context, vector<LogicalType> &return_types, vector<string> &names,
                    MultiFileBindData &bind_data) override {
        auto &vtab = Vtab();
        auto first_file = bind_data.file_list->GetFirstFile();
        auto &vortex_bind = bind_data.bind_data->Cast<VortexMultiFileBindData>();

        // Schema collection writer: a pair of vectors that the FFI populates.
        struct SchemaWriter {
            vector<string> &names;
            vector<LogicalType> &types;
        };
        SchemaWriter writer = {names, return_types};

        duckdb_vx_error error_out = nullptr;
        auto ctx = reinterpret_cast<duckdb_client_context>(&context);
        vtab.bind_reader(ctx, vortex_bind.handle, first_file.path.c_str(), first_file.path.size(),
                         reinterpret_cast<duckdb_vx_mff_schema_writer>(&writer), &error_out);
        if (error_out) {
            throw BinderException(IntoErrString(error_out));
        }
    }

    unique_ptr<GlobalTableFunctionState> InitializeGlobalState(ClientContext &context,
                                                                MultiFileBindData &bind_data,
                                                                MultiFileGlobalState &multi_file_state) override {
        auto &vtab = Vtab();
        auto &vortex_bind = bind_data.bind_data->Cast<VortexMultiFileBindData>();
        duckdb_vx_error error_out = nullptr;
        auto ctx = reinterpret_cast<duckdb_client_context>(&context);
        auto handle = vtab.init_global(ctx, vortex_bind.handle, &error_out);
        if (error_out) {
            throw BinderException(IntoErrString(error_out));
        }
        return make_uniq<VortexInterfaceGlobalState>(vtab, handle, multi_file_state);
    }

    unique_ptr<LocalTableFunctionState> InitializeLocalState(ExecutionContext &,
                                                              GlobalTableFunctionState &gstate) override {
        auto &vtab = Vtab();
        auto &g = gstate.Cast<VortexInterfaceGlobalState>();
        auto handle = vtab.init_local(g.handle);
        return make_uniq<VortexInterfaceLocalState>(vtab, handle);
    }

    void GetVirtualColumns(ClientContext &, MultiFileBindData &, virtual_column_map_t &result) override {
        result.insert(make_pair(COLUMN_IDENTIFIER_FILE_ROW_NUMBER,
                                TableColumn("file_row_number", LogicalType::BIGINT)));
    }

    shared_ptr<BaseFileReader> CreateReader(ClientContext &, GlobalTableFunctionState &, BaseUnionData &,
                                            const MultiFileBindData &) override {
        // UNION BY NAME path - not supported yet.
        throw NotImplementedException("UNION BY NAME is not yet supported by the Vortex multi-file function");
    }

    shared_ptr<BaseFileReader> CreateReader(ClientContext &context, GlobalTableFunctionState &gstate,
                                            const OpenFileInfo &file, idx_t file_idx,
                                            const MultiFileBindData &bind_data) override {
        auto &vtab = Vtab();
        auto &vortex_bind = bind_data.bind_data->Cast<VortexMultiFileBindData>();
        auto &vortex_g = gstate.Cast<VortexInterfaceGlobalState>();
        duckdb_vx_error error_out = nullptr;
        auto ctx = reinterpret_cast<duckdb_client_context>(&context);
        auto handle = vtab.create_reader(ctx, vortex_g.handle, vortex_bind.handle, file.path.c_str(),
                                         file.path.size(), file_idx, &error_out);
        if (error_out) {
            throw IOException(IntoErrString(error_out));
        }
        auto reader = make_shared_ptr<VortexFileReader>(file, vtab, handle);
        // BaseFileReader exposes its file-local schema via the `columns` field;
        // the multi-file reader uses it to build the global<->local column
        // mapping. We don't yet support per-file schema variation, so inherit
        // the bind-time global schema directly.
        reader->columns = bind_data.columns;
        return reader;
    }

    unique_ptr<NodeStatistics> GetCardinality(ClientContext &context, const MultiFileBindData &data,
                                              idx_t file_count) override {
        auto &vtab = Vtab();
        auto &vortex_bind = data.bind_data->Cast<VortexMultiFileBindData>();
        duckdb_vx_node_statistics stats = {};
        if (!vtab.cardinality(vortex_bind.handle, file_count, &stats)) {
            return MultiFileReaderInterface::GetCardinality(context, data, file_count);
        }
        auto out = make_uniq<NodeStatistics>();
        out->has_estimated_cardinality = stats.has_estimated_cardinality;
        out->estimated_cardinality = stats.estimated_cardinality;
        out->has_max_cardinality = stats.has_max_cardinality;
        out->max_cardinality = stats.max_cardinality;
        return out;
    }

    unique_ptr<MultiFileReaderInterface> Copy() override {
        auto copy = make_uniq<VortexMultiFileReaderInterface>();
        copy->vtab = vtab;
        return copy;
    }

private:
    const duckdb_vx_mff_vtab_t &Vtab() const {
        if (!vtab) {
            throw InternalException("VortexMultiFileReaderInterface used before InitializeOptions");
        }
        return *vtab;
    }

    const duckdb_vx_mff_vtab_t *vtab = nullptr;
};

/**
 * The OP type required by MultiFileFunction<OP>. Holds a pointer to the vtab so
 * CreateInterface can construct a VortexMultiFileReaderInterface bound to it.
 */
struct VortexMultiFileFunctionOp {
    static unique_ptr<MultiFileReaderInterface> CreateInterface(ClientContext &) {
        return make_uniq<VortexMultiFileReaderInterface>();
    }
};

void mff_pushdown_complex_filter(ClientContext &context,
                                 LogicalGet &get,
                                 FunctionData *bind_data_p,
                                 vector<unique_ptr<Expression>> &filters) {
    auto &data = bind_data_p->Cast<MultiFileBindData>();

    MultiFilePushdownInfo info(get);
    auto new_list =
        data.multi_file_reader->ComplexFilterPushdown(context, *data.file_list, data.file_options, info, filters);

    if (new_list) {
        data.file_list = std::move(new_list);
        MultiFileReader::PruneReaders(data, *data.file_list);
    }

    auto &vortex_bind = data.bind_data->Cast<VortexMultiFileBindData>();
    duckdb_vx_error error_out = nullptr;
    for (auto iter = filters.begin(); iter != filters.end();) {
        duckdb_vx_expr ffi_expr = reinterpret_cast<duckdb_vx_expr>(iter->get());
        const bool pushed = vortex_bind.vtab.pushdown_complex_filter(vortex_bind.handle, ffi_expr, &error_out);
        if (error_out) {
            throw BinderException(IntoErrString(error_out));
        }
        iter = pushed ? filters.erase(iter) : std::next(iter);
    }
}

unique_ptr<BaseStatistics> mff_statistics(ClientContext &context, const FunctionData *bind_data_p,
                                          column_t column_index) {
    auto stats = MultiFileFunction<VortexMultiFileFunctionOp>::MultiFileScanStats(context, bind_data_p,
                                                                                  column_index);
    if (stats) {
        return stats;
    }

    auto &data = bind_data_p->Cast<MultiFileBindData>();
    if (IsVirtualColumn(column_index) || !data.bind_data || !data.file_list) {
        return nullptr;
    }
    if (data.file_list->GetExpandResult() == FileExpandResult::MULTIPLE_FILES) {
        return nullptr;
    }
    if (column_index >= data.names.size() || column_index >= data.types.size()) {
        return nullptr;
    }

    auto &vortex_bind = data.bind_data->Cast<VortexMultiFileBindData>();
    if (!vortex_bind.vtab.statistics) {
        return nullptr;
    }

    duckdb_column_statistics raw_stats = {};
    const auto &name = data.names[column_index];
    if (!vortex_bind.vtab.statistics(vortex_bind.handle, name.c_str(), name.size(), &raw_stats)) {
        return nullptr;
    }
    return ColumnStatsFrom(raw_stats, data.types[column_index]);
}

vector<PartitionStatistics> mff_get_partition_stats(ClientContext &context, GetPartitionStatsInput &input) {
    vector<PartitionStatistics> result;
    if (!input.bind_data) {
        return result;
    }

    auto &data = input.bind_data->Cast<MultiFileBindData>();
    if (!data.bind_data || !data.file_list) {
        return result;
    }

    auto &vortex_bind = data.bind_data->Cast<VortexMultiFileBindData>();
    if (!vortex_bind.vtab.partition_stats) {
        return result;
    }

    auto ctx = reinterpret_cast<duckdb_client_context>(&context);
    idx_t row_start = 0;
    for (const auto &file : data.file_list->Files()) {
        duckdb_vx_mff_partition_stats ffi_stats = {};
        duckdb_vx_error error_out = nullptr;
        const bool found = vortex_bind.vtab.partition_stats(ctx, vortex_bind.handle, file.path.c_str(),
                                                            file.path.size(), &ffi_stats, &error_out);
        if (error_out) {
            throw IOException(IntoErrString(error_out));
        }
        if (!found) {
            return {};
        }

        PartitionStatistics stats;
        stats.row_start = optional_idx(row_start);
        stats.count = static_cast<idx_t>(ffi_stats.row_count);
        stats.count_type = CountType::COUNT_EXACT;
        result.push_back(std::move(stats));
        row_start += static_cast<idx_t>(ffi_stats.row_count);
    }
    return result;
}

} // namespace

extern "C" void duckdb_vx_mff_schema_writer_add_column(duckdb_vx_mff_schema_writer writer,
                                                       const char *name,
                                                       size_t name_len,
                                                       duckdb_logical_type type) {
    struct SchemaWriter {
        vector<string> &names;
        vector<LogicalType> &types;
    };
    auto &w = *reinterpret_cast<SchemaWriter *>(writer);
    w.names.emplace_back(name, name_len);
    w.types.emplace_back(*reinterpret_cast<LogicalType *>(type));
}

extern "C" duckdb_state duckdb_vx_mff_register(duckdb_database ffi_db, const duckdb_vx_mff_vtab_t *vtab) {
    D_ASSERT(ffi_db);
    D_ASSERT(vtab);

    const auto &wrapper = *reinterpret_cast<DatabaseWrapper *>(ffi_db);
    auto &db = *wrapper.database->instance;

    // The catalog-owned TableFunctionInfo carries the vtab copy that each bind
    // resolves through InitializeOptions. Keeping it there avoids a shared
    // global pointer across databases/tests.
    auto info = make_shared_ptr<VortexMultiFileFunctionInfo>(*vtab);

    MultiFileFunction<VortexMultiFileFunctionOp> mff(vtab->name);
    mff.function_info = info;
    mff.statistics = mff_statistics;
    mff.filter_pushdown = vtab->filter_pushdown;
    mff.filter_prune = vtab->filter_prune;
    mff.pushdown_complex_filter = mff_pushdown_complex_filter;
    mff.get_partition_stats = mff_get_partition_stats;
    mff.late_materialization = true;
    mff.get_row_id_columns = [](ClientContext &, optional_ptr<FunctionData>) -> vector<column_t> {
        return {COLUMN_IDENTIFIER_FILE_INDEX, COLUMN_IDENTIFIER_FILE_ROW_NUMBER};
    };

    // Bind-time EXPLAIN output. Adds keys like "Function", "Files",
    // "Projection", "Filters". MultiFileFunction also installs a
    // dynamic_to_string that lists files at scan time; we leave that as-is.
    mff.to_string = [](TableFunctionToStringInput &input) {
        InsertionOrderPreservingMap<string> result;
        const auto &bind = input.bind_data->Cast<MultiFileBindData>();
        const auto &vortex_bind = bind.bind_data->Cast<VortexMultiFileBindData>();
        auto map = reinterpret_cast<duckdb_vx_mff_string_map>(&result);
        vortex_bind.vtab.to_string(vortex_bind.handle, map);
        return result;
    };

    try {
        // CreateFunctionSet returns a TableFunctionSet that bundles both the
        // single-VARCHAR and LIST(VARCHAR) overloads (matching read_parquet's
        // shape). This is what enables `read_vortex_v2(['a.vortex','b.vortex'])`.
        auto function_set = MultiFileReader::CreateFunctionSet(mff);
        auto &system_catalog = Catalog::GetSystemCatalog(db);
        auto data = CatalogTransaction::GetSystemTransaction(db);
        CreateTableFunctionInfo tf_info(function_set);
        tf_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        system_catalog.CreateFunction(data, tf_info);
    } catch (const std::exception &e) {
        ErrorData err(e);
        DUCKDB_LOG_ERROR(db, "Failed to create Vortex multi-file function:\t" + err.Message());
        return DuckDBError;
    }
    return DuckDBSuccess;
}
