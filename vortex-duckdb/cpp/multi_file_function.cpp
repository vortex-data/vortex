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
#include <utility>

DUCKDB_INCLUDES_BEGIN
#include "duckdb.h"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/multi_file/base_file_reader.hpp"
#include "duckdb/common/multi_file/multi_file_data.hpp"
#include "duckdb/common/multi_file/multi_file_function.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
DUCKDB_INCLUDES_END

using namespace duckdb;
using vortex::IntoErrString;

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

    const duckdb_vx_mff_vtab_t &vtab;
    duckdb_vx_mff_bind_data handle;
};

/**
 * Global state for a single multi-file scan. Distinct from MultiFileGlobalState
 * (which DuckDB owns); this is the *interface*-owned global state slot.
 */
class VortexInterfaceGlobalState : public GlobalTableFunctionState {
public:
    VortexInterfaceGlobalState(const duckdb_vx_mff_vtab_t &vtab, duckdb_vx_mff_global handle)
        : vtab(vtab), handle(handle) {
    }
    ~VortexInterfaceGlobalState() override {
        if (handle) {
            vtab.free_global(handle);
        }
    }

    const duckdb_vx_mff_vtab_t &vtab;
    duckdb_vx_mff_global handle;
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

    void PrepareReader(ClientContext &, GlobalTableFunctionState &) override {
        // Translate the multi-file column ids into projected column names, then
        // hand DuckDB's TableFilterSet through to Rust as a borrow. The reader
        // stores the resulting projection/filter so it can apply them when the
        // scan starts.
        std::vector<duckdb_vx_mff_column> ffi_proj;
        ffi_proj.reserve(column_ids.size());
        for (idx_t i = 0; i < column_ids.size(); i++) {
            auto local_id = column_ids[MultiFileLocalIndex(i)];
            // `local_id` is an index into our local `columns` schema. The
            // multi-file reader only routes physical columns here; virtual
            // columns are handled separately and never reach this list.
            const auto &col = columns[local_id];
            ffi_proj.push_back({col.name.c_str(), col.name.size()});
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
        duckdb_column_statistics stats = {};
        if (!vtab.get_statistics(handle, name.c_str(), name.size(), &stats)) {
            return nullptr;
        }
        // Materialize into a BaseStatistics matching the column's type. We reuse the
        // logic in cpp/table_function.cpp by constructing an Unknown stats and setting
        // the bits we have. Because we don't carry the type here (BaseFileReader has
        // it via columns), look it up.
        for (auto &col : columns) {
            if (col.name != name) {
                continue;
            }
            BaseStatistics out = BaseStatistics::CreateUnknown(col.type);
            if (stats.min) {
                auto min_val = *reinterpret_cast<Value *>(stats.min);
                duckdb_destroy_value(&stats.min);
                if (col.type.IsNumeric()) {
                    NumericStats::SetMin(out, min_val);
                } else if (col.type.id() == LogicalTypeId::VARCHAR ||
                           col.type.id() == LogicalTypeId::BLOB) {
                    StringStats::SetMin(out, StringValue::Get(min_val));
                }
            }
            if (stats.max) {
                auto max_val = *reinterpret_cast<Value *>(stats.max);
                duckdb_destroy_value(&stats.max);
                if (col.type.IsNumeric()) {
                    NumericStats::SetMax(out, max_val);
                } else if (col.type.id() == LogicalTypeId::VARCHAR ||
                           col.type.id() == LogicalTypeId::BLOB) {
                    StringStats::SetMax(out, StringValue::Get(max_val));
                }
            }
            if (!stats.has_null) {
                out.Set(StatsInfo::CANNOT_HAVE_NULL_VALUES);
            }
            return out.ToUnique();
        }
        return nullptr;
    }

    double GetProgressInFile(ClientContext &) override {
        return vtab.progress_in_file(handle);
    }

private:
    const duckdb_vx_mff_vtab_t &vtab;
    duckdb_vx_mff_reader handle;
};

/**
 * Cross-file orchestrator. Implements only the methods the basic scan pipeline
 * needs; everything else (hive partitioning, COPY, union-by-name, virtual cols)
 * defaults to the base-class behaviour.
 */
class VortexMultiFileReaderInterface : public MultiFileReaderInterface {
public:
    explicit VortexMultiFileReaderInterface(const duckdb_vx_mff_vtab_t &vtab_p) : vtab(vtab_p) {
    }

    unique_ptr<BaseFileReaderOptions> InitializeOptions(ClientContext &context,
                                                        optional_ptr<TableFunctionInfo>) override {
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
                                                                MultiFileGlobalState &) override {
        auto &vortex_bind = bind_data.bind_data->Cast<VortexMultiFileBindData>();
        duckdb_vx_error error_out = nullptr;
        auto ctx = reinterpret_cast<duckdb_client_context>(&context);
        auto handle = vtab.init_global(ctx, vortex_bind.handle, &error_out);
        if (error_out) {
            throw BinderException(IntoErrString(error_out));
        }
        return make_uniq<VortexInterfaceGlobalState>(vtab, handle);
    }

    unique_ptr<LocalTableFunctionState> InitializeLocalState(ExecutionContext &,
                                                              GlobalTableFunctionState &gstate) override {
        auto &g = gstate.Cast<VortexInterfaceGlobalState>();
        auto handle = vtab.init_local(g.handle);
        return make_uniq<VortexInterfaceLocalState>(vtab, handle);
    }

    shared_ptr<BaseFileReader> CreateReader(ClientContext &, GlobalTableFunctionState &, BaseUnionData &,
                                            const MultiFileBindData &) override {
        // UNION BY NAME path - not supported yet.
        throw NotImplementedException("UNION BY NAME is not yet supported by the Vortex multi-file function");
    }

    shared_ptr<BaseFileReader> CreateReader(ClientContext &context, GlobalTableFunctionState &gstate,
                                            const OpenFileInfo &file, idx_t file_idx,
                                            const MultiFileBindData &bind_data) override {
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
        return make_uniq<VortexMultiFileReaderInterface>(vtab);
    }

private:
    const duckdb_vx_mff_vtab_t &vtab;
};

/**
 * The OP type required by MultiFileFunction<OP>. Holds a pointer to the vtab so
 * CreateInterface can construct a VortexMultiFileReaderInterface bound to it.
 */
struct VortexMultiFileFunctionOp {
    static const duckdb_vx_mff_vtab_t *current_vtab;

    static unique_ptr<MultiFileReaderInterface> CreateInterface(ClientContext &) {
        if (!current_vtab) {
            throw InternalException("VortexMultiFileFunctionOp::CreateInterface called without a registered vtab");
        }
        return make_uniq<VortexMultiFileReaderInterface>(*current_vtab);
    }
};

const duckdb_vx_mff_vtab_t *VortexMultiFileFunctionOp::current_vtab = nullptr;

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

    // Capture the vtab pointer so MultiFileFunction<OP>::MultiFileBind can find it
    // when DuckDB re-enters CreateInterface during bind. The catalog will also
    // hold a copy via TableFunctionInfo so this stays alive for the lifetime of
    // the registered function.
    auto info = make_shared_ptr<VortexMultiFileFunctionInfo>(*vtab);
    VortexMultiFileFunctionOp::current_vtab = &info->vtab;

    MultiFileFunction<VortexMultiFileFunctionOp> mff(vtab->name);
    mff.function_info = info;

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

    // Late materialization is not enabled yet: it requires the per-file reader
    // to accept AddVirtualColumn calls for file_index / file_row_number and
    // produce those columns at scan time. Until that's wired (see follow-up),
    // we leave `late_materialization = false` so DuckDB doesn't request
    // virtual columns through paths the reader can't satisfy.

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
