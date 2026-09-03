// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "copy_function.hpp"
#include "data.hpp"
#include "error.hpp"
#include "vortex_duckdb.h"
#include "table_function.h"
#include "vortex.h"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/parser/keyword_helper.hpp"
#include "duckdb/parser/parsed_data/create_copy_function_info.hpp"

unique_ptr<FunctionData> copy_to_bind(ClientContext &,
                                      CopyFunctionBindInput &,
                                      const vector<string> &names,
                                      const vector<LogicalType> &types) {
    vector<const char *> ffi_names(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        ffi_names[i] = names[i].c_str();
    }

    vector<duckdb_logical_type> ffi_types(types.size());
    for (size_t i = 0; i < types.size(); ++i) {
        // duckdb C api doesn't allow passing const LogicalTypes. We never
        // modify input in copy function.
        ffi_types[i] = reinterpret_cast<duckdb_logical_type>(const_cast<LogicalType *>(&types[i]));
    }

    duckdb_vx_error error_out = nullptr;
    const duckdb_vx_data ffi_bind_data = duckdb_copy_function_copy_to_bind(ffi_names.data(),
                                                                           ffi_names.size(),
                                                                           ffi_types.data(),
                                                                           ffi_types.size(),
                                                                           &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }
    auto cdata = unique_ptr<CData>(reinterpret_cast<CData *>(ffi_bind_data));
    return make_uniq<VortexCopyBindData>(std::move(cdata), names);
}

unique_ptr<GlobalFunctionData>
copy_to_initialize_global(ClientContext &, FunctionData &bind_data, const string &file_path) {
    const VortexCopyBindData &bind = bind_data.Cast<VortexCopyBindData>();
    const void *const ffi_bind = bind.ffi_bind->DataPtr();

    duckdb_vx_error error_out = nullptr;
    const duckdb_vx_data ffi_global =
        duckdb_copy_function_copy_to_initialize_global(ffi_bind, file_path.c_str(), &error_out);
    if (error_out) {
        throw ExecutorException(IntoErrString(error_out));
    }

    auto cdata = unique_ptr<CData>(reinterpret_cast<CData *>(ffi_global));
    return make_uniq<VortexCopyGlobalState>(std::move(cdata));
}

void copy_to_sink(ExecutionContext &,
                  FunctionData &bind_data,
                  GlobalFunctionData &gstate,
                  LocalFunctionData &,
                  DataChunk &input) {
    const VortexCopyBindData &bind = bind_data.Cast<VortexCopyBindData>();
    const VortexCopyGlobalState &global = gstate.Cast<VortexCopyGlobalState>();

    const void *const ffi_bind = bind.ffi_bind->DataPtr();
    const void *const ffi_global = global.ffi_global->DataPtr();

    duckdb_data_chunk ffi_chunk = reinterpret_cast<duckdb_data_chunk>(&input);
    duckdb_vx_error error_out = nullptr;
    duckdb_copy_function_copy_to_sink(ffi_bind, ffi_global, ffi_chunk, &error_out);
    if (error_out) {
        throw ExecutorException(IntoErrString(error_out));
    }
}

// CopyToFileInfo::file_stats is owned by the operator's sink state, which outlives gstate.
void copy_to_get_written_statistics(ClientContext &,
                                    FunctionData &,
                                    GlobalFunctionData &gstate,
                                    CopyFunctionFileStatistics &statistics) {
    gstate.Cast<VortexCopyGlobalState>().written_stats = &statistics;
}

void copy_to_finalize(ClientContext &, FunctionData &bind_data, GlobalFunctionData &gstate) {
    auto &global = gstate.Cast<VortexCopyGlobalState>();
    void *const ffi_global = global.ffi_global->DataPtr();
    duckdb_vx_error error_out = nullptr;
    duckdb_copy_function_copy_to_finalize(ffi_global, &error_out);
    if (error_out) {
        throw ExecutorException(IntoErrString(error_out));
    }

    if (!global.written_stats) {
        return;
    }
    auto &names = bind_data.Cast<VortexCopyBindData>().column_names;
    duckdb_vx_written_file_statistics file_stats;
    if (!duckdb_copy_function_get_written_file_statistics(ffi_global, &file_stats)) {
        // Statistics were requested (written_stats is set) but the finished write produced none;
        // that is an internal inconsistency, not a silently empty result.
        throw InternalException("vortex COPY: written statistics were requested but not produced");
    }
    if (file_stats.num_columns != names.size()) {
        throw InternalException("vortex COPY: %llu statistics columns for %llu written columns",
                                file_stats.num_columns,
                                names.size());
    }
    D_ASSERT(global.written_stats != nullptr);
    global.written_stats->row_count = file_stats.row_count;
    global.written_stats->file_size_bytes = file_stats.file_size_bytes;
    global.written_stats->footer_size_bytes = Value::UBIGINT(file_stats.footer_size_bytes);
    // Keyed by top-level column name only. The vortex footer reports one statistics set per
    // top-level field, so nested struct/list leaf columns get no statistics here (unlike parquet,
    // which recurses to leaf paths). Flat tables are fully covered.
    for (idx_t i = 0; i < file_stats.num_columns; i++) {
        duckdb_vx_written_column_statistics col_stats {};
        duckdb_vx_error col_error = nullptr;
        if (!duckdb_copy_function_get_written_column_statistics(ffi_global, i, &col_stats, &col_error)) {
            if (col_error) {
                throw ExecutorException(IntoErrString(col_error));
            }
            throw InternalException("vortex COPY: no statistics for column %llu after finalize", i);
        }
        case_insensitive_map_t<Value> column;
        column["num_values"] = Value::UBIGINT(col_stats.num_values);
        if (col_stats.has_column_size) {
            column["column_size_bytes"] = Value::UBIGINT(col_stats.column_size_bytes);
        }
        if (col_stats.has_null_count) {
            column["null_count"] = Value::UBIGINT(col_stats.null_count);
        }
        if (col_stats.min) {
            column["min"] = Value(reinterpret_cast<Value *>(col_stats.min)->ToString());
            duckdb_destroy_value(&col_stats.min);
        }
        if (col_stats.max) {
            column["max"] = Value(reinterpret_cast<Value *>(col_stats.max)->ToString());
            duckdb_destroy_value(&col_stats.max);
        }
        if (col_stats.has_nan_stat) {
            column["has_nan"] = Value::BOOLEAN(col_stats.contains_nan);
        }
        // DuckLake keys column statistics by a quoted, dot-separated path (see
        // DuckLakeUtil::ParseQuotedList); match the parquet writer, which quotes each name.
        global.written_stats->column_statistics.emplace(KeywordHelper::WriteQuoted(names[i], '"'),
                                                        std::move(column));
    }
}

unique_ptr<PreparedBatchData> copy_to_prepare_batch(ClientContext &,
                                                    FunctionData &bind_data,
                                                    GlobalFunctionData &,
                                                    unique_ptr<ColumnDataCollection> collection) {
    const VortexCopyBindData &bind = bind_data.Cast<VortexCopyBindData>();

    const void *const ffi_bind = bind.ffi_bind->DataPtr();
    auto ffi_batch = unique_ptr<CData>(reinterpret_cast<CData *>(duckdb_copy_function_prepare_batch_new()));
    duckdb_vx_error error_out = nullptr;

    for (DataChunk &chunk : collection->Chunks()) {
        duckdb_data_chunk ffi_chunk = reinterpret_cast<duckdb_data_chunk>(&chunk);
        duckdb_copy_function_prepare_batch_push(ffi_bind, ffi_batch->DataPtr(), ffi_chunk, &error_out);
        if (error_out) {
            throw ExecutorException(IntoErrString(error_out));
        }
    }
    return make_uniq<VortexCopyPreparedBatchData>(std::move(ffi_batch));
}

void copy_to_flush_batch(ClientContext &,
                         FunctionData &,
                         GlobalFunctionData &gstate,
                         PreparedBatchData &batch_data) {
    const VortexCopyGlobalState &global = gstate.Cast<VortexCopyGlobalState>();
    const VortexCopyPreparedBatchData &batch = batch_data.Cast<VortexCopyPreparedBatchData>();

    const void *const ffi_global = global.ffi_global->DataPtr();
    const void *const ffi_batch = batch.ffi_copy_prepared->DataPtr();
    duckdb_vx_error error_out = nullptr;
    duckdb_copy_function_flush_batch(ffi_global, ffi_batch, &error_out);
    if (error_out) {
        throw ExecutorException(IntoErrString(error_out));
    }
}

extern "C" duckdb_state duckdb_vx_register_copy_function(duckdb_database ffi_db) {
    D_ASSERT(ffi_db);
    const DatabaseWrapper &wrapper = *reinterpret_cast<DatabaseWrapper *>(ffi_db);
    DatabaseInstance &db = *wrapper.database->instance;

    CopyFunction fn("vortex");
    fn.copy_to_bind = copy_to_bind;
    fn.copy_to_initialize_global = copy_to_initialize_global;
    // required by duckdb
    fn.copy_to_initialize_local = [](auto &, auto &) {
        return make_uniq<LocalFunctionData>();
    };
    fn.copy_to_sink = copy_to_sink;
    // required by duckdb for PARTITION_BY
    fn.copy_to_combine = [](ExecutionContext &, FunctionData &, GlobalFunctionData &, LocalFunctionData &) {
    };
    fn.copy_to_finalize = copy_to_finalize;
    fn.prepare_batch = copy_to_prepare_batch;
    fn.flush_batch = copy_to_flush_batch;
    fn.copy_to_get_written_statistics = copy_to_get_written_statistics;
    fn.extension = "vortex";

    fn.execution_mode = [](bool preserve_insertion_order, bool supports_batch_index) {
        using enum CopyFunctionExecutionMode;
        if (!preserve_insertion_order) {
            return PARALLEL_COPY_TO_FILE;
        }
        if (supports_batch_index) {
            return BATCH_COPY_TO_FILE;
        }
        return REGULAR_COPY_TO_FILE;
    };

    try {
        Catalog &system_catalog = Catalog::GetSystemCatalog(db);
        CatalogTransaction data = CatalogTransaction::GetSystemTransaction(db);
        CreateCopyFunctionInfo copy_info(std::move(fn));
        system_catalog.CreateCopyFunction(data, copy_info);
    } catch (const std::exception &e) {
        ErrorData data(e);
        DUCKDB_LOG_ERROR(db, "Failed to create Vortex copy function:\t" + data.Message());
        return DuckDBError;
    }
    return DuckDBSuccess;
}
