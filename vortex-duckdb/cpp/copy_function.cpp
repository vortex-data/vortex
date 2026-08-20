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
    return make_uniq<VortexCopyBindData>(std::move(cdata));
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

void copy_to_finalize(ClientContext &, FunctionData &, GlobalFunctionData &gstate) {
    const VortexCopyGlobalState &global = gstate.Cast<VortexCopyGlobalState>();

    void *const ffi_global = global.ffi_global->DataPtr();
    duckdb_vx_error error_out = nullptr;
    duckdb_copy_function_copy_to_finalize(ffi_global, &error_out);
    if (error_out) {
        throw ExecutorException(IntoErrString(error_out));
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
