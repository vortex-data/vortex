// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "duckdb_vx/data.hpp"
#include "duckdb_vx/error.hpp"
#include "duckdb_vx/table_function.h"
#include "vortex.h"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/parser/parsed_data/create_copy_function_info.hpp"

using namespace duckdb;
using vortex::CData;
using vortex::IntoErrString;

struct CopyBindData final : TableFunctionData {
    CopyBindData(unique_ptr<CData> ffi_data) : ffi_data(std::move(ffi_data)) {
    }
    unique_ptr<CData> ffi_data;
};

struct CopyGlobalData final : GlobalFunctionData {
    CopyGlobalData(unique_ptr<CData> ffi_data) : ffi_data(std::move(ffi_data)) {
    }

    unique_ptr<CData> ffi_data;
};

unique_ptr<FunctionData> copy_to_bind(ClientContext &,
                                      CopyFunctionBindInput &,
                                      const vector<string> &column_names,
                                      const vector<LogicalType> &column_types) {
    vector<const char *> ffi_column_names(column_names.size());
    for (size_t i = 0; i < column_names.size(); ++i) {
        ffi_column_names[i] = column_names[i].c_str();
    }

    vector<duckdb_logical_type> ffi_column_types(column_types.size());
    for (size_t i = 0; i < column_types.size(); ++i) {
        // duckdb C api doesn't allow passing const LogicalTypes. We never
        // modify input in copy function.
        ffi_column_types[i] =
            reinterpret_cast<duckdb_logical_type>(const_cast<LogicalType *>(&column_types[i]));
    }

    duckdb_vx_error error_out = nullptr;
    const duckdb_vx_data ffi_bind_data = duckdb_copy_function_copy_to_bind(ffi_column_names.data(),
                                                                           ffi_column_names.size(),
                                                                           ffi_column_types.data(),
                                                                           ffi_column_types.size(),
                                                                           &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }
    auto cdata = unique_ptr<CData>(reinterpret_cast<CData *>(ffi_bind_data));
    return make_uniq<CopyBindData>(std::move(cdata));
}

unique_ptr<GlobalFunctionData>
copy_to_initialize_global(ClientContext &, FunctionData &bind_data, const string &file_path) {
    void *const ffi_bind = bind_data.Cast<CopyBindData>().ffi_data->DataPtr();

    duckdb_vx_error error_out = nullptr;
    const duckdb_vx_data ffi_global =
        duckdb_copy_function_copy_to_initialize_global(ffi_bind, file_path.c_str(), &error_out);
    if (error_out) {
        throw ExecutorException(IntoErrString(error_out));
    }

    auto cdata = unique_ptr<CData>(reinterpret_cast<CData *>(ffi_global));
    return make_uniq<CopyGlobalData>(std::move(cdata));
}

void copy_to_sink(ExecutionContext &,
                  FunctionData &bind_data,
                  GlobalFunctionData &gstate,
                  LocalFunctionData &,
                  DataChunk &input) {
    void *const ffi_bind = bind_data.Cast<CopyBindData>().ffi_data->DataPtr();
    void *const ffi_global = gstate.Cast<CopyGlobalData>().ffi_data->DataPtr();
    auto ffi_chunk = reinterpret_cast<duckdb_data_chunk>(&input);
    duckdb_vx_error error_out = nullptr;
    duckdb_copy_function_copy_to_sink(ffi_bind, ffi_global, ffi_chunk, &error_out);
    if (error_out) {
        throw ExecutorException(IntoErrString(error_out));
    }
}

void copy_to_finalize(ClientContext &, FunctionData &, GlobalFunctionData &gstate) {
    void *const ffi_global = gstate.Cast<CopyGlobalData>().ffi_data->DataPtr();
    duckdb_vx_error error_out = nullptr;
    duckdb_copy_function_copy_to_finalize(ffi_global, &error_out);
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
    fn.copy_to_initialize_local = [](auto &, auto &) {
        return make_uniq<LocalFunctionData>();
    };
    fn.copy_to_sink = copy_to_sink;
    fn.copy_to_finalize = copy_to_finalize;
    fn.extension = "vortex";

    // TODO(joe): expose this via c our api
    fn.execution_mode = [](bool, bool) {
        return CopyFunctionExecutionMode::REGULAR_COPY_TO_FILE;
    };
    // TODO(joe): handle parameters as in table_function

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
