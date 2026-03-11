// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx.h"
#include "duckdb_vx/data.hpp"
#include "duckdb_vx/error.hpp"

#include "duckdb_vx/duckdb_diagnostics.h"
DUCKDB_INCLUDES_BEGIN
#include "duckdb/function/copy_function.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/parser/parsed_data/create_copy_function_info.hpp"
DUCKDB_INCLUDES_END

using namespace duckdb;

namespace vortex {

struct CCopyBindData final : TableFunctionData {
    CCopyBindData(const duckdb_vx_copy_func_vtab_t vtab_p, unique_ptr<CData> ffi_data_p)
        : vtab(vtab_p), ffi_data(std::move(ffi_data_p)) {
    }

    const duckdb_vx_copy_func_vtab_t vtab;
    unique_ptr<CData> ffi_data;
};

struct CCopyGlobalData final : GlobalFunctionData {
    explicit CCopyGlobalData(unique_ptr<CData> ffi_data_p) : ffi_data(std::move(ffi_data_p)) {
    }

    unique_ptr<CData> ffi_data;
};

struct CCopyLocalData final : LocalFunctionData {
    explicit CCopyLocalData(unique_ptr<CData> ffi_data_p) : ffi_data(std::move(ffi_data_p)) {
    }

    unique_ptr<CData> ffi_data;
};

static duckdb_vx_copy_func_vtab_t copy_vtab_one;

unique_ptr<FunctionData> c_bind_one(ClientContext & /*context*/,
                                    CopyFunctionBindInput &info,
                                    const vector<string> &column_names,
                                    const vector<LogicalType> &column_types) {

    auto c_column_names = vector<char *>();
    c_column_names.reserve(column_names.size());
    for (const auto &col_id : column_names) {
        c_column_names.push_back(const_cast<char *>(col_id.c_str()));
    }

    auto c_column_types = vector<duckdb_logical_type>();
    c_column_types.reserve(c_column_types.size());
    for (auto &col_type : column_types) {
        c_column_types.push_back(reinterpret_cast<duckdb_logical_type>(const_cast<LogicalType *>(&col_type)));
    }

    duckdb_vx_error error_out = nullptr;
    // TODO(myrrc): do we pass ownership of c_column_names in bind?
    // If yes, it's a UB as we'd double delete on function return
    auto ffi_bind_data = copy_vtab_one.bind(reinterpret_cast<duckdb_vx_copy_func_bind_input>(&info),
                                            c_column_names.data(),
                                            c_column_names.size(),
                                            c_column_types.data(),
                                            c_column_types.size(),
                                            &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }

    return make_uniq<CCopyBindData>(
        // This should only be filled out once
        copy_vtab_one,
        unique_ptr<CData>(reinterpret_cast<CData *>(ffi_bind_data)));
}

unique_ptr<GlobalFunctionData>
c_init_global(ClientContext &context, FunctionData &bind_data, const string &file_path) {
    auto &bind = bind_data.Cast<CCopyBindData>();
    duckdb_vx_error error_out = nullptr;
    auto global_data = bind.vtab.init_global(reinterpret_cast<duckdb_client_context>(&context),
                                             bind.ffi_data->DataPtr(),
                                             file_path.c_str(),
                                             &error_out);
    if (error_out) {
        throw ExecutorException(IntoErrString(error_out));
    }

    return make_uniq<CCopyGlobalData>(unique_ptr<CData>(reinterpret_cast<CData *>(global_data)));
}

unique_ptr<LocalFunctionData> c_init_local(ExecutionContext & /*context*/, FunctionData &bind_data) {
    auto &bind = bind_data.Cast<CCopyBindData>();
    duckdb_vx_error error_out = nullptr;
    auto data = bind.vtab.init_local(bind.ffi_data->DataPtr(), &error_out);
    if (error_out) {
        throw ExecutorException(IntoErrString(error_out));
    }

    return make_uniq<CCopyLocalData>(unique_ptr<CData>(reinterpret_cast<CData *>(data)));
}

void c_copy_to_sink(ExecutionContext & /*context*/,
                    FunctionData &bind_data,
                    GlobalFunctionData &gstate,
                    LocalFunctionData &lstate,
                    DataChunk &input) {
    auto &bind = bind_data.Cast<CCopyBindData>();
    auto &global = gstate.Cast<CCopyGlobalData>();
    auto &local = lstate.Cast<CCopyLocalData>();
    duckdb_vx_error error_out = nullptr;
    bind.vtab.copy_to_sink(bind.ffi_data->DataPtr(),
                           global.ffi_data->DataPtr(),
                           local.ffi_data->DataPtr(),
                           reinterpret_cast<duckdb_data_chunk>(&input),
                           &error_out);
    if (error_out) {
        throw ExecutorException(IntoErrString(error_out));
    }
}

void copy_to_finalize(ClientContext & /*context*/, FunctionData &bind_data, GlobalFunctionData &gstate) {
    auto &bind = bind_data.Cast<CCopyBindData>();
    auto &global = gstate.Cast<CCopyGlobalData>();
    duckdb_vx_error error_out = nullptr;
    bind.vtab.copy_to_finalize(bind.ffi_data->DataPtr(), global.ffi_data->DataPtr(), &error_out);
    if (error_out) {
        throw ExecutorException(IntoErrString(error_out));
    }
}

extern "C" duckdb_vx_copy_func_vtab_t *get_vtab_one() {
    return &copy_vtab_one;
}

extern "C" duckdb_state duckdb_vx_copy_func_register_vtab_one(duckdb_database ffi_db) {
    if (!ffi_db) {
        return DuckDBError;
    }

    auto wrapper = reinterpret_cast<duckdb::DatabaseWrapper *>(ffi_db);
    auto db = wrapper->database->instance;
    auto copy_function = CopyFunction(copy_vtab_one.name);

    copy_function.copy_to_bind = c_bind_one;
    copy_function.copy_to_initialize_global = c_init_global;
    copy_function.copy_to_initialize_local = c_init_local;

    copy_function.copy_to_sink = c_copy_to_sink;
    copy_function.copy_to_finalize = copy_to_finalize;
    copy_function.extension = copy_vtab_one.extension;

    // TODO(joe): expose this via c our api
    copy_function.execution_mode = [](bool /*preserve_insertion_order*/, bool /*supports_batch_index*/) {
        return CopyFunctionExecutionMode::REGULAR_COPY_TO_FILE;
    };
    // TODO(joe): handle parameters as in table_function

    try {
        auto &system_catalog = Catalog::GetSystemCatalog(*db);
        auto data = CatalogTransaction::GetSystemTransaction(*db);
        CreateCopyFunctionInfo copy_info(std::move(copy_function));
        system_catalog.CreateCopyFunction(data, copy_info);
    } catch (...) {
        return DuckDBError;
    }
    return DuckDBSuccess;
}

} // namespace vortex
