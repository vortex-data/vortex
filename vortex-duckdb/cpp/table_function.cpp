#include "duckdb.h"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/function/table_function.hpp"

#include "duckdb_vx.h"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb_vx/data.hpp"
#include "duckdb_vx/error.hpp"

using namespace duckdb;

namespace vortex {

struct CTableFunctionInfo final : TableFunctionInfo {
    explicit CTableFunctionInfo(const duckdb_vx_tfunc_vtab_t &vtab) : vtab(vtab) {
    }

    duckdb_vx_tfunc_vtab_t vtab;
};

struct CTableBindData final : TableFunctionData {
    CTableBindData(unique_ptr<CTableFunctionInfo> info_p, unique_ptr<vortex::CData> ffi_data_p)
        : info(std::move(info_p)), ffi_data(std::move(ffi_data_p)) {
    }

    unique_ptr<FunctionData> Copy() const override {
        assert(info->vtab.bind_data_clone != nullptr);

        duckdb_vx_error error_out = nullptr;
        const auto copied_ffi_data = info->vtab.bind_data_clone(ffi_data->DataPtr(), &error_out);
        if (error_out) {
            throw BinderException(IntoErrString(error_out));
        }
        return make_uniq<CTableBindData>(
            make_uniq<CTableFunctionInfo>(info->vtab),
            unique_ptr<vortex::CData>(reinterpret_cast<vortex::CData *>(copied_ffi_data)));
    }

    unique_ptr<CTableFunctionInfo> info;
    unique_ptr<vortex::CData> ffi_data;
};

struct CTableGlobalData final : GlobalTableFunctionState {
    explicit CTableGlobalData(unique_ptr<vortex::CData> ffi_data_p) : ffi_data(std::move(ffi_data_p)) {
    }

    unique_ptr<vortex::CData> ffi_data;

    idx_t MaxThreads() const override {
        return GlobalTableFunctionState::MAX_THREADS;
    }
};

struct CTableLocalData final : LocalTableFunctionState {
    explicit CTableLocalData(unique_ptr<vortex::CData> ffi_data_p) : ffi_data(std::move(ffi_data_p)) {
    }

    unique_ptr<vortex::CData> ffi_data;
};

/**
 * Result of the bind function encapsulates the output schema.
 */
struct CTableBindResult {
    vector<LogicalType> &return_types;
    vector<string> &names;
};

unique_ptr<FunctionData> c_bind(ClientContext &context, TableFunctionBindInput &input,
                                vector<LogicalType> &return_types, vector<string> &names) {
    const auto &info = input.table_function.function_info->Cast<CTableFunctionInfo>();

    // Setup bind info to pass into the callback.
    CTableBindResult result = {
        return_types,
        names,
    };

    duckdb_vx_error error_out = nullptr;
    auto ffi_bind_data = info.vtab.bind(reinterpret_cast<duckdb_vx_tfunc_bind_input>(&input),
                                        reinterpret_cast<duckdb_vx_tfunc_bind_result>(&result), &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }

    return make_uniq<CTableBindData>(
        make_uniq<CTableFunctionInfo>(info.vtab),
        unique_ptr<vortex::CData>(reinterpret_cast<vortex::CData *>(ffi_bind_data)));
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
    };

    duckdb_vx_error error_out = nullptr;
    auto ffi_global_data = bind.info->vtab.init_global(&ffi_input, &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }

    return make_uniq<CTableGlobalData>(
        unique_ptr<vortex::CData>(reinterpret_cast<vortex::CData *>(ffi_global_data)));
}

unique_ptr<LocalTableFunctionState> c_init_local(ExecutionContext &context, TableFunctionInitInput &input,
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
    };

    duckdb_vx_error error_out = nullptr;
    auto ffi_local_data = bind.info->vtab.init_local(&ffi_input, global_data, &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }

    return make_uniq<CTableLocalData>(
        unique_ptr<vortex::CData>(reinterpret_cast<vortex::CData *>(ffi_local_data)));
}

void c_function(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
    const auto &bind = input.bind_data->Cast<CTableBindData>();

    const auto bind_data = bind.ffi_data->DataPtr();
    auto global_data = input.global_state->Cast<CTableGlobalData>().ffi_data->DataPtr();
    auto local_data = input.local_state->Cast<CTableLocalData>().ffi_data->DataPtr();

    duckdb_vx_error error_out = nullptr;
    bind.info->vtab.function(bind_data, global_data, local_data, reinterpret_cast<duckdb_data_chunk>(&output),
                             &error_out);
    if (error_out) {
        throw InvalidInputException(IntoErrString(error_out));
    }
}

void c_pushdown_complex_filter(ClientContext &context, LogicalGet &get, FunctionData *bind_data,
                               vector<unique_ptr<Expression>> &filters) {
    if (filters.empty()) {
        return;
    }

    auto &bind = bind_data->Cast<CTableBindData>();

    for (auto iter = filters.begin(); iter != filters.end();) {
        duckdb_vx_error error_out = nullptr;
        auto pushed = bind.info->vtab.pushdown_complex_filter(
            bind_data->Cast<CTableBindData>().ffi_data->DataPtr(),
            reinterpret_cast<duckdb_vx_expr>(iter->get()), &error_out);
        if (error_out) {
            throw BinderException(IntoErrString(error_out));
        }

        if (pushed) {
            // If the pushdown complex filter returns true, we can remove the filter from the list.
            iter = filters.erase(iter);
        } else {
            ++iter;
        }
    }
}

extern "C" size_t duckdb_vx_tfunc_bind_input_get_parameter_count(duckdb_vx_tfunc_bind_input ffi_input) {
    if (!ffi_input) {
        return 0;
    }
    const auto input = reinterpret_cast<TableFunctionBindInput *>(ffi_input);
    return input->inputs.size();
}

extern "C" duckdb_value duckdb_vx_tfunc_bind_input_get_parameter(duckdb_vx_tfunc_bind_input ffi_input,
                                                                 size_t index) {
    if (!ffi_input || index >= duckdb_vx_tfunc_bind_input_get_parameter_count(ffi_input)) {
        return nullptr;
    }
    const auto info = reinterpret_cast<TableFunctionBindInput *>(ffi_input);
    return reinterpret_cast<duckdb_value>(new Value(info->inputs[index]));
}

extern "C" duckdb_value duckdb_vx_tfunc_bind_input_get_named_parameter(duckdb_vx_tfunc_bind_input ffi_input,
                                                                       const char *name) {
    if (!ffi_input || !name) {
        return nullptr;
    }
    const auto info = reinterpret_cast<TableFunctionBindInput *>(ffi_input);

    auto t = info->named_parameters.find(name);
    if (t == info->named_parameters.end()) {
        return nullptr;
    }
    return reinterpret_cast<duckdb_value>(new Value(t->second));
}

extern "C" void duckdb_vx_tfunc_bind_result_add_column(duckdb_vx_tfunc_bind_result ffi_result,
                                                       const char *name_str, size_t name_len,
                                                       duckdb_logical_type ffi_type) {
    if (!name_str || !ffi_type) {
        return;
    }
    const auto result = reinterpret_cast<CTableBindResult *>(ffi_result);
    const auto logical_type = reinterpret_cast<LogicalType *>(ffi_type);

    result->names.emplace_back(name_str, name_len);
    result->return_types.push_back(*logical_type);
}

extern "C" duckdb_state duckdb_vx_tfunc_register(duckdb_connection ffi_conn,
                                                 const duckdb_vx_tfunc_vtab_t *vtab) {
    if (!ffi_conn || !vtab) {
        return DuckDBError;
    }

    auto conn = reinterpret_cast<Connection *>(ffi_conn);
    auto tf = new TableFunction(vtab->name, {}, c_function, c_bind, c_init_global, c_init_local);

    tf->pushdown_complex_filter = c_pushdown_complex_filter;

    tf->projection_pushdown = vtab->projection_pushdown;
    tf->filter_pushdown = vtab->filter_pushdown;
    tf->filter_prune = vtab->filter_prune;
    tf->sampling_pushdown = vtab->sampling_pushdown;
    tf->late_materialization = vtab->late_materialization;

    // Set up the parameters
    for (size_t i = 0; i < vtab->parameter_count; i++) {
        auto logical_type = reinterpret_cast<LogicalType *>(vtab->parameters[i]);
        tf->arguments.push_back(*logical_type);
    }
    // And the named parameters
    for (size_t i = 0; i < vtab->named_parameter_count; i++) {
        auto logical_type = reinterpret_cast<LogicalType *>(vtab->named_parameter_types[i]);
        tf->named_parameters.insert({vtab->named_parameter_names[i], *logical_type});
    }

    // Assign the VTable to the function info so we can access it later to invoke the callbacks.
    tf->function_info = make_shared_ptr<CTableFunctionInfo>(*vtab);

    try {
        conn->context->RunFunctionInTransaction([&]() {
            auto &catalog = Catalog::GetSystemCatalog(*conn->context);
            CreateTableFunctionInfo tf_info(*tf);
            catalog.CreateTableFunction(*conn->context, tf_info);
        });
    } catch (...) {
        return DuckDBError;
    }
    return DuckDBSuccess;
}

} // namespace vortex
