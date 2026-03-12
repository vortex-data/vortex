// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx/duckdb_diagnostics.h"

DUCKDB_INCLUDES_BEGIN
#include "duckdb.h"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/insertion_order_preserving_map.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
DUCKDB_INCLUDES_END

#include "duckdb_vx.h"
#include "duckdb_vx/data.hpp"
#include "duckdb_vx/error.hpp"

using namespace duckdb;

namespace vortex {
struct CTableFunctionInfo final : TableFunctionInfo {
    explicit CTableFunctionInfo(const duckdb_vx_tfunc_vtab_t &vtab)
        : vtab(vtab), max_threads(vtab.max_threads) {
    }

    duckdb_vx_tfunc_vtab_t vtab;
    idx_t max_threads;
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
        return make_uniq<CTableBindData>(make_uniq<CTableFunctionInfo>(info->vtab),
                                         unique_ptr<CData>(reinterpret_cast<CData *>(copied_ffi_data)));
    }

    unique_ptr<CTableFunctionInfo> info;
    unique_ptr<CData> ffi_data;
};

struct CTableGlobalData final : GlobalTableFunctionState {
    explicit CTableGlobalData(unique_ptr<vortex::CData> ffi_data_p, idx_t max_threads_p)
        : ffi_data(std::move(ffi_data_p)), max_threads(max_threads_p) {
    }

    unique_ptr<vortex::CData> ffi_data;
    idx_t max_threads;

    idx_t MaxThreads() const override {
        return max_threads;
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
                                     unique_ptr<CData>(reinterpret_cast<CData *>(ffi_bind_data)));
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

    return make_uniq<CTableGlobalData>(
        unique_ptr<vortex::CData>(reinterpret_cast<vortex::CData *>(ffi_global_data)),
        bind.info->max_threads);
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

    return make_uniq<CTableLocalData>(
        unique_ptr<vortex::CData>(reinterpret_cast<vortex::CData *>(ffi_local_data)));
}

void c_function(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
    const auto &bind = input.bind_data->Cast<CTableBindData>();

    auto ctx = reinterpret_cast<duckdb_client_context>(&context);
    const auto bind_data = bind.ffi_data->DataPtr();
    auto global_data = input.global_state->Cast<CTableGlobalData>().ffi_data->DataPtr();
    auto local_data = input.local_state->Cast<CTableLocalData>().ffi_data->DataPtr();

    duckdb_vx_error error_out = nullptr;
    bind.info->vtab.function(ctx,
                             bind_data,
                             global_data,
                             local_data,
                             reinterpret_cast<duckdb_data_chunk>(&output),
                             &error_out);
    if (error_out) {
        throw InvalidInputException(IntoErrString(error_out));
    }
}

void c_pushdown_complex_filter(ClientContext & /*context*/,
                               LogicalGet & /*get*/,
                               FunctionData *bind_data,
                               vector<unique_ptr<Expression>> &filters) {
    if (filters.empty()) {
        return;
    }

    auto &bind = bind_data->Cast<CTableBindData>();

    for (auto iter = filters.begin(); iter != filters.end();) {
        duckdb_vx_error error_out = nullptr;
        auto pushed =
            bind.info->vtab.pushdown_complex_filter(bind_data->Cast<CTableBindData>().ffi_data->DataPtr(),
                                                    reinterpret_cast<duckdb_vx_expr>(iter->get()),
                                                    &error_out);
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

unique_ptr<NodeStatistics> c_cardinality(ClientContext & /*context*/, const FunctionData *bind_data) {
    auto &bind = bind_data->Cast<CTableBindData>();

    duckdb_vx_node_statistics node_stats_out = {
        .estimated_cardinality = 0,
        .max_cardinality = 0,
        .has_estimated_cardinality = false,
        .has_max_cardinality = false,
    };
    bind.info->vtab.cardinality(bind_data->Cast<CTableBindData>().ffi_data->DataPtr(), &node_stats_out);

    auto stats = make_uniq<NodeStatistics>();
    stats->has_estimated_cardinality = node_stats_out.has_estimated_cardinality;
    stats->estimated_cardinality = node_stats_out.estimated_cardinality;
    stats->has_max_cardinality = node_stats_out.has_max_cardinality;
    stats->max_cardinality = node_stats_out.max_cardinality;

    return stats;
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
    auto value = duckdb::make_uniq<Value>(t->second);
    return reinterpret_cast<duckdb_value>(value.release());
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

virtual_column_map_t c_get_virtual_columns(ClientContext & /*context*/,
                                           optional_ptr<FunctionData> bind_data) {
    auto &bind = bind_data->Cast<CTableBindData>();

    auto result = virtual_column_map_t();
    bind.info->vtab.get_virtual_columns(bind_data->Cast<CTableBindData>().ffi_data->DataPtr(),
                                        reinterpret_cast<duckdb_vx_tfunc_virtual_cols_result>(&result));
    return result;
}

extern "C" void duckdb_vx_tfunc_virtual_columns_push(duckdb_vx_tfunc_virtual_cols_result ffi_result,
                                                     idx_t column_idx,
                                                     const char *name_str,
                                                     size_t name_len,
                                                     duckdb_logical_type ffi_type) {
    if (!ffi_result || !name_str || !ffi_type) {
        return;
    }

    auto result = reinterpret_cast<virtual_column_map_t *>(ffi_result);
    const auto logical_type = reinterpret_cast<LogicalType *>(ffi_type);
    const auto name = string(name_str, name_len);

    auto table_col = TableColumn(std::move(name), *logical_type);
    result->emplace(column_idx, std::move(table_col));
}

OperatorPartitionData c_get_partition_data(ClientContext & /*context*/,
                                           TableFunctionGetPartitionInput &input) {
    if (input.partition_info.RequiresPartitionColumns()) {
        throw InternalException("TableScan::GetPartitionData: partition columns not supported");
    }
    auto &bind = input.bind_data->Cast<CTableBindData>();
    auto &global = input.global_state->Cast<CTableGlobalData>();
    auto &local = input.local_state->Cast<CTableLocalData>();

    duckdb_vx_error error_out = nullptr;
    auto index = bind.info->vtab.get_partition_data(bind.ffi_data->DataPtr(),
                                                    global.ffi_data->DataPtr(),
                                                    local.ffi_data->DataPtr(),
                                                    &error_out);
    if (error_out) {
        throw InvalidInputException(IntoErrString(error_out));
    }
    return OperatorPartitionData(index);
}

InsertionOrderPreservingMap<string> c_to_string(TableFunctionToStringInput &input) {
    InsertionOrderPreservingMap<string> result;
    auto &bind = input.bind_data->Cast<CTableBindData>();

    // Call the Rust side to get custom string representation if available
    if (bind.info->vtab.to_string) {
        auto map = bind.info->vtab.to_string(bind.ffi_data->DataPtr());
        if (map) {
            // Copy the map contents to the result
            auto *cpp_map = reinterpret_cast<InsertionOrderPreservingMap<string> *>(map);
            for (const auto &[key, value] : *cpp_map) {
                result[key] = value;
            }
            // Free the map allocated by Rust
            duckdb_vx_string_map_free(map);
        }
    }

    return result;
}

extern "C" duckdb_state duckdb_vx_tfunc_register(duckdb_database ffi_db, const duckdb_vx_tfunc_vtab_t *vtab) {
    if (!ffi_db || !vtab) {
        return DuckDBError;
    }

    auto wrapper = reinterpret_cast<duckdb::DatabaseWrapper *>(ffi_db);
    auto db = wrapper->database->instance;
    auto tf = TableFunction(vtab->name, {}, c_function, c_bind, c_init_global, c_init_local);

    tf.pushdown_complex_filter = c_pushdown_complex_filter;

    tf.projection_pushdown = vtab->projection_pushdown;
    tf.filter_pushdown = vtab->filter_pushdown;
    tf.filter_prune = vtab->filter_prune;
    tf.sampling_pushdown = vtab->sampling_pushdown;
    tf.late_materialization = vtab->late_materialization;
    tf.cardinality = c_cardinality;
    tf.get_partition_data = c_get_partition_data;
    tf.get_virtual_columns = c_get_virtual_columns;
    tf.to_string = c_to_string;
    tf.table_scan_progress = c_table_scan_progress;

    // Set up the parameters
    tf.arguments.reserve(vtab->parameter_count);
    for (size_t i = 0; i < vtab->parameter_count; i++) {
        auto logical_type = reinterpret_cast<LogicalType *>(vtab->parameters[i]);
        tf.arguments.emplace_back(*logical_type);
    }
    // And the named parameters
    for (size_t i = 0; i < vtab->named_parameter_count; i++) {
        auto logical_type = reinterpret_cast<LogicalType *>(vtab->named_parameter_types[i]);
        tf.named_parameters.emplace(vtab->named_parameter_names[i], *logical_type);
    }

    // Assign the VTable to the function info so we can access it later to invoke the callbacks.
    tf.function_info = make_shared_ptr<CTableFunctionInfo>(*vtab);

    try {
        auto &system_catalog = Catalog::GetSystemCatalog(*db);
        auto data = CatalogTransaction::GetSystemTransaction(*db);
        CreateTableFunctionInfo tf_info(tf);
        system_catalog.CreateFunction(data, tf_info);
    } catch (...) {
        return DuckDBError;
    }
    return DuckDBSuccess;
}

extern "C" duckdb_vx_string_map duckdb_vx_string_map_create() {
    auto map = new InsertionOrderPreservingMap<string>();
    return reinterpret_cast<duckdb_vx_string_map>(map);
}

extern "C" void duckdb_vx_string_map_insert(duckdb_vx_string_map map, const char *key, const char *value) {
    if (!map || !key || !value) {
        return;
    }
    auto *cpp_map = reinterpret_cast<InsertionOrderPreservingMap<string> *>(map);
    (*cpp_map)[string(key)] = string(value);
}

extern "C" void duckdb_vx_string_map_free(duckdb_vx_string_map map) {
    if (!map) {
        return;
    }
    auto *cpp_map = reinterpret_cast<InsertionOrderPreservingMap<string> *>(map);
    delete cpp_map;
}
} // namespace vortex
