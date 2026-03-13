// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb_vx/table_function.h"
#include "vortex.h"

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

std::string to_string(const vx_string *str) {
    return {vx_string_ptr(str), vx_string_len(str)};
}

std::string move_vx_err(vx_error *error) {
    const vx_string *vx_str = vx_error_get_message(error);
    string str {vx_string_ptr(vx_str), vx_string_len(vx_str)};
    vx_error_free(error);
    return str;
}

LogicalTypeId from_ptype(vx_ptype ptype) {
    using enum LogicalTypeId;
    switch (ptype) {
    case PTYPE_U8:
        return UTINYINT;
    case PTYPE_U16:
        return USMALLINT;
    case PTYPE_U32:
        return UINTEGER;
    case PTYPE_U64:
        return UBIGINT;
    case PTYPE_I8:
        return TINYINT;
    case PTYPE_I16:
        return SMALLINT;
    case PTYPE_I32:
        return INTEGER;
    case PTYPE_I64:
        return BIGINT;
    case PTYPE_F16:
        throw BinderException("F16 type not supported in Duckdb");
    case PTYPE_F32:
        return FLOAT;
    case PTYPE_F64:
        return DOUBLE;
    }
    throw BinderException(StringUtil::Format("value %d out of range for vx_ptype", ptype));
}

LogicalType from_dtype(const vx_dtype *dtype);
LogicalType from_struct(const vx_dtype *dtype) {
    const vx_struct_fields *struct_dtype = vx_dtype_struct_dtype(dtype);
    uint64_t struct_size = vx_struct_fields_nfields(struct_dtype);
    child_list_t<LogicalType> children(struct_size);
    for (uint64_t i = 0; i < struct_size; ++i) {
        const vx_string *field_name = vx_struct_fields_field_name(struct_dtype, i);
        const vx_dtype *field_dtype = vx_struct_fields_field_dtype(struct_dtype, i);
        children[i] = {to_string(field_name), from_dtype(field_dtype)};
    }

    return LogicalType::STRUCT(children);
}

LogicalType from_dtype(const vx_dtype *dtype) {
    using enum LogicalTypeId;
    switch (vx_dtype_get_variant(dtype)) {
    case DTYPE_NULL:
        return SQLNULL;
    case DTYPE_BOOL:
        return BOOLEAN;
    case DTYPE_PRIMITIVE:
        return from_ptype(vx_dtype_primitive_ptype(dtype));
    case DTYPE_UTF8:
        return VARCHAR;
    case DTYPE_BINARY:
        return BLOB;
    case DTYPE_STRUCT:
        return from_struct(dtype);
    case DTYPE_DECIMAL: {
        uint8_t width = vx_dtype_decimal_precision(dtype);
        uint8_t scale = vx_dtype_decimal_scale(dtype);
        return LogicalType::DECIMAL(width, scale);
    };
    case DTYPE_LIST: {
        LogicalType child_type = from_dtype(vx_dtype_list_element(dtype));
        return LogicalType::LIST(child_type);
    }
    case DTYPE_FIXED_SIZE_LIST: {
        LogicalType child_type = from_dtype(vx_dtype_fixed_size_list_element(dtype));
        idx_t idx = vx_dtype_fixed_size_list_size(dtype);
        return LogicalType::ARRAY(child_type, idx);
    };
    case DTYPE_EXTENSION: { // TODO Temporal
        throw BinderException("DTYPE_EXTENSION not supported");
    };
    };
    throw BinderException(StringUtil::Format("value %d out of range for vx_dtype", dtype));
}

// TODO This belongs in C++ part

class Session {
public:
    Session() : session(vx_session_new()) {
    }

    Session(const Session &other) : session(vx_session_clone(other.session.get())) {
    }
    Session(Session &&other) noexcept {
        std::swap(session, other.session);
    }

    Session &operator=(const Session &other) {
        session.reset(vx_session_clone(other.session.get()));
        return *this;
    }

    Session &operator=(Session &&other) noexcept {
        std::swap(session, other.session);
        return *this;
    }

    vx_session *handle() const noexcept {
        return session.get();
    }

private:
    struct deleter {
        void operator()(vx_session *session) {
            vx_session_free(session);
        }
    };
    std::unique_ptr<vx_session, deleter> session;
};

class DataSource {
public:
    DataSource(const Session &session, vx_data_source_options &opts) {
        vx_error *err;
        data_source.reset(vx_data_source_new(session.handle(), &opts, &err));
        if (err) {
            throw BinderException(move_vx_err(err));
        }
    }

    DataSource(const DataSource &other) : data_source(vx_data_source_clone(other.data_source.get())) {
    }

    DataSource(DataSource &&other) noexcept {
        std::swap(data_source, other.data_source);
    }

    DataSource &operator=(const DataSource &other) {
        data_source.reset(vx_data_source_clone(other.data_source.get()));
        return *this;
    }

    DataSource &operator=(DataSource &&other) noexcept {
        std::swap(data_source, other.data_source);
        return *this;
    }

    vx_data_source_row_count row_count() const noexcept {
        vx_data_source_row_count rc;
        vx_data_source_get_row_count(handle(), &rc);
        return rc;
    }

    const vx_data_source *handle() const noexcept {
        return data_source.get();
    }

private:
    struct deleter {
        void operator()(const vx_data_source *data_source) {
            vx_data_source_free(data_source);
        }
    };
    std::unique_ptr<const vx_data_source, deleter> data_source;
};

class Array {
public:
    const vx_array *handle() const noexcept {
        return array.get();
    }

private:
    friend class Partition;

    Array(const vx_array *array) : array(array) {
    }

    struct deleter {
        void operator()(const vx_array *array) {
            vx_array_free(array);
        }
    };
    std::unique_ptr<const vx_array, deleter> array;
};

class Partition {
public:
    Partition(const Partition &) = delete;
    Partition &operator=(const Partition &) = delete;
    Partition(Partition &&other) noexcept {
        std::swap(partition, other.partition);
    }

    std::optional<Array> next_array() {
        vx_error *err = nullptr;
        const vx_array *array = vx_partition_next(handle(), &err);
        if (err) {
            throw BinderException(move_vx_err(err));
        }
        if (!array) {
            return std::nullopt;
        }
        return Array {array};
    }

    vx_partition *handle() const noexcept {
        return partition.get();
    }

private:
    friend class Scan;
    Partition(vx_partition *partition) : partition(partition) {
    }

    struct deleter {
        void operator()(vx_partition *partition) {
            vx_partition_free(partition);
        }
    };
    std::unique_ptr<vx_partition, deleter> partition;
};

class Scan {
public:
    Scan(const DataSource &data_source, vx_scan_options &options) {
        vx_error *err = nullptr;
        scan.reset(vx_data_source_scan(data_source.handle(), &options, &err));
        if (err) {
            throw BinderException(move_vx_err(err));
        }
    }

    Scan(const Scan &) = delete;
    Scan &operator=(const Scan &) = delete;

    Scan(Scan &&other) noexcept {
        std::swap(scan, other.scan);
    }

    Scan &operator=(Scan &&other) noexcept {
        std::swap(scan, other.scan);
        return *this;
    }

    double progress() const noexcept {
        return vx_scan_progress(handle());
    }

    std::optional<Partition> next_partition() {
        vx_error *err = nullptr;
        vx_partition *partition = vx_scan_next(handle(), &err);
        if (err) {
            throw BinderException(move_vx_err(err));
        }
        if (!partition) {
            return std::nullopt;
        }
        return Partition {partition};
    }

    vx_scan *handle() const noexcept {
        return scan.get();
    }

private:
    struct deleter {
        void operator()(vx_scan *scan) {
            vx_scan_free(scan);
        }
    };
    std::unique_ptr<vx_scan, deleter> scan;
};

namespace vortex {
struct CTableFunctionInfo final : TableFunctionInfo {
    explicit CTableFunctionInfo(const duckdb_vx_tfunc_vtab_t &vtab) : vtab(vtab) {
    }
    duckdb_vx_tfunc_vtab_t vtab;
};

struct CTableBindData final : TableFunctionData {
    CTableBindData(unique_ptr<CTableFunctionInfo> info_p,
                   Session &&session,
                   DataSource &&data_source,
                   const vector<string> &column_names)
        : info(std::move(info_p)), session(std::move(session)), data_source(std::move(data_source)),
          column_names(column_names) {
    }

    ~CTableBindData() override = default;

    unique_ptr<FunctionData> Copy() const override {
        return make_uniq<CTableBindData>(make_uniq<CTableFunctionInfo>(info->vtab),
                                         Session {session},
                                         DataSource {data_source},
                                         column_names);
    }

    unique_ptr<CTableFunctionInfo> info; // TODO remove

    Session session;
    DataSource data_source;

    vector<string> column_names;
    vector<const vx_expression *> filters;
};

struct CTableGlobalData final : GlobalTableFunctionState {
    explicit CTableGlobalData(Scan &&scan, idx_t max_threads)
        : scan(std::move(scan)), max_threads(max_threads) {
    }

    ~CTableGlobalData() override = default;

    idx_t MaxThreads() const override {
        return max_threads;
    }

    Scan scan;
    idx_t max_threads;
};

struct CTableLocalData final : LocalTableFunctionState {
    explicit CTableLocalData() {
    }
    std::optional<uint64_t> batch_id;
};

unique_ptr<FunctionData> c_bind(ClientContext &context,
                                TableFunctionBindInput &info,
                                vector<LogicalType> &types,
                                vector<string> &names) {
    if (info.inputs.size() != 1) {
        throw BinderException("expected single file glob parameter");
    }
    std::string files_glob = StringValue::Get(info.inputs[0]);

    Session session;

    vx_data_source_options opts = {
        // files_glob lives till end of c_bind, vx_data_source_new copies the argument
        .files = files_glob.data(),
        .fs_use_vortex = nullptr,
        .fs_set_userdata = nullptr,
        .fs_open = nullptr,
        .fs_create = nullptr,
        .fs_list = nullptr,
        .fs_close = nullptr,
        .fs_size = nullptr,
        .fs_read = nullptr,
        .fs_write = nullptr,
        .fs_sync = nullptr,
        .glob = nullptr,
        .cache_init = nullptr,
        .cache_free = nullptr,
        .cache_get = nullptr,
        .cache_put = nullptr,
        .cache_delete = nullptr};

    DataSource data_source {session, opts};

    const vx_dtype *dtype = vx_data_source_dtype(data_source.handle());
    const vx_struct_fields *struct_dtype = vx_dtype_struct_dtype(dtype);

    for (uint64_t i = 0; i < vx_struct_fields_nfields(struct_dtype); ++i) {
        const vx_string *field_name = vx_struct_fields_field_name(struct_dtype, i);
        const vx_dtype *field_dtype = vx_struct_fields_field_dtype(struct_dtype, i);
        if (!field_dtype) {
            throw BinderException(
                StringUtil::Format("Field dtype %s at index %d can't be parsed", to_string(field_name), i));
        }
        names.emplace_back(to_string(field_name));
        types.emplace_back(from_dtype(field_dtype));
    }

    auto &vtab = info.table_function.function_info->Cast<CTableFunctionInfo>().vtab;
    return make_uniq<CTableBindData>(make_uniq<CTableFunctionInfo>(vtab),
                                     std::move(session),
                                     std::move(data_source),
                                     names);
}

const vx_expression *make_projection(const vector<column_t> &column_ids,
                                     const vector<idx_t> &projection_ids,
                                     const vector<string> &column_names) {
    vector<const char *> projected_names;
    projected_names.reserve(column_names.size());

    if (projection_ids.empty()) {
        for (column_t id : column_ids) {
            if (id == COLUMN_IDENTIFIER_EMPTY) {
                continue;
            }
            if (column_names.size() < id) {
                throw InvalidInputException(
                    StringUtil::Format("Expected column id %d but there are %d columns",
                                       id,
                                       column_names.size()));
            }
            // column_names[id] lives till end of projection(). Initialized
            // vx_expression copies the buffer, so it's safe to use data()
            projected_names.emplace_back(column_names[id].data());
        }
    } else {
        for (idx_t projection_id : projection_ids) {
            if (column_ids.size() < projection_id) {
                throw InvalidInputException(
                    StringUtil::Format("Expected projection id %d but there are %d columns",
                                       projection_id,
                                       column_ids.size()));
            }
            column_t id = column_ids[projection_id];
            if (column_names.size() < id) {
                throw InvalidInputException(
                    StringUtil::Format("Expected column id %d but there are %d columns",
                                       id,
                                       column_names.size()));
            }
            projected_names.emplace_back(column_names[id].data());
        }
    }

    vx_expression *root = vx_expression_root();
    // TODO vx_expression may take a string vx_array
    const vx_expression *expr = vx_expression_select(projected_names.data(), projected_names.size(), root);
    vx_expression_free(root);
    return expr;
}

const vx_expression *make_filter(optional_ptr<TableFilterSet> table_filters,
                                 const vector<column_t> &column_ids,
                                 const vector<string> &column_names,
                                 const vector<const vx_expression *> &additional_filters,
                                 const vx_dtype *dtype) {
    return nullptr;
}

unique_ptr<GlobalTableFunctionState> c_init_global(ClientContext &, TableFunctionInitInput &input) {
    const auto &bind = input.bind_data->Cast<CTableBindData>();

    vx_scan_selection selection = {
        .idx = nullptr,
        .idx_len = 0,
        .include = VX_S_INCLUDE_ALL,
    };

    const vx_dtype *dtype = vx_data_source_dtype(bind.data_source.handle());
    const vx_expression *projection =
        make_projection(input.column_ids, input.projection_ids, bind.column_names);
    const vx_expression *filter =
        make_filter(input.filters, input.column_ids, bind.column_names, bind.filters, dtype);

    vx_scan_options options = {
        .projection = projection,
        .filter = filter,
        .row_range_begin = 0,
        .row_range_end = 0,
        .selection = selection,
        .limit = 0,
        .ordered = 0,
    };

    Scan scan {bind.data_source, options};
    return make_uniq<CTableGlobalData>(std::move(scan), bind.info->vtab.max_threads);
}

unique_ptr<LocalTableFunctionState>
c_init_local(ExecutionContext &, TableFunctionInitInput &, GlobalTableFunctionState *) {
    return make_uniq<CTableLocalData>();
}

void c_function(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
    auto &global_state = input.global_state->Cast<CTableGlobalData>();
    auto &batch_id = input.local_state->Cast<CTableLocalData>().batch_id;

    auto next_partition = global_state.scan.next_partition();
    if (!next_partition) {
        return;
    }
    Partition partition = std::move(*next_partition);

    std::optional<Array> array;
    auto export_array = input.bind_data->Cast<CTableBindData>().info->vtab.export_array;
    while ((array = partition.next_array())) {
        uint64_t export_res = export_array(array->handle(), reinterpret_cast<duckdb_data_chunk>(&output));
        if (export_res == std::numeric_limits<uint64_t>::max()) {
            batch_id = std::nullopt;
        } else {
            batch_id = export_res;
        }
    }
}

const vx_expression *from_filter(const Expression &filter) {
    return nullptr;
}

void c_pushdown_complex_filter(ClientContext &,
                               LogicalGet &,
                               FunctionData *bind_data,
                               vector<unique_ptr<Expression>> &filters) {
    auto &bind = bind_data->Cast<CTableBindData>();
    // We don't erase filters, see Nick's comment in datasource.rs
    for (auto iter = filters.begin(); iter != filters.end(); ++iter) {
        bind.filters.emplace_back(from_filter(**iter));
    }
}

unique_ptr<NodeStatistics> c_cardinality(ClientContext &, const FunctionData *bind_data) {
    auto stats = make_uniq<NodeStatistics>();
    vx_data_source_row_count rc = bind_data->Cast<CTableBindData>().data_source.row_count();
    switch (rc.cardinality) {
    case VX_CARD_ESTIMATE: {
        stats->has_estimated_cardinality = true;
        stats->has_max_cardinality = false;
        stats->estimated_cardinality = rc.rows;
        return stats;
    }
    case VX_CARD_MAXIMUM: {
        stats->has_estimated_cardinality = true;
        stats->has_max_cardinality = true;
        stats->estimated_cardinality = rc.rows;
        stats->max_cardinality = rc.rows;
        return stats;
    }
    default: {
        stats->has_estimated_cardinality = false;
        stats->has_max_cardinality = false;
        return stats;
    }
    }
}

OperatorPartitionData c_get_partition_data(ClientContext &, TableFunctionGetPartitionInput &input) {
    if (input.partition_info.RequiresPartitionColumns()) {
        throw InternalException("TableScan::GetPartitionData: partition columns not supported");
    }
    if (auto &batch_id = input.local_state->Cast<CTableLocalData>().batch_id; batch_id) {
        return OperatorPartitionData(*batch_id);
    }
    throw InvalidInputException("Batch id missing, no batches exported");
}

extern "C" duckdb_state duckdb_vx_tfunc_register(duckdb_database ffi_db, const duckdb_vx_tfunc_vtab_t *vtab) {
    if (!ffi_db || !vtab) {
        return DuckDBError;
    }

    auto wrapper = reinterpret_cast<duckdb::DatabaseWrapper *>(ffi_db);
    auto db = wrapper->database->instance;
    auto tf = TableFunction(vtab->name, {}, c_function, c_bind, c_init_global, c_init_local);

    tf.pushdown_complex_filter = c_pushdown_complex_filter;

    // tf.projection_pushdown = vtab->projection_pushdown;
    // tf.filter_pushdown = vtab->filter_pushdown;
    // tf.filter_prune = vtab->filter_prune;
    // tf.sampling_pushdown = vtab->sampling_pushdown;
    // tf.late_materialization = vtab->late_materialization;

    tf.projection_pushdown = true;
    tf.filter_pushdown = false;
    tf.filter_prune = false;
    tf.sampling_pushdown = false;
    tf.late_materialization = false;

    tf.cardinality = c_cardinality;
    tf.get_partition_data = c_get_partition_data;

    tf.to_string = [](TableFunctionToStringInput &) {
        InsertionOrderPreservingMap<string> result;
        result.insert("Function", "Vortex Scan");
        // TODO filters
        return result;
    };

    tf.get_virtual_columns = [](auto &, auto) {
        virtual_column_map_t map = {{COLUMN_IDENTIFIER_EMPTY, TableColumn {"", LogicalType::BOOLEAN}}};
        return map;
    };

    tf.table_scan_progress = [](auto &, auto *, const GlobalTableFunctionState *state) {
        return state->Cast<CTableGlobalData>().scan.progress();
    };

    tf.arguments.reserve(vtab->parameter_count);
    for (size_t i = 0; i < vtab->parameter_count; i++) {
        auto logical_type = reinterpret_cast<LogicalType *>(vtab->parameters[i]);
        tf.arguments.emplace_back(*logical_type);
    }

    for (size_t i = 0; i < vtab->named_parameter_count; i++) {
        auto logical_type = reinterpret_cast<LogicalType *>(vtab->named_parameter_types[i]);
        tf.named_parameters.emplace(vtab->named_parameter_names[i], *logical_type);
    }

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

//
// TODO this stuff should be removed
//

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

    if (!info->named_parameters.contains(name)) {
        return nullptr;
    }
    auto value = duckdb::make_uniq<Value>(info->named_parameters.at(name));
    return reinterpret_cast<duckdb_value>(value.release());
}

extern "C" void duckdb_vx_tfunc_bind_result_add_column(duckdb_vx_tfunc_bind_result ffi_result,
                                                       const char *name_str,
                                                       size_t name_len,
                                                       duckdb_logical_type ffi_type) {
    return;
}

} // namespace vortex
