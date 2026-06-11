// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "aggregate_fn_pushdown.hpp"
#include "data.hpp"
#include "error.hpp"
#include "scalar_fn_pushdown.hpp"
#include "vortex_duckdb.h"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/geometry_crs.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"

#include <cstring>
#include <string>

using namespace duckdb;

extern "C" char *duckdb_vx_value_to_string(duckdb_value value) {
    if (!value) {
        return nullptr;
    }

    try {
        // Cast the value to DuckDB's internal Value type
        auto *ddb_value = reinterpret_cast<Value *>(value);

        if (!ddb_value) {
            return nullptr;
        }

        // Use the ToString method to get the string representation
        std::string str_value = ddb_value->ToString();

        size_t str_len = str_value.length() + 1;
        char *result = static_cast<char *>(duckdb_malloc(str_len));
        if (!result) {
            return nullptr;
        }

        // Copy the string and null terminate
        std::memcpy(result, str_value.c_str(), str_len);
        return result;

    } catch (...) {
        return nullptr;
    }
}

CData::CData(void *data_ptr, duckdb_delete_callback_t callback) : data(data_ptr), delete_callback(callback) {
}

CData::~CData() {
    if (data && delete_callback) {
        delete_callback(data);
    }
    data = nullptr;
    delete_callback = nullptr;
}

void *CData::DataPtr() const {
    return data;
}

extern "C" duckdb_vx_data duckdb_vx_data_create(void *data, duckdb_delete_callback_t delete_callback) {
    return reinterpret_cast<duckdb_vx_data>(new CData(data, delete_callback));
}

extern "C" const char *duckdb_data_chunk_to_string(duckdb_data_chunk chunk, duckdb_vx_error *err) {
    try {
        auto dchunk = reinterpret_cast<DataChunk *>(chunk);
        auto str = dchunk->ToString();
        auto result = static_cast<char *>(duckdb_malloc(str.size() + 1));
        memcpy(result, str.c_str(), str.size() + 1);
        *err = nullptr;
        return result;
    } catch (std::runtime_error &e) {
        auto s = e.what();
        *err = duckdb_vx_error_create(s, strlen(s));
        return nullptr;
    }
}

extern "C" void duckdb_data_chunk_verify(duckdb_data_chunk chunk, duckdb_vx_error *err) {
    try {
        auto dchunk = reinterpret_cast<DataChunk *>(chunk);
        dchunk->Verify();
        *err = nullptr;
    } catch (std::runtime_error &e) {
        auto s = e.what();
        *err = duckdb_vx_error_create(s, strlen(s));
    }
}

extern "C" duckdb_vx_error duckdb_vx_error_create(const char *message, size_t message_length) {
    return reinterpret_cast<duckdb_vx_error>(new std::string(message, message_length));
}

extern "C" const char *duckdb_vx_error_value(duckdb_vx_error err) {
    auto str = reinterpret_cast<std::string *>(err);
    return str->c_str();
}

extern "C" void duckdb_vx_error_free(duckdb_vx_error err) {
    auto str = reinterpret_cast<std::string *>(err);
    delete str;
}

std::string IntoErrString(duckdb_vx_error error) {
    if (!error) {
        return {};
    }
    std::string *const error_str = reinterpret_cast<std::string *>(error);
    std::string out = std::move(*error_str);
    duckdb_vx_error_free(error);
    return out;
}

duckdb_state SetError(duckdb_vx_error *error_out, std::string_view message) {
    D_ASSERT(error_out != nullptr && "SetError called with null error_out");
    *error_out = duckdb_vx_error_create(message.data(), message.size());
    return DuckDBError;
}

extern "C" duckdb_logical_type duckdb_vx_logical_type_copy(duckdb_logical_type ty) {
    D_ASSERT(ty);
    auto *src = reinterpret_cast<LogicalType *>(ty);
    auto copy = make_uniq<LogicalType>(*src);
    return reinterpret_cast<duckdb_logical_type>(copy.release());
}

extern "C" char *duckdb_vx_logical_type_stringify(duckdb_logical_type c_type) {
    auto type = reinterpret_cast<LogicalType *>(c_type);
    auto str = type->ToString();
    auto result = static_cast<char *>(duckdb_malloc(str.size() + 1));
    memcpy(result, str.c_str(), str.size() + 1);
    return result;
}

extern "C" duckdb_logical_type duckdb_vx_create_geometry(const char *crs) {
    D_ASSERT(crs);
    auto geom = (*crs == '\0') ? LogicalType::GEOMETRY() : LogicalType::GEOMETRY(std::string(crs));
    auto copy = make_uniq<LogicalType>(std::move(geom));
    return reinterpret_cast<duckdb_logical_type>(copy.release());
}

static unique_ptr<TableRef> VortexScanReplacement(ClientContext &context,
                                                  ReplacementScanInput &input,
                                                  optional_ptr<ReplacementScanData>) {
    auto table_name = ReplacementScan::GetFullPath(input);
    if (!ReplacementScan::CanReplace(table_name, {"vortex"})) {
        return nullptr;
    }
    auto table_function = make_uniq<TableFunctionRef>();

    vector<unique_ptr<ParsedExpression>> children(1);
    children[0] = make_uniq<ConstantExpression>(Value(table_name));
    table_function->function = make_uniq<FunctionExpression>("read_vortex", std::move(children));

    if (!FileSystem::HasGlob(table_name)) {
        auto &fs = FileSystem::GetFileSystem(context);
        table_function->alias = fs.ExtractBaseName(table_name);
    }

    return table_function;
}

extern "C" duckdb_state duckdb_vx_register_scan_replacement(duckdb_database duckdb_database) {
    if (!duckdb_database) {
        return DuckDBError;
    }

    auto wrapper = reinterpret_cast<DatabaseWrapper *>(duckdb_database);
    if (!wrapper) {
        return DuckDBError;
    }

    auto &config = DBConfig::GetConfig(*wrapper->database->instance);
    config.replacement_scans.emplace_back(VortexScanReplacement);

    return DuckDBSuccess;
}

// buffer_ptr is shared_ptr, two pointers long, but duckdb_vx_reusable_dict is
// one pointer long, so we need a wrapper.
using Buffer = buffer_ptr<VectorChildBuffer>;
struct ReusableDict {
    Buffer buffer;
    ReusableDict(Buffer buffer) : buffer(std::move(buffer)) {
    }
};

extern "C" duckdb_vx_reusable_dict duckdb_vx_reusable_dict_create(duckdb_logical_type ffi_type, idx_t size) {
    const LogicalType &type = *reinterpret_cast<LogicalType *>(ffi_type);
    auto buffer = DictionaryVector::CreateReusableDictionary(type, size);
    auto ptr = std::make_unique<ReusableDict>(std::move(buffer));
    return reinterpret_cast<duckdb_vx_reusable_dict>(ptr.release());
}

extern "C" void duckdb_vx_reusable_dict_destroy(duckdb_vx_reusable_dict *dict) {
    if (dict && *dict) {
        delete reinterpret_cast<ReusableDict *>(*dict);
    }
}

extern "C" duckdb_vx_reusable_dict duckdb_vx_reusable_dict_clone(duckdb_vx_reusable_dict dict) {
    ReusableDict *wrapper = reinterpret_cast<ReusableDict *>(dict);
    auto ptr = std::make_unique<ReusableDict>(wrapper->buffer);
    return reinterpret_cast<duckdb_vx_reusable_dict>(ptr.release());
}

extern "C" void duckdb_vx_reusable_dict_set_vector(duckdb_vx_reusable_dict reusable,
                                                   duckdb_vector *out_vector) {
    auto *wrapper = reinterpret_cast<ReusableDict *>(reusable);
    *out_vector = reinterpret_cast<duckdb_vector>(&wrapper->buffer->data);
}

extern "C" void duckdb_vx_vector_dictionary_reusable(duckdb_vector ffi_vector,
                                                     duckdb_vx_reusable_dict reusable,
                                                     duckdb_selection_vector ffi_sel_vec) {
    auto vector = reinterpret_cast<Vector *>(ffi_vector);
    auto *wrapper = reinterpret_cast<ReusableDict *>(reusable);
    auto sel_vec = reinterpret_cast<SelectionVector *>(ffi_sel_vec);
    vector->Dictionary(wrapper->buffer, *sel_vec);
}

extern "C" duckdb_value duckdb_vx_value_create_null(duckdb_logical_type ty) {
    const auto logical_type = reinterpret_cast<LogicalType *>(ty);
    auto value = make_uniq<Value>(*logical_type);
    return reinterpret_cast<duckdb_value>(value.release());
}

extern "C" duckdb_value duckdb_vx_value_create_geometry(const uint8_t *wkb, idx_t len, const char *crs) {
    const auto bytes = reinterpret_cast<const_data_ptr_t>(wkb);
    auto value = (crs == nullptr || *crs == '\0')
                     ? Value::GEOMETRY(bytes, len)
                     : Value::GEOMETRY(bytes, len, CoordinateReferenceSystem(std::string(crs)));
    auto owned = make_uniq<Value>(std::move(value));
    return reinterpret_cast<duckdb_value>(owned.release());
}

extern "C" duckdb_blob duckdb_vx_value_get_geometry(duckdb_value value) {
    if (value == nullptr) {
        return {nullptr, 0};
    }
    const auto val = reinterpret_cast<Value *>(value);
    if (val->type().id() != LogicalTypeId::GEOMETRY) {
        return {nullptr, 0};
    }
    const auto &str = StringValue::Get(*val);
    const auto size = str.size();
    auto buf = reinterpret_cast<void *>(duckdb_malloc(size));
    if (size > 0) {
        memcpy(buf, str.c_str(), size);
    }
    return {buf, size};
}

static void VortexOptimizeFunction(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
    plan = TryPushdownScalarFunctions(input.context, std::move(plan));
    plan = TryPushdownAggregateFunctions(input.context, std::move(plan));
}

struct VortexOptimizerExtension final : OptimizerExtension {
    inline VortexOptimizerExtension() : OptimizerExtension(VortexOptimizeFunction, nullptr, {}) {
    }
};

extern "C" duckdb_state duckdb_vx_optimizer_extension_register(duckdb_database ffi_db) {
    D_ASSERT(ffi_db);
    const DatabaseWrapper &wrapper = *reinterpret_cast<DatabaseWrapper *>(ffi_db);
    DatabaseInstance &db = *wrapper.database->instance;
    try {
        DBConfig::GetConfig(db).GetCallbackManager().Register(VortexOptimizerExtension());
    } catch (const std::exception &e) {
        ErrorData data(e);
        DUCKDB_LOG_ERROR(db, "Failed to create Vortex optimizer extension:\t" + data.Message());
        return DuckDBError;
    }
    return DuckDBSuccess;
}
