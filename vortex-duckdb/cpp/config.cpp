// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "include/duckdb_vx/config.h"
#include "duckdb.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/config.hpp"
#include <string>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <thread>

using namespace duckdb;

extern "C" {

duckdb_config duckdb_vx_database_get_config(duckdb_database database) {
    if (!database) {
        return nullptr;
    }

    auto wrapper = reinterpret_cast<DatabaseWrapper *>(database);
    auto &config = DBConfig::GetConfig(*wrapper->database->instance);
    return reinterpret_cast<duckdb_config>(&config);
}

duckdb_state duckdb_vx_get_config_value(duckdb_config config, const char *key, duckdb_value *out_value) {
    if (!config || !key || !out_value) {
        return DuckDBError;
    }

    try {
        // Cast the config to DuckDB's internal config type
        auto *db_config = reinterpret_cast<DBConfig *>(config);

        if (!db_config) {
            return DuckDBError;
        }

        std::string key_str(key);

        // First check set_variable_defaults (the primary location for config values)
        auto set_it = db_config->options.set_variable_defaults.find(key_str);
        if (set_it != db_config->options.set_variable_defaults.end()) {
            *out_value = reinterpret_cast<duckdb_value>(new Value(set_it->second));
            return DuckDBSuccess;
        }

        // Then check user_options
        auto user_it = db_config->options.user_options.find(key_str);
        if (user_it != db_config->options.user_options.end()) {
            *out_value = reinterpret_cast<duckdb_value>(new Value(user_it->second));
            return DuckDBSuccess;
        }

        // Key not found in any map
        return DuckDBError;

    } catch (...) {
        return DuckDBError;
    }
}

int duckdb_vx_config_has_key(duckdb_config config, const char *key) {
    if (!config || !key) {
        return 0;
    }

    try {
        auto *db_config = reinterpret_cast<DBConfig *>(config);
        if (!db_config) {
            return 0;
        }

        std::string key_str(key);

        // Check if the key exists in set_variable_defaults (primary location)
        if (db_config->options.set_variable_defaults.find(key_str) !=
            db_config->options.set_variable_defaults.end()) {
            return 1;
        }

        // Check if the key exists in user_options
        if (db_config->options.user_options.find(key_str) != db_config->options.user_options.end()) {
            return 1;
        }

        return 0;

    } catch (...) {
        return 0;
    }
}

char *duckdb_vx_value_to_string(duckdb_value value) {
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

duckdb_state duckdb_vx_add_extension_option(duckdb_config config,
                                            const char *name,
                                            const char *description,
                                            duckdb_logical_type logical_type,
                                            duckdb_value default_value) {
    if (!name || !description || !logical_type || !default_value) {
        return DuckDBError;
    }

    try {
        auto *db_config = reinterpret_cast<DBConfig *>(config);
        if (!db_config) {
            return DuckDBError;
        }

        auto *type = reinterpret_cast<LogicalType *>(logical_type);
        auto *value = reinterpret_cast<Value *>(default_value);

        db_config->AddExtensionOption(string(name), string(description), *type, *value);

        return DuckDBSuccess;
    } catch (...) {
        return DuckDBError;
    }
}

} // extern "C"
