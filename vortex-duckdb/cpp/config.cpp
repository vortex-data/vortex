// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "include/duckdb_vx/config.h"
#include "duckdb.hpp"
#include <string>
#include <memory>

using namespace duckdb;

extern "C" {

duckdb_state duckdb_vx_get_config_value(duckdb_config config, const char* key, duckdb_value* out_value) {
    if (!config || !key || !out_value) {
        return DuckDBError;
    }

    try {
        // Cast the config to DuckDB's internal config type
        auto* db_config = reinterpret_cast<DBConfig*>(config);
        
        if (!db_config) {
            return DuckDBError;
        }

        std::string key_str(key);
        
        // First check set_variables (the primary location for config values)
        auto set_it = db_config->options.set_variables.find(key_str);
        if (set_it != db_config->options.set_variables.end()) {
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
        
    } catch (const std::exception& e) {
        return DuckDBError;
    } catch (...) {
        return DuckDBError;
    }
}

int duckdb_vx_config_has_key(duckdb_config config, const char* key) {
    if (!config || !key) {
        return 0;
    }

    try {
        auto* db_config = reinterpret_cast<DBConfig*>(config);
        if (!db_config) {
            return 0;
        }
        
        std::string key_str(key);
        
        // Check if the key exists in set_variables (primary location)
        if (db_config->options.set_variables.find(key_str) != db_config->options.set_variables.end()) {
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

} // extern "C"