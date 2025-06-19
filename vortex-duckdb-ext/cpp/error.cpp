#include "duckdb/common/types/vector_buffer.hpp"
#include "duckdb/common/types/vector.hpp"

#include "duckdb_vx.h"

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

namespace vortex {

std::string IntoErrString(duckdb_vx_error error) {
    if (!error) {
        return nullptr;
    }
    return *reinterpret_cast<std::string *>(error);
}

} // namespace vortex
