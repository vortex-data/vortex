#include "duckdb/common/types/vector_buffer.hpp"
#include "duckdb/common/types/vector.hpp"

#include "duckdb_vx.h"

//! Create a DuckDB vortex error.
extern "C" duckdb_vx_error duckdb_vx_error_create(const char *message, size_t message_length) {
	return reinterpret_cast<duckdb_vx_error>(new std::string(message, message_length));
}

std::string IntoErrString(duckdb_vx_error error) {
	if (!error) {
		return nullptr;
	}
	return *reinterpret_cast<std::string *>(error);
}
