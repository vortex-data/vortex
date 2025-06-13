#include "duckdb/common/types/data_chunk.hpp"

#include "duckdb_vx.h"
#include "duckdb_vx/data_chunk.h"

const char *duckdb_data_chunk_to_string(duckdb_data_chunk chunk) {
	auto dchunk = reinterpret_cast<duckdb::DataChunk *>(chunk);
	auto str = dchunk->ToString();
	auto result = static_cast<char *>(duckdb_malloc(str.size() + 1));
	memcpy(result, str.c_str(), str.size() + 1);
	return result;
}

void duckdb_data_chunk_verify2(duckdb_data_chunk chunk) {
	auto dchunk = reinterpret_cast<duckdb::DataChunk *>(chunk);
	dchunk->Verify();
}