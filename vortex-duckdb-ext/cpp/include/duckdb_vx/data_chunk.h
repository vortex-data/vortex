#pragma once

#include "duckdb.h"

const char *duckdb_data_chunk_to_string2(duckdb_data_chunk chunk);

void duckdb_data_chunk_verify2(duckdb_data_chunk chunk);