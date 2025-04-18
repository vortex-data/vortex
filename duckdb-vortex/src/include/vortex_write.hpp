#pragma once

#include "duckdb/main/extension_util.hpp"

namespace duckdb {
	void RegisterVortexWriteFunction(duckdb::DatabaseInstance &instance);
}