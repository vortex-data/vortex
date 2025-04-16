#pragma once

#include "vortex.hpp"

inline void HandleError(VXError *error) {
	if (error != nullptr && error->code != 0) {
		auto msg = std::string(error->message);
		vx_error_free(error);
		throw duckdb::InvalidInputException(msg);
	}
}