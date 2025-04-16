#pragma once

#include "vortex.hpp"

inline void HandleError(FFIError *error) {
	if (error != nullptr && error->code != 0) {
		auto msg = std::string(error->message);
		FFIError_free(error);
		throw duckdb::InvalidInputException(msg);
	}
}