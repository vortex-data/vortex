#pragma once

#include "vortex.hpp"


inline void HandleError(const FFIError *error) {
	if (error != nullptr && error->code != 0) {
		throw duckdb::InvalidInputException(error->message);
	}
}