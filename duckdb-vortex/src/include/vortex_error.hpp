#pragma once

#include "vortex.hpp"

inline void HandleError(vx_error *error) {
	if (error != nullptr) {
		auto msg = std::string(vx_error_get_message(error));
		vx_error_free(error);
		throw duckdb::InvalidInputException(msg);
	}
}

template <typename Func>
auto Try(Func func) {
	vx_error *error = nullptr;
	auto result = func(&error);
	HandleError(error);
	return result;
}
