#pragma once

#include <string>
#include <type_traits>

#include "duckdb.hpp"
#include "vortex.hpp"

namespace vortex {

inline void HandleError(vx_error *error) {
	if (error != nullptr) {
		auto msg_str = vx_error_get_message(error);
		auto msg = std::string(vx_string_ptr(msg_str), vx_string_len(msg_str));
		vx_error_free(error);
		throw duckdb::InvalidInputException(msg);
	}
}

template <typename Func>
auto Try(Func func) {
	vx_error *error = nullptr;
	// Handle both void and non-void return types.
	if constexpr (std::is_void_v<std::invoke_result_t<Func, vx_error **>>) {
		func(&error);
		HandleError(error);
	} else {
		auto result = func(&error);
		HandleError(error);
		return result;
	}
}

} // namespace vortex
