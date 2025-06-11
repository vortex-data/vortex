#pragma once

#include "duckdb_vx/data.h"

namespace vortex {

class CData final {
public:
	CData(void *data_ptr, duckdb_delete_callback_t callback);

	// Disable copy constructor to prevent accidental copies.
	CData(const CData &) = delete;

	// Disable assignment operator to prevent accidental assignments.
	CData &operator=(const CData &) = delete;

	~CData();

	void *DataPtr() const;

private:
	void *data = nullptr;
	duckdb_delete_callback_t delete_callback = nullptr;
};

} // namespace vortex
