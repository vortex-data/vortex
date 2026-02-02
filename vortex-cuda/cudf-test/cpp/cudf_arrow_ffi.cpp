// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "cudf_arrow_ffi.h"

#include <cudf/interop.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/aggregation.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>

// Global table storage (in real code, you'd want proper handle management)
static std::unique_ptr<cudf::table> g_loaded_table;

extern "C" {

CudfResult cudf_init() {
    try {
        // Initialize RMM with default CUDA memory resource
        static rmm::mr::cuda_memory_resource cuda_mr;
        rmm::mr::set_current_device_resource(&cuda_mr);
        return CudfResult{CUDF_SUCCESS, nullptr};
    } catch (const std::exception& e) {
        // Note: In production, you'd want to properly manage this string's lifetime
        return CudfResult{CUDF_ERROR_INIT_FAILED, strdup(e.what())};
    }
}

CudfResult cudf_load_from_arrow_device(
    const ArrowSchema* schema,
    const ArrowDeviceArray* device_array
) {
    if (!schema || !device_array) {
        return CudfResult{CUDF_ERROR_INVALID_ARGUMENT, "schema or device_array is null"};
    }

    try {
        // Use cudf's from_arrow_device to import the data
        // This takes ownership of the ArrowDeviceArray
        g_loaded_table = cudf::from_arrow_device(schema, device_array);

        return CudfResult{CUDF_SUCCESS, nullptr};
    } catch (const std::exception& e) {
        return CudfResult{CUDF_ERROR_LOAD_FAILED, strdup(e.what())};
    }
}

CudfResult cudf_load_column_from_arrow_device(
    const ArrowSchema* schema,
    const ArrowDeviceArray* device_array
) {
    if (!schema || !device_array) {
        return CudfResult{CUDF_ERROR_INVALID_ARGUMENT, "schema or device_array is null"};
    }

    try {
        // Use cudf's from_arrow_device_column to import a single column
        auto column = cudf::from_arrow_device_column(schema, device_array);

        // Wrap the column in a table for consistent handling
        std::vector<std::unique_ptr<cudf::column>> columns;
        columns.push_back(std::move(column));
        g_loaded_table = std::make_unique<cudf::table>(std::move(columns));

        return CudfResult{CUDF_SUCCESS, nullptr};
    } catch (const std::exception& e) {
        return CudfResult{CUDF_ERROR_LOAD_FAILED, strdup(e.what())};
    }
}

CudfResult cudf_get_row_count(int64_t* count) {
    if (!count) {
        return CudfResult{CUDF_ERROR_INVALID_ARGUMENT, "count pointer is null"};
    }

    if (!g_loaded_table) {
        return CudfResult{CUDF_ERROR_NO_DATA, "no table loaded"};
    }

    try {
        *count = static_cast<int64_t>(g_loaded_table->num_rows());
        return CudfResult{CUDF_SUCCESS, nullptr};
    } catch (const std::exception& e) {
        return CudfResult{CUDF_ERROR_OPERATION_FAILED, strdup(e.what())};
    }
}

CudfResult cudf_get_column_count(int32_t* count) {
    if (!count) {
        return CudfResult{CUDF_ERROR_INVALID_ARGUMENT, "count pointer is null"};
    }

    if (!g_loaded_table) {
        return CudfResult{CUDF_ERROR_NO_DATA, "no table loaded"};
    }

    try {
        *count = static_cast<int32_t>(g_loaded_table->num_columns());
        return CudfResult{CUDF_SUCCESS, nullptr};
    } catch (const std::exception& e) {
        return CudfResult{CUDF_ERROR_OPERATION_FAILED, strdup(e.what())};
    }
}

CudfResult cudf_count_valid(int32_t column_index, int64_t* valid_count) {
    if (!valid_count) {
        return CudfResult{CUDF_ERROR_INVALID_ARGUMENT, "valid_count pointer is null"};
    }

    if (!g_loaded_table) {
        return CudfResult{CUDF_ERROR_NO_DATA, "no table loaded"};
    }

    try {
        auto view = g_loaded_table->view();
        if (column_index < 0 || column_index >= view.num_columns()) {
            return CudfResult{CUDF_ERROR_INVALID_ARGUMENT, "column index out of bounds"};
        }

        auto column_view = view.column(column_index);

        // count_all aggregation counts all non-null values
        auto agg = cudf::make_count_aggregation<cudf::reduce_aggregation>();
        auto result = cudf::reduce(column_view, *agg, cudf::data_type{cudf::type_id::INT64});

        // Get the scalar value
        auto* int_scalar = static_cast<cudf::numeric_scalar<int64_t>*>(result.get());
        *valid_count = int_scalar->value();

        return CudfResult{CUDF_SUCCESS, nullptr};
    } catch (const std::exception& e) {
        return CudfResult{CUDF_ERROR_OPERATION_FAILED, strdup(e.what())};
    }
}

CudfResult cudf_sum_int64(int32_t column_index, int64_t* sum) {
    if (!sum) {
        return CudfResult{CUDF_ERROR_INVALID_ARGUMENT, "sum pointer is null"};
    }

    if (!g_loaded_table) {
        return CudfResult{CUDF_ERROR_NO_DATA, "no table loaded"};
    }

    try {
        auto view = g_loaded_table->view();
        if (column_index < 0 || column_index >= view.num_columns()) {
            return CudfResult{CUDF_ERROR_INVALID_ARGUMENT, "column index out of bounds"};
        }

        auto column_view = view.column(column_index);

        auto agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        auto result = cudf::reduce(column_view, *agg, cudf::data_type{cudf::type_id::INT64});

        auto* int_scalar = static_cast<cudf::numeric_scalar<int64_t>*>(result.get());
        *sum = int_scalar->value();

        return CudfResult{CUDF_SUCCESS, nullptr};
    } catch (const std::exception& e) {
        return CudfResult{CUDF_ERROR_OPERATION_FAILED, strdup(e.what())};
    }
}

CudfResult cudf_free_table() {
    g_loaded_table.reset();
    return CudfResult{CUDF_SUCCESS, nullptr};
}

void cudf_free_error(const char* error_msg) {
    if (error_msg) {
        free(const_cast<char*>(error_msg));
    }
}

} // extern "C"
