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
#include <optional>
#include <cstring>

// Internal struct definitions for opaque types

struct cudf_context {
    std::unique_ptr<rmm::mr::cuda_memory_resource> cuda_mr;
};

struct cudf_tableview {
    cudf::unique_table_view_t view;

    explicit cudf_tableview(cudf::unique_table_view_t v) : view(std::move(v)) {}
};

struct cudf_columnview {
    cudf::unique_column_view_t view;

    explicit cudf_columnview(cudf::unique_column_view_t v) : view(std::move(v)) {}
};

// Helper to create an error string
static cudf_err_t make_error(const char* msg) {
    return strdup(msg);
}

static cudf_err_t make_error(const std::string& msg) {
    return strdup(msg.c_str());
}

extern "C" {

cudf_err_t cudf_context_create(cudf_context_t** ctx) {
    if (!ctx) {
        return make_error("ctx pointer is null");
    }

    try {
        auto context = std::make_unique<cudf_context>();
        context->cuda_mr = std::make_unique<rmm::mr::cuda_memory_resource>();
        rmm::mr::set_current_device_resource(context->cuda_mr.get());
        *ctx = context.release();
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

void cudf_context_free(cudf_context_t* ctx) {
    delete ctx;
}

cudf_err_t cudf_tableview_from_device(
    cudf_context_t* ctx,
    const ArrowSchema* schema,
    const ArrowDeviceArray* device_array,
    cudf_tableview_t** out
) {
    if (!ctx) {
        return make_error("context is null");
    }
    if (!schema || !device_array) {
        return make_error("schema or device_array is null");
    }
    if (!out) {
        return make_error("out pointer is null");
    }

    try {
        auto view = cudf::from_arrow_device(schema, device_array);
        *out = new cudf_tableview(std::move(view));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

cudf_err_t cudf_columnview_from_device(
    cudf_context_t* ctx,
    const ArrowSchema* schema,
    const ArrowDeviceArray* device_array,
    cudf_columnview_t** out
) {
    if (!ctx) {
        return make_error("context is null");
    }
    if (!schema || !device_array) {
        return make_error("schema or device_array is null");
    }
    if (!out) {
        return make_error("out pointer is null");
    }

    try {
        auto view = cudf::from_arrow_device_column(schema, device_array);
        *out = new cudf_columnview(std::move(view));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

cudf_err_t cudf_tableview_num_rows(const cudf_tableview_t* tv, int64_t* count) {
    if (!tv) {
        return make_error("table view is null");
    }
    if (!count) {
        return make_error("count pointer is null");
    }

    try {
        *count = static_cast<int64_t>(tv->view->num_rows());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

cudf_err_t cudf_tableview_num_columns(const cudf_tableview_t* tv, int32_t* count) {
    if (!tv) {
        return make_error("table view is null");
    }
    if (!count) {
        return make_error("count pointer is null");
    }

    try {
        *count = static_cast<int32_t>(tv->view->num_columns());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

cudf_err_t cudf_columnview_size(const cudf_columnview_t* cv, int64_t* count) {
    if (!cv) {
        return make_error("column view is null");
    }
    if (!count) {
        return make_error("count pointer is null");
    }

    try {
        *count = static_cast<int64_t>(cv->view->size());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

cudf_err_t cudf_tableview_count_valid(const cudf_tableview_t* tv, int32_t column_index, int64_t* valid_count) {
    if (!tv) {
        return make_error("table view is null");
    }
    if (!valid_count) {
        return make_error("valid_count pointer is null");
    }

    try {
        if (column_index < 0 || column_index >= tv->view->num_columns()) {
            return make_error("column index out of bounds");
        }

        auto col_view = tv->view->column(column_index);
        auto agg = cudf::make_count_aggregation<cudf::reduce_aggregation>();
        auto result = cudf::reduce(col_view, *agg, cudf::data_type{cudf::type_id::INT64});

        auto* int_scalar = static_cast<cudf::numeric_scalar<int64_t>*>(result.get());
        *valid_count = int_scalar->value();

        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

cudf_err_t cudf_columnview_count_valid(const cudf_columnview_t* cv, int64_t* valid_count) {
    if (!cv) {
        return make_error("column view is null");
    }
    if (!valid_count) {
        return make_error("valid_count pointer is null");
    }

    try {
        auto agg = cudf::make_count_aggregation<cudf::reduce_aggregation>();
        auto result = cudf::reduce(*cv->view, *agg, cudf::data_type{cudf::type_id::INT64});

        auto* int_scalar = static_cast<cudf::numeric_scalar<int64_t>*>(result.get());
        *valid_count = int_scalar->value();

        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

cudf_err_t cudf_tableview_sum_int64(const cudf_tableview_t* tv, int32_t column_index, int64_t* sum) {
    if (!tv) {
        return make_error("table view is null");
    }
    if (!sum) {
        return make_error("sum pointer is null");
    }

    try {
        if (column_index < 0 || column_index >= tv->view->num_columns()) {
            return make_error("column index out of bounds");
        }

        auto col_view = tv->view->column(column_index);
        auto agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        auto result = cudf::reduce(col_view, *agg, cudf::data_type{cudf::type_id::INT64});

        auto* int_scalar = static_cast<cudf::numeric_scalar<int64_t>*>(result.get());
        *sum = int_scalar->value();

        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

cudf_err_t cudf_columnview_sum_int64(const cudf_columnview_t* cv, int64_t* sum) {
    if (!cv) {
        return make_error("column view is null");
    }
    if (!sum) {
        return make_error("sum pointer is null");
    }

    try {
        auto agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        auto result = cudf::reduce(*cv->view, *agg, cudf::data_type{cudf::type_id::INT64});

        auto* int_scalar = static_cast<cudf::numeric_scalar<int64_t>*>(result.get());
        *sum = int_scalar->value();

        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

void cudf_tableview_free(cudf_tableview_t* tv) {
    delete tv;
}

void cudf_columnview_free(cudf_columnview_t* cv) {
    delete cv;
}

void cudf_err_free(cudf_err_t err) {
    if (err) {
        free(const_cast<char*>(err));
    }
}

} // extern "C"
