// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <mutex>
#include <nanoarrow/common/inline_types.h>
#include <nanoarrow/hpp/unique.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <random>
#include <thread>
#include <unistd.h>

using FFI_ArrowArrayStream = ArrowArrayStream;
using FFI_ArrowSchema = ArrowSchema;
#define USE_OWN_ARROW 1
#include <vortex.h>

#include "common.h"

namespace fs = std::filesystem;
using namespace std::string_literals;
using namespace std::string_view_literals;
using Catch::Matchers::ContainsSubstring;
using nanoarrow::UniqueArray;
using nanoarrow::UniqueArrayStream;
using nanoarrow::UniqueArrayView;
using nanoarrow::UniqueSchema;

struct TempPath : fs::path {
    TempPath() = default;
    explicit TempPath(fs::path p) : fs::path(std::move(p)) {
    }

    TempPath(const TempPath &) = delete;
    TempPath &operator=(const TempPath &) = delete;

    TempPath(TempPath &&other) noexcept : fs::path(std::move(other)) {
    }
    TempPath &operator=(TempPath &&other) noexcept {
        if (this != &other) {
            fs::remove(*this);
            fs::path::operator=(std::move(other));
        }
        return *this;
    }

    ~TempPath() {
        if (!empty()) {
            fs::remove(*this);
        }
    }
};

// StructArray { age=u8, height=u16? }
[[nodiscard]] const vx_dtype *sample_dtype() {
    vx_struct_fields_builder *builder = vx_struct_fields_builder_new();

    constexpr auto age = "age"sv;
    const vx_string *age_name = vx_string_new(age.data(), age.size());
    const vx_dtype *age_type = vx_dtype_new_primitive(PTYPE_U8, false);
    vx_struct_fields_builder_add_field(builder, age_name, age_type);

    constexpr auto height = "height"sv;
    const vx_string *height_name = vx_string_new(height.data(), height.size());
    const vx_dtype *height_type = vx_dtype_new_primitive(PTYPE_U16, true);
    vx_struct_fields_builder_add_field(builder, height_name, height_type);

    vx_struct_fields *fields = vx_struct_fields_builder_finalize(builder);
    return vx_dtype_new_struct(fields, false);
}

constexpr size_t SAMPLE_ROWS = 100;
std::vector<uint8_t> sample_age() {
    std::vector<uint8_t> buf;
    for (uint8_t age = 0; age < SAMPLE_ROWS; ++age) {
        buf.push_back(age);
    }
    return buf;
}

std::vector<uint16_t> sample_height() {
    std::vector<uint16_t> buf;
    for (uint16_t height = 0; height < SAMPLE_ROWS; ++height) {
        buf.push_back((height + 1) % 200);
    }
    return buf;
}

[[nodiscard]] const vx_array *sample_array() {
    vx_validity validity = {};
    validity.type = VX_VALIDITY_NON_NULLABLE;

    vx_struct_column_builder *builder = vx_struct_column_builder_new(&validity, SAMPLE_ROWS);
    vx_error *error = nullptr;

    std::vector<uint8_t> age_buffer = sample_age();
    const vx_array *age_array =
        vx_array_new_primitive(PTYPE_U8, age_buffer.data(), age_buffer.size(), &validity, &error);
    defer {
        vx_array_free(age_array);
    };
    require_no_error(error);

    vx_struct_column_builder_add_field(builder, "age", age_array, &error);
    require_no_error(error);

    std::vector<uint16_t> height_buffer = sample_height();
    validity.type = VX_VALIDITY_ALL_VALID;
    const vx_array *height_array =
        vx_array_new_primitive(PTYPE_U16, height_buffer.data(), height_buffer.size(), &validity, &error);
    defer {
        vx_array_free(height_array);
    };
    require_no_error(error);

    vx_struct_column_builder_add_field(builder, "height", height_array, &error);
    require_no_error(error);

    const vx_array *array = vx_struct_column_builder_finalize(builder, &error);
    require_no_error(error);
    return array;
}

UniqueSchema sample_schema() {
    UniqueSchema schema;
    REQUIRE(ArrowSchemaInitFromType(schema.get(), NANOARROW_TYPE_STRUCT) == NANOARROW_OK);
    REQUIRE(ArrowSchemaAllocateChildren(schema.get(), 2) == NANOARROW_OK);
    REQUIRE(ArrowSchemaInitFromType(schema->children[0], NANOARROW_TYPE_UINT8) == NANOARROW_OK);
    REQUIRE(ArrowSchemaSetName(schema->children[0], "age") == NANOARROW_OK);
    REQUIRE(ArrowSchemaInitFromType(schema->children[1], NANOARROW_TYPE_UINT16) == NANOARROW_OK);
    REQUIRE(ArrowSchemaSetName(schema->children[1], "height") == NANOARROW_OK);
    return schema;
}

UniqueArrayStream sample_array_stream() {
    UniqueSchema schema = sample_schema();
    UniqueArray arr;

    REQUIRE(ArrowArrayInitFromSchema(arr.get(), schema.get(), nullptr) == NANOARROW_OK);
    REQUIRE(ArrowArrayStartAppending(arr.get()) == NANOARROW_OK);

    auto ages = sample_age();
    auto heights = sample_height();
    for (size_t i = 0; i < ages.size(); ++i) {
        REQUIRE(ArrowArrayAppendInt(arr->children[0], ages[i]) == NANOARROW_OK);
        REQUIRE(ArrowArrayAppendInt(arr->children[1], heights[i]) == NANOARROW_OK);
        REQUIRE(ArrowArrayFinishElement(arr.get()) == NANOARROW_OK);
    }

    REQUIRE(ArrowArrayFinishBuildingDefault(arr.get(), nullptr) == NANOARROW_OK);

    UniqueArrayStream stream;
    REQUIRE(ArrowBasicArrayStreamInit(stream.get(), schema.get(), 1) == NANOARROW_OK);

    ArrowBasicArrayStreamSetArray(stream.get(), 0, arr.get());
    return stream;
}

[[nodiscard]] TempPath write_sample(vx_session *session) {
    const fs::path path = std::filesystem::temp_directory_path() /
                          fs::path("test-" + std::to_string(std::random_device {}()) + ".vortex");

    const vx_dtype *dtype = sample_dtype();
    defer {
        vx_dtype_free(dtype);
    };

    vx_error *error = nullptr;
    vx_array_sink *sink = vx_array_sink_open_file(session, path.c_str(), dtype, &error);
    REQUIRE(sink != nullptr);
    require_no_error(error);

    const vx_array *array = sample_array();
    defer {
        vx_array_free(array);
    };
    vx_array_sink_push(sink, array, &error);
    require_no_error(error);

    vx_array_sink_close(sink, &error);
    require_no_error(error);

    INFO("Written vortex file "s + path.generic_string());
    return TempPath {path};
}

TEST_CASE("Creating datasources", "[datasource]") {
    vx_session *session = vx_session_new();
    defer {
        vx_session_free(session);
    };
    vx_error *error = nullptr;

    const vx_data_source *ds = vx_data_source_new(session, nullptr, &error);
    REQUIRE(ds == nullptr);
    REQUIRE(error != nullptr);
    vx_error_free(error);

    vx_data_source_options opts = {};
    ds = vx_data_source_new(session, &opts, &error);
    REQUIRE(ds == nullptr);
    REQUIRE(error != nullptr);
    REQUIRE_THAT(to_string(error), ContainsSubstring("opts.paths"));
    vx_error_free(error);

    // First file is opened eagerly
    opts.paths = "nonexistent";
    ds = vx_data_source_new(session, &opts, &error);
    REQUIRE(ds == nullptr);
    REQUIRE(error != nullptr);
    REQUIRE_THAT(to_string(error), ContainsSubstring("No such file or directory"));
    vx_error_free(error);

    opts.paths = "/tmp2/*.vortex";
    ds = vx_data_source_new(session, &opts, &error);
    REQUIRE(ds == nullptr);
    REQUIRE(error != nullptr);
    vx_error_free(error);

    TempPath file = write_sample(session);
    opts.paths = file.c_str();
    ds = vx_data_source_new(session, &opts, &error);
    require_no_error(error);
    REQUIRE(ds != nullptr);
    vx_data_source_free(ds);
}

TEST_CASE("Write file", "[sink]") {
    vx_session *session = vx_session_new();
    defer {
        vx_session_free(session);
    };
    TempPath path = write_sample(session);
}

TEST_CASE("Write file and read dtypes", "[datasource]") {
    vx_session *session = vx_session_new();
    defer {
        vx_session_free(session);
    };
    TempPath path = write_sample(session);
    vx_error *error = nullptr;

    vx_data_source_options opts = {};
    opts.paths = path.c_str();

    const vx_data_source *ds = vx_data_source_new(session, &opts, &error);
    require_no_error(error);
    REQUIRE(ds != nullptr);
    defer {
        vx_data_source_free(ds);
    };

    vx_estimate row_count;
    vx_data_source_get_row_count(ds, &row_count);

    CHECK(row_count.type == VX_ESTIMATE_EXACT);
    CHECK(row_count.estimate == SAMPLE_ROWS);

    const vx_dtype *data_source_dtype = vx_data_source_dtype(ds);
    REQUIRE(vx_dtype_get_variant(data_source_dtype) == DTYPE_STRUCT);

    const vx_struct_fields *fields = vx_dtype_struct_dtype(data_source_dtype);
    const size_t len = vx_struct_fields_nfields(fields);
    REQUIRE(len == 2);

    const vx_dtype *age_dtype = vx_struct_fields_field_dtype(fields, 0);
    defer {
        vx_dtype_free(age_dtype);
    };
    const vx_string *age_name = vx_struct_fields_field_name(fields, 0);
    REQUIRE(vx_dtype_get_variant(age_dtype) == DTYPE_PRIMITIVE);
    REQUIRE(vx_dtype_primitive_ptype(age_dtype) == PTYPE_U8);
    REQUIRE_FALSE(vx_dtype_is_nullable(age_dtype));
    REQUIRE(to_string_view(age_name) == "age");

    const vx_dtype *height_dtype = vx_struct_fields_field_dtype(fields, 1);
    defer {
        vx_dtype_free(height_dtype);
    };
    const vx_string *height_name = vx_struct_fields_field_name(fields, 1);
    REQUIRE(vx_dtype_get_variant(height_dtype) == DTYPE_PRIMITIVE);
    REQUIRE(vx_dtype_primitive_ptype(height_dtype) == PTYPE_U16);
    REQUIRE(vx_dtype_is_nullable(height_dtype));
    REQUIRE(to_string_view(height_name) == "height");
}

void verify_age_field(const vx_array *age_field) {
    REQUIRE(vx_array_has_dtype(age_field, DTYPE_PRIMITIVE));
    REQUIRE(vx_dtype_primitive_ptype(vx_array_dtype(age_field)) == PTYPE_U8);
    REQUIRE(vx_array_len(age_field) == SAMPLE_ROWS);
    for (size_t i = 0; i < SAMPLE_ROWS; ++i) {
        REQUIRE(vx_array_get_u8(age_field, i) == i);
    }
}

void verify_height_field(const vx_array *height_field) {
    REQUIRE(vx_array_has_dtype(height_field, DTYPE_PRIMITIVE));
    REQUIRE(vx_dtype_primitive_ptype(vx_array_dtype(height_field)) == PTYPE_U16);
    REQUIRE(vx_array_len(height_field) == SAMPLE_ROWS);
    for (size_t i = 0; i < SAMPLE_ROWS; ++i) {
        REQUIRE(vx_array_get_u16(height_field, i) > 0);
    }
}

void verify_sample_array(const vx_array *array) {
    REQUIRE(vx_array_len(array) == SAMPLE_ROWS);
    REQUIRE(vx_array_has_dtype(array, DTYPE_STRUCT));

    const vx_struct_fields *fields = vx_dtype_struct_dtype(vx_array_dtype(array));
    size_t len = vx_struct_fields_nfields(fields);
    REQUIRE(len == 2);

    const vx_dtype *age_dtype = vx_struct_fields_field_dtype(fields, 0);
    REQUIRE(age_dtype != nullptr);
    REQUIRE(vx_dtype_get_variant(age_dtype) == DTYPE_PRIMITIVE);
    REQUIRE(vx_dtype_primitive_ptype(age_dtype) == PTYPE_U8);
    vx_dtype_free(age_dtype);
    const vx_string *age_name = vx_struct_fields_field_name(fields, 0);
    REQUIRE(to_string_view(age_name) == "age");

    const vx_dtype *height_dtype = vx_struct_fields_field_dtype(fields, 1);
    REQUIRE(vx_dtype_get_variant(height_dtype) == DTYPE_PRIMITIVE);
    REQUIRE(vx_dtype_primitive_ptype(height_dtype) == PTYPE_U16);
    vx_dtype_free(height_dtype);
    const vx_string *height_name = vx_struct_fields_field_name(fields, 1);
    REQUIRE(to_string_view(height_name) == "height");

    vx_error *error = nullptr;
    vx_validity validity = {};
    vx_array_get_validity(array, &validity, &error);
    require_no_error(error);
    REQUIRE(validity.type == VX_VALIDITY_NON_NULLABLE);

    const vx_array *age_field = vx_array_get_field(array, 0, &error);
    require_no_error(error);
    verify_age_field(age_field);
    vx_array_free(age_field);

    const vx_array *height_field = vx_array_get_field(array, 1, &error);
    require_no_error(error);
    verify_height_field(height_field);
    vx_array_free(height_field);

    REQUIRE(vx_array_get_field(array, 2, &error) == nullptr);
    REQUIRE(error != nullptr);
    vx_error_free(error);
}

TEST_CASE("Requesting scans", "[datasource]") {
    vx_session *session = vx_session_new();
    defer {
        vx_session_free(session);
    };
    TempPath path = write_sample(session);

    vx_data_source_options ds_options = {};
    ds_options.paths = path.c_str();

    vx_error *error = nullptr;
    const vx_data_source *ds = vx_data_source_new(session, &ds_options, &error);
    require_no_error(error);
    REQUIRE(ds != nullptr);
    defer {
        vx_data_source_free(ds);
    };

    {
        vx_scan *scan = vx_data_source_scan(ds, nullptr, nullptr, &error);
        require_no_error(error);
        REQUIRE(scan != nullptr);
        vx_scan_free(scan);
    }

    vx_scan_options options = {};
    options.max_threads = 1;

    {
        vx_scan *scan = vx_data_source_scan(ds, &options, nullptr, &error);
        require_no_error(error);
        REQUIRE(scan != nullptr);
        vx_scan_free(scan);
    }
}

TEST_CASE("Basic scan", "[datasource]") {
    vx_session *session = vx_session_new();
    defer {
        vx_session_free(session);
    };
    TempPath path = write_sample(session);
    vx_error *error = nullptr;

    vx_data_source_options ds_options = {};
    ds_options.paths = path.c_str();

    const vx_data_source *ds = vx_data_source_new(session, &ds_options, &error);
    require_no_error(error);
    REQUIRE(ds != nullptr);
    defer {
        vx_data_source_free(ds);
    };

    vx_estimate estimate = {};
    vx_scan *scan = vx_data_source_scan(ds, nullptr, &estimate, &error);
    require_no_error(error);
    defer {
        vx_scan_free(scan);
    };
    REQUIRE(scan != nullptr);
    REQUIRE(estimate.type == VX_ESTIMATE_EXACT);
    REQUIRE(estimate.estimate == 1);

    vx_partition *partition = vx_scan_next_partition(scan, &error);
    require_no_error(error);
    defer {
        vx_partition_free(partition);
    };

    estimate = {};
    vx_partition_row_count(partition, &estimate, &error);
    require_no_error(error);
    REQUIRE(estimate.type == VX_ESTIMATE_EXACT);
    REQUIRE(estimate.estimate == SAMPLE_ROWS);

    REQUIRE(vx_scan_next_partition(scan, &error) == nullptr);
    require_no_error(error);

    const vx_array *array = vx_partition_next(partition, &error);
    require_no_error(error);
    REQUIRE(array != nullptr);
    defer {
        vx_array_free(array);
    };

    REQUIRE(vx_partition_next(partition, &error) == nullptr);
    require_no_error(error);

    verify_sample_array(array);
}

TEST_CASE("Multithreaded scan", "[datasource]") {
    vx_session *session = vx_session_new();
    defer {
        vx_session_free(session);
    };

    constexpr size_t NUM_FILES = 10;
    std::vector<TempPath> paths(NUM_FILES);
    std::string paths_str;
    for (size_t i = 0; i < NUM_FILES; ++i) {
        paths[i] = write_sample(session);
        if (i == 0) {
            paths_str = paths[i].c_str();
        } else {
            paths_str += ","s + paths[i].c_str();
        }
    }

    vx_data_source_options ds_options = {};
    ds_options.paths = paths_str.c_str();

    vx_error *error = nullptr;
    const vx_data_source *ds = vx_data_source_new(session, &ds_options, &error);
    require_no_error(error);
    REQUIRE(ds != nullptr);
    defer {
        vx_data_source_free(ds);
    };

    vx_estimate estimate = {};
    vx_scan *scan = vx_data_source_scan(ds, nullptr, &estimate, &error);
    require_no_error(error);
    defer {
        vx_scan_free(scan);
    };
    REQUIRE(scan != nullptr);
    REQUIRE(estimate.type == VX_ESTIMATE_INEXACT);
    REQUIRE(estimate.estimate == NUM_FILES);

    // Catch doesn't support multithreaded assertions, so we throw here
    std::mutex mutex;
    std::vector<std::thread> threads(NUM_FILES);
    threads.reserve(NUM_FILES);
    std::vector<const vx_array *> arrays(NUM_FILES);
    for (size_t i = 0; i < NUM_FILES; ++i) {
        threads[i] = std::thread([&, i] {
            vx_error *error_tl = nullptr;
            vx_partition *partition = nullptr;
            {
                std::lock_guard _(mutex);
                partition = vx_scan_next_partition(scan, &error_tl);
                require_no_error(error_tl, false);
                if (!partition) {
                    throw std::runtime_error("partition = nullptr");
                }
            }

            defer {
                vx_partition_free(partition);
            };

            estimate = {};
            vx_partition_row_count(partition, &estimate, &error_tl);
            require_no_error(error_tl, false);
            if (estimate.type != VX_ESTIMATE_EXACT) {
                throw std::runtime_error("estimate type mismatch");
            }
            if (estimate.estimate != SAMPLE_ROWS) {
                throw std::runtime_error("estimate mismatch");
            }

            const vx_array *array = vx_partition_next(partition, &error_tl);
            require_no_error(error_tl, false);
            if (!array) {
                throw std::runtime_error("array = nullptr");
            }

            arrays[i] = array;
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }

    vx_partition *const partition = vx_scan_next_partition(scan, &error);
    require_no_error(error);
    REQUIRE(partition == nullptr);

    for (const vx_array *array : arrays) {
        REQUIRE(array != nullptr);
        defer {
            vx_array_free(array);
        };
        verify_sample_array(array);
    }
}

const vx_array *scan_with_options(vx_scan_options &options) {
    vx_session *session = vx_session_new();
    defer {
        vx_session_free(session);
    };
    TempPath path = write_sample(session);
    vx_error *error = nullptr;

    vx_data_source_options ds_options = {};
    ds_options.paths = path.c_str();

    const vx_data_source *ds = vx_data_source_new(session, &ds_options, &error);
    require_no_error(error);
    REQUIRE(ds != nullptr);
    defer {
        vx_data_source_free(ds);
    };

    vx_scan *scan = vx_data_source_scan(ds, &options, nullptr, &error);
    require_no_error(error);
    REQUIRE(scan != nullptr);
    defer {
        vx_scan_free(scan);
    };

    vx_partition *partition = vx_scan_next_partition(scan, &error);
    require_no_error(error);
    REQUIRE(partition != nullptr);
    defer {
        vx_partition_free(partition);
    };

    const vx_array *array = vx_partition_next(partition, &error);
    require_no_error(error);
    REQUIRE(array != nullptr);

    return array;
}

TEST_CASE("Project all fields", "[projection]") {
    vx_scan_options opts = {};
    const vx_array *array = scan_with_options(opts);
    defer {
        vx_array_free(array);
    };
    verify_sample_array(array);
}

TEST_CASE("Project root", "[projection]") {
    vx_expression *root = vx_expression_root();
    defer {
        vx_expression_free(root);
    };
    vx_scan_options opts = {};
    opts.projection = root;
    const vx_array *array = scan_with_options(opts);
    defer {
        vx_array_free(array);
    };
    verify_sample_array(array);
}

TEST_CASE("Project single field", "[projection]") {
    vx_expression *root = vx_expression_root();
    defer {
        vx_expression_free(root);
    };
    vx_scan_options opts = {};

    vx_expression *age_field = vx_expression_get_item("age", root);
    REQUIRE(age_field != nullptr);
    defer {
        vx_expression_free(age_field);
    };

    {
        opts.projection = age_field;
        const vx_array *array = scan_with_options(opts);
        defer {
            vx_array_free(array);
        };
        verify_age_field(array);
    }

    vx_expression *height_field = vx_expression_get_item("height", root);
    REQUIRE(height_field != nullptr);
    defer {
        vx_expression_free(height_field);
    };

    {
        opts.projection = height_field;
        const vx_array *array = scan_with_options(opts);
        defer {
            vx_array_free(array);
        };
        verify_height_field(array);
    }
}

TEST_CASE("Filter with literal expression", "[filter]") {
    vx_expression *root = vx_expression_root();
    defer {
        vx_expression_free(root);
    };

    vx_expression *age_field = vx_expression_get_item("age", root);
    REQUIRE(age_field != nullptr);
    defer {
        vx_expression_free(age_field);
    };

    uint8_t threshold = 50;
    vx_scalar *threshold_scalar = vx_scalar_new_u8(threshold, false);
    REQUIRE(threshold_scalar != nullptr);
    defer {
        vx_scalar_free(threshold_scalar);
    };

    vx_error *literal_error = nullptr;
    vx_expression *threshold_expr = vx_expression_literal(threshold_scalar, &literal_error);
    require_no_error(literal_error);
    REQUIRE(threshold_expr != nullptr);
    defer {
        vx_expression_free(threshold_expr);
    };

    vx_expression *filter = vx_expression_binary(VX_OPERATOR_GTE, age_field, threshold_expr);
    REQUIRE(filter != nullptr);
    defer {
        vx_expression_free(filter);
    };

    vx_scan_options opts = {};
    opts.filter = filter;
    const vx_array *array = scan_with_options(opts);
    defer {
        vx_array_free(array);
    };

    REQUIRE(vx_array_len(array) == SAMPLE_ROWS - threshold);

    vx_error *error = nullptr;
    const vx_array *filtered_age = vx_array_get_field(array, 0, &error);
    require_no_error(error);
    REQUIRE(filtered_age != nullptr);
    defer {
        vx_array_free(filtered_age);
    };

    for (size_t i = 0; i < vx_array_len(filtered_age); ++i) {
        REQUIRE(vx_array_get_u8(filtered_age, i) == static_cast<uint8_t>(threshold + i));
    }
}

TEST_CASE("Project UTF-8 literal expression", "[projection]") {
    constexpr auto value = "constant"sv;
    vx_error *scalar_error = nullptr;
    vx_scalar *literal_scalar = vx_scalar_new_utf8(value.data(), value.size(), false, &scalar_error);
    require_no_error(scalar_error);
    REQUIRE(literal_scalar != nullptr);
    defer {
        vx_scalar_free(literal_scalar);
    };

    vx_error *literal_error = nullptr;
    vx_expression *literal_expr = vx_expression_literal(literal_scalar, &literal_error);
    require_no_error(literal_error);
    REQUIRE(literal_expr != nullptr);
    defer {
        vx_expression_free(literal_expr);
    };

    vx_scan_options opts = {};
    opts.projection = literal_expr;
    const vx_array *array = scan_with_options(opts);
    defer {
        vx_array_free(array);
    };

    REQUIRE(vx_array_len(array) == SAMPLE_ROWS);

    for (size_t i : {size_t {0}, SAMPLE_ROWS - 1}) {
        const vx_string *actual = vx_array_get_utf8(array, static_cast<uint32_t>(i));
        REQUIRE(actual != nullptr);
        defer {
            vx_string_free(actual);
        };
        REQUIRE(to_string_view(actual) == value);
    }
}

void compare_schemas(const UniqueSchema &left, const UniqueSchema &right) {
    REQUIRE(std::string_view {left->format} == std::string_view {right->format});
    REQUIRE(left->n_children == right->n_children);
    for (int64_t i = 0; i < left->n_children; i++) {
        std::string_view name_left = left->children[i]->name;
        std::string_view name_right = right->children[i]->name;
        REQUIRE(name_left == name_right);
        compare_schemas(left->children[i], right->children[i]);
    }
}

void compare_schema_with_sample(const UniqueSchema &left) {
    compare_schemas(left, sample_schema());
}

void compare_stream_with_sample(UniqueArrayStream &left) {
    UniqueArrayStream right = sample_array_stream();
    UniqueSchema schema_right = sample_schema();

    ArrowError error;
    UniqueSchema schema_left;
    ArrowErrorCode res = ArrowArrayStreamGetSchema(left.get(), schema_left.get(), &error);
    REQUIRE(res == NANOARROW_OK);

    while (true) {
        UniqueArray chunk_left, chunk_right;
        int next_left = left->get_next(left.get(), chunk_left.get());
        int next_right = right->get_next(right.get(), chunk_right.get());
        REQUIRE(next_left == next_right);
        if (next_left != NANOARROW_OK || next_right != NANOARROW_OK) {
            return;
        }

        bool done_left = chunk_left->release == nullptr;
        bool done_right = chunk_right->release == nullptr;
        REQUIRE(done_left == done_right);
        if (done_left || done_right) {
            return;
        }

        REQUIRE(chunk_left->length == chunk_right->length);

        UniqueArrayView view_left, view_right;
        res = ArrowArrayViewInitFromSchema(view_left.get(), schema_left.get(), nullptr);
        REQUIRE(res == NANOARROW_OK);
        res = ArrowArrayViewInitFromSchema(view_right.get(), schema_right.get(), nullptr);
        REQUIRE(res == NANOARROW_OK);

        res = ArrowArrayViewSetArray(view_left.get(), chunk_left.get(), nullptr);
        REQUIRE(res == NANOARROW_OK);
        res = ArrowArrayViewSetArray(view_right.get(), chunk_right.get(), nullptr);
        REQUIRE(res == NANOARROW_OK);

        for (int64_t i = 0; i < chunk_left->length; i++) {
            auto name_left = ArrowArrayViewGetIntUnsafe(view_left->children[0], i);
            auto name_right = ArrowArrayViewGetIntUnsafe(view_right->children[0], i);
            REQUIRE(name_left == name_right);

            auto age_left = ArrowArrayViewGetIntUnsafe(view_left->children[1], i);
            auto age_right = ArrowArrayViewGetIntUnsafe(view_right->children[1], i);
            REQUIRE(age_left == age_right);
        }
    }
}

TEST_CASE("Scan Arrow schema", "[scan]") {
    vx_session *session = vx_session_new();
    defer {
        vx_session_free(session);
    };

    TempPath path = write_sample(session);
    vx_error *error = nullptr;

    vx_data_source_options ds_options = {};
    ds_options.paths = path.c_str();

    const vx_data_source *ds = vx_data_source_new(session, &ds_options, &error);
    require_no_error(error);
    REQUIRE(ds != nullptr);
    defer {
        vx_data_source_free(ds);
    };

    vx_scan *scan = vx_data_source_scan(ds, nullptr, nullptr, &error);
    require_no_error(error);
    REQUIRE(scan != nullptr);
    defer {
        vx_scan_free(scan);
    };

    ArrowSchema schema;
    const vx_dtype *dtype = vx_scan_dtype(scan, &error);
    require_no_error(error);
    REQUIRE(dtype != nullptr);
    defer {
        vx_dtype_free(dtype);
    };

    int res = vx_dtype_to_arrow_schema(dtype, &schema, &error);
    REQUIRE(res == 0);
    require_no_error(error);

    UniqueSchema unique_schema;
    ArrowSchemaMove(&schema, unique_schema.get());
    compare_schema_with_sample(unique_schema);
}

TEST_CASE("Scan to Arrow", "[scan]") {
    vx_session *session = vx_session_new();
    defer {
        vx_session_free(session);
    };
    TempPath path = write_sample(session);
    vx_error *error = nullptr;

    vx_data_source_options ds_options = {};
    ds_options.paths = path.c_str();

    const vx_data_source *ds = vx_data_source_new(session, &ds_options, &error);
    require_no_error(error);
    REQUIRE(ds != nullptr);
    defer {
        vx_data_source_free(ds);
    };

    vx_scan *scan = vx_data_source_scan(ds, nullptr, nullptr, &error);
    require_no_error(error);
    REQUIRE(scan != nullptr);
    defer {
        vx_scan_free(scan);
    };

    vx_partition *partition = vx_scan_next_partition(scan, &error);
    require_no_error(error);
    REQUIRE(partition != nullptr);

    UniqueArrayStream unique_stream;
    {
        ArrowArrayStream stream = {};
        int res = vx_partition_scan_arrow(session, partition, &stream, &error);
        REQUIRE(res == 0);
        require_no_error(error);
        ArrowArrayStreamMove(&stream, unique_stream.get());
    }
    compare_stream_with_sample(unique_stream);
}
