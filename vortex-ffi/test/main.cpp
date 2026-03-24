// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <unistd.h>

// TODO remove
typedef void FFI_ArrowSchema;
typedef void FFI_ArrowArrayStream;

#include "vortex.h"
#include "common.h"

namespace fs = std::filesystem;
using namespace std::string_literals;
using namespace std::string_view_literals;
using Catch::Matchers::ContainsSubstring;

TEST_CASE("Session creation", "[session]") {
    vx_session *session = vx_session_new();
    REQUIRE(session != nullptr);
    vx_session *session2 = vx_session_clone(session);
    REQUIRE(session2 != nullptr);
    REQUIRE(session != session2);
    vx_session_free(session);
    vx_session_free(session2);
}

TEST_CASE("Creating and iterating binaries", "[binary]") {
    for (std::string_view str : {"ololo"sv, "Широкая строка"sv, "مرحبا بالعالم"sv}) {
        const vx_binary *binary = vx_binary_new(str.data(), str.size());

        REQUIRE(binary != nullptr);
        const size_t len = vx_binary_len(binary);
        REQUIRE(len == str.size());

        const char *ptr = vx_binary_ptr(binary);
        REQUIRE(std::string_view {ptr, len} == str);

        const vx_binary *binary2 = vx_binary_clone(binary);
        vx_binary_free(binary);

        ptr = vx_binary_ptr(binary2);
        REQUIRE(std::string_view {ptr, len} == str);

        vx_binary_free(binary2);
    }
}

TEST_CASE("Creating dtypes", "[dtype]") {
    const vx_dtype *dtype = vx_dtype_new_null();
    REQUIRE(dtype != nullptr);
    CHECK(vx_dtype_get_variant(dtype) == DTYPE_NULL);
    CHECK(vx_dtype_is_nullable(dtype));
    vx_dtype_free(dtype);

    dtype = vx_dtype_new_decimal(5, 2, false);
    REQUIRE(dtype != nullptr);
    CHECK(vx_dtype_get_variant(dtype) == DTYPE_DECIMAL);
    CHECK(vx_dtype_decimal_precision(dtype) == 5);
    CHECK(vx_dtype_decimal_scale(dtype) == 2);
    CHECK_FALSE(vx_dtype_is_nullable(dtype));

    CHECK(vx_dtype_struct_dtype(dtype) == nullptr);
    CHECK(vx_dtype_list_element(dtype) == nullptr);

    vx_dtype_free(dtype);
}

constexpr size_t STRUCT_LEN = 10;
TEST_CASE("Creating structs", "[struct]") {
    vx_struct_fields_builder *builder = vx_struct_fields_builder_new();
    REQUIRE(builder != nullptr);

    for (size_t i = 0; i < STRUCT_LEN; ++i) {
        const std::string target_name = "name"s + std::to_string(i);
        const vx_string *name = vx_string_new(target_name.data(), target_name.size());
        const vx_dtype *dtype = i % 2 ? vx_dtype_new_binary(false) : vx_dtype_new_primitive(PTYPE_F32, true);
        vx_struct_fields_builder_add_field(builder, name, dtype);
    }
    const vx_struct_fields *fields = vx_struct_fields_builder_finalize(builder);
    REQUIRE(fields != nullptr);

    const size_t len = vx_struct_fields_nfields(fields);
    CHECK(len == STRUCT_LEN);
    for (size_t i = 0; i < len; ++i) {
        // borrowed
        const vx_string *name = vx_struct_fields_field_name(fields, i);
        // owned TODO(myrrc): that's weird API
        const vx_dtype *dtype = vx_struct_fields_field_dtype(fields, i);

        std::string_view name_view {vx_string_ptr(name), vx_string_len(name)};
        std::string target_name = "name"s + std::to_string(i);

        CHECK(name_view == target_name);

        if (i % 2) {
            CHECK_FALSE(vx_dtype_is_nullable(dtype));
            CHECK(vx_dtype_get_variant(dtype) == DTYPE_BINARY);
        } else {
            CHECK(vx_dtype_is_nullable(dtype));
            CHECK(vx_dtype_get_variant(dtype) == DTYPE_PRIMITIVE);
        }

        vx_dtype_free(dtype);
    }

    vx_struct_fields_free(fields);
}

struct TempPath : fs::path {
    ~TempPath() { fs::remove(*this); }
};

//constexpr size_t SAMPLE_ROWS = 10;
[[nodiscard]] TempPath write_sample(vx_session *session, fs::path && path) {
    REQUIRE(path.is_absolute());

    constexpr auto col1 = "col1"sv;
    constexpr auto col2 = "col2"sv;

    vx_error *error = nullptr;
    vx_struct_fields_builder *builder = vx_struct_fields_builder_new();

    const vx_string *col1_name = vx_string_new(col1.data(), col1.size());
    const vx_dtype *col1_dtype = vx_dtype_new_primitive(PTYPE_U8, false);
    vx_struct_fields_builder_add_field(builder, col1_name, col1_dtype);

    const vx_string *col2_name = vx_string_new(col2.data(), col2.size());
    const vx_dtype *col2_dtype = vx_dtype_new_utf8(true);
    vx_struct_fields_builder_add_field(builder, col2_name, col2_dtype);

    const vx_struct_fields *fields = vx_struct_fields_builder_finalize(builder);
    const vx_dtype *file_dtype = vx_dtype_new_struct(fields, false);

    vx_array_sink *sink = vx_array_sink_open_file(session, path.c_str(), file_dtype, &error);
    REQUIRE(sink != nullptr);
    require_no_error(error);
    vx_dtype_free(file_dtype);

    //for (size_t i = 0; i < SAMPLE_ROWS; ++i) {
    //    vx_array_sink_push(sink, array, &error);
    //    require_no_error(error);
    //    vx_array_free(array);
    //}

    vx_array_sink_close(sink, &error);
    require_no_error(error);

    INFO("Written vortex file "s + path.generic_string());
    return {path};
}

TEST_CASE("Creating datasources", "[datasource]") {
    vx_session *session = vx_session_new();
    vx_error *error = nullptr;

    const vx_data_source *ds = vx_data_source_new(session, nullptr, &error);
    REQUIRE(ds == nullptr);
    REQUIRE(error != nullptr);
    vx_error_free(error);

    vx_data_source_options opts = {};
    ds = vx_data_source_new(session, &opts, &error);
    REQUIRE(ds == nullptr);
    REQUIRE(error != nullptr);
    REQUIRE(to_string_view(error) == "Other error: empty opts.files");
    vx_error_free(error);

    // First file is opened eagerly
    opts.files = "nonexistent";
    ds = vx_data_source_new(session, &opts, &error);
    REQUIRE(ds == nullptr);
    REQUIRE(error != nullptr);
    REQUIRE_THAT(to_string(error), ContainsSubstring("No such file or directory"));
    vx_error_free(error);

    opts.files = "/tmp/*.vortex";
    ds = vx_data_source_new(session, &opts, &error);
    REQUIRE(ds == nullptr);
    REQUIRE(error != nullptr);
    // TODO Object store error: Generic LocalFileSystem error: Unable to walk dir: File
    // system loop found: /dev/fd/6 points to an ancestor /
    // REQUIRE_THAT(to_string(error), ContainsSubstring("No such file or directory"));
    vx_error_free(error);

    TempPath file = write_sample(session, fs::current_path() / "empty.vortex");

    for (const char *files :
         // TODO Object store error: Generic LocalFileSystem error: Unable to walk dir: File
         // system loop found: /dev/fd/6 points to an ancestor /
         //{ file.c_str(), "*.vortex"}
         {file.c_str()}) {
        INFO("reading "s + files);
        opts.files = files;
        ds = vx_data_source_new(session, &opts, &error);
        require_no_error(error);
        REQUIRE(ds != nullptr);
        vx_data_source_free(ds);
    }

    vx_session_free(session);
}

TEST_CASE("Write file and read back types", "[datasource]") {
    vx_session *session = vx_session_new();
    TempPath path = write_sample(session, fs::current_path() / "write-read-types.vortex");
    vx_error *error = nullptr;

    vx_data_source_options opts = {};
    opts.files = path.c_str();

    const vx_data_source *ds = vx_data_source_new(session, &opts, &error);
    require_no_error(error);
    REQUIRE(ds != nullptr);

    vx_data_source_row_count row_count = {};
    vx_data_source_get_row_count(ds, &row_count);

    CHECK(row_count.cardinality == VX_CARD_MAXIMUM);
    CHECK(row_count.rows == 0);

    const vx_dtype *data_source_dtype = vx_data_source_dtype(ds);
    REQUIRE(vx_dtype_get_variant(data_source_dtype) == DTYPE_STRUCT);

    const vx_struct_fields *fields = vx_dtype_struct_dtype(data_source_dtype);
    const size_t len = vx_struct_fields_nfields(fields);
    REQUIRE(len == 2);

    const vx_dtype *col1_dtype = vx_struct_fields_field_dtype(fields, 0);
    const vx_string *col1_name = vx_struct_fields_field_name(fields, 0);

    REQUIRE(vx_dtype_get_variant(col1_dtype) == DTYPE_PRIMITIVE);
    REQUIRE(vx_dtype_primitive_ptype(col1_dtype) == PTYPE_U8);
    REQUIRE_FALSE(vx_dtype_is_nullable(col1_dtype));
    REQUIRE(to_string_view(col1_name) == "col1");
    vx_dtype_free(col1_dtype);

    const vx_dtype *col2_dtype = vx_struct_fields_field_dtype(fields, 1);
    const vx_string *col2_name = vx_struct_fields_field_name(fields, 1);

    REQUIRE(vx_dtype_get_variant(col2_dtype) == DTYPE_UTF8);
    REQUIRE(vx_dtype_is_nullable(col2_dtype));
    REQUIRE(to_string_view(col2_name) == "col2");
    vx_dtype_free(col2_dtype);

    vx_data_source_free(ds);
    vx_session_free(session);
}

TEST_CASE("Write file and read back", "[datasource]") {
    vx_session *session = vx_session_new();
    TempPath path = write_sample(session, fs::current_path() / "write-read.vortex");
    vx_error *error = nullptr;

    vx_data_source_options ds_options = {};
    ds_options.files = path.c_str();

    const vx_data_source *ds = vx_data_source_new(session, &ds_options, &error);
    require_no_error(error);
    REQUIRE(ds != nullptr);

    vx_scan* scan = vx_data_source_scan(ds, nullptr, &error);
    require_no_error(error);
    REQUIRE(scan != nullptr);
    vx_scan_free(scan);

    vx_scan_options scan_options = {};
    scan = vx_data_source_scan(ds, &scan_options, &error);
    require_no_error(error);
    REQUIRE(scan != nullptr);
    vx_scan_free(scan);

    vx_data_source_free(ds);
    vx_session_free(session);
}
