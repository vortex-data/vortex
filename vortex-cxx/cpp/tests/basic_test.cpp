// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <gtest/gtest.h>
#include <filesystem>

#include "vortex.hpp"
#include "vortex_cxx_bridge/lib.h"

#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow.h>

class VortexTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        std::string test_data_path = GetTestDataPath("test_data.vortex");
        vortex::ffi::generate_test_vortex_file(test_data_path.c_str());
    }

protected:
    // Helper function to construct file paths in system temp directory
    static std::string GetTestDataPath(const std::string &filename) {
        std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
        std::filesystem::path vortex_test_dir = temp_dir / "vortex_test";

        if (!std::filesystem::exists(vortex_test_dir)) {
            std::filesystem::create_directories(vortex_test_dir);
        }

        std::filesystem::path target_path = vortex_test_dir / filename;
        return target_path.string();
    }
    // Helper function to validate struct array data
    // NOTE: This depends on the test data generated from `generate_test_vortex_file` in `src/lib.rs`
    void ValidateStructArray(const nanoarrow::UniqueArray &struct_array,
                             const nanoarrow::UniqueSchema &schema) {
        // Validate struct array properties
        ASSERT_EQ(struct_array->length, 5);
        ASSERT_EQ(struct_array->null_count, 0);
        ASSERT_EQ(schema->n_children, 2);

        // Create array views for easier access
        nanoarrow::UniqueArrayView array_view;
        ArrowErrorCode init_result = ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), nullptr);
        ASSERT_EQ(init_result, NANOARROW_OK);
        ArrowErrorCode set_result = ArrowArrayViewSetArray(array_view.get(), struct_array.get(), nullptr);
        ASSERT_EQ(set_result, NANOARROW_OK);

        // Test field "a" (first child)
        ArrowArrayView *field_a_view = array_view->children[0];
        ASSERT_EQ(field_a_view->array->length, 5);
        ASSERT_EQ(field_a_view->array->null_count, 0);

        int32_t values_a[5];
        for (int64_t i = 0; i < 5; ++i) {
            values_a[i] = static_cast<int32_t>(ArrowArrayViewGetIntUnsafe(field_a_view, i));
        }
        ASSERT_EQ(values_a[0], 10);
        ASSERT_EQ(values_a[1], 20);
        ASSERT_EQ(values_a[2], 30);
        ASSERT_EQ(values_a[3], 40);
        ASSERT_EQ(values_a[4], 50);

        // Test field "b" (second child)
        ArrowArrayView *field_b_view = array_view->children[1];
        ASSERT_EQ(field_b_view->array->length, 5);
        ASSERT_EQ(field_b_view->array->null_count, 0);

        int32_t values_b[5];
        for (int64_t i = 0; i < 5; ++i) {
            values_b[i] = static_cast<int32_t>(ArrowArrayViewGetIntUnsafe(field_b_view, i));
        }
        ASSERT_EQ(values_b[0], 10);
        ASSERT_EQ(values_b[1], 20);
        ASSERT_EQ(values_b[2], 30);
        ASSERT_EQ(values_b[3], 40);
        ASSERT_EQ(values_b[4], 50);
    }
};

TEST_F(VortexTest, ScanToArray) {
    auto file = vortex::VortexFile::Open(GetTestDataPath("test_data.vortex"));

    // Test scanning to Arrow C ABI
    auto [arrow, schema] = file.CreateScanBuilder().IntoArray();

    // Create nanoarrow UniqueArray and UniqueSchema from C ABI structs
    nanoarrow::UniqueArray struct_array;
    ArrowArrayMove(&arrow, struct_array.get());

    nanoarrow::UniqueSchema schema_obj;
    ArrowSchemaMove(&schema, schema_obj.get());

    ValidateStructArray(struct_array, schema_obj);
}

TEST_F(VortexTest, ScanToStream) {
    auto file = vortex::VortexFile::Open(GetTestDataPath("test_data.vortex"));

    // Test scanning to ArrowArrayStream
    auto stream = file.CreateScanBuilder().IntoStream();

    // Create nanoarrow ArrayStream wrapper
    nanoarrow::UniqueArrayStream array_stream;
    ArrowArrayStreamMove(&stream, array_stream.get());

    // Test that we can get the schema
    nanoarrow::UniqueSchema schema;
    int get_schema_result = array_stream->get_schema(array_stream.get(), schema.get());
    ASSERT_EQ(get_schema_result, 0);
    ASSERT_EQ(schema->n_children, 2);

    // Test that we can read arrays from the stream
    nanoarrow::UniqueArray array;
    int get_next_result = array_stream->get_next(array_stream.get(), array.get());
    ASSERT_EQ(get_next_result, 0);

    ValidateStructArray(array, schema);
}

TEST_F(VortexTest, ScanOptionsWithLimitWithRowRange) {
    auto file = vortex::VortexFile::Open(GetTestDataPath("test_data.vortex"));

    auto stream = file.CreateScanBuilder().WithLimit(2).WithRowRange(1, 4).IntoStream();

    // Create nanoarrow ArrayStream wrapper
    nanoarrow::UniqueArrayStream array_stream;
    ArrowArrayStreamMove(&stream, array_stream.get());

    // Get the schema
    nanoarrow::UniqueSchema schema;
    int get_schema_result = array_stream->get_schema(array_stream.get(), schema.get());
    ASSERT_EQ(get_schema_result, 0);

    // Read the array from the stream
    nanoarrow::UniqueArray array;
    int get_next_result = array_stream->get_next(array_stream.get(), array.get());
    ASSERT_EQ(get_next_result, 0);

    // Should have limited rows (3 instead of 5)
    ASSERT_EQ(array->length, 2);
    ASSERT_EQ(array->null_count, 0);
    ASSERT_EQ(schema->n_children, 2);

    // Create array view for validation
    nanoarrow::UniqueArrayView array_view;
    ArrowErrorCode init_result = ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), nullptr);
    ASSERT_EQ(init_result, NANOARROW_OK);
    ArrowErrorCode set_result = ArrowArrayViewSetArray(array_view.get(), array.get(), nullptr);
    ASSERT_EQ(set_result, NANOARROW_OK);

    // Test field "a" - first 2 values
    ArrowArrayView *field_a_view = array_view->children[0];
    ASSERT_EQ(field_a_view->array->length, 2);
    ASSERT_EQ(field_a_view->array->null_count, 0);

    int32_t values_a[2];
    for (int64_t i = 0; i < 2; ++i) {
        values_a[i] = static_cast<int32_t>(ArrowArrayViewGetIntUnsafe(field_a_view, i));
    }
    ASSERT_EQ(values_a[0], 20);
    ASSERT_EQ(values_a[1], 30);
}

TEST_F(VortexTest, WriteArrayStream) {
    auto file = vortex::VortexFile::Open(GetTestDataPath("test_data.vortex"));

    // Create an ArrowArrayStream by scanning the file
    auto stream = file.CreateScanBuilder().IntoStream();

    // Write the stream to a new Vortex file
    vortex::VortexWriteOptions write_options;
    ASSERT_NO_THROW(write_options.WriteArrayStream(stream, GetTestDataPath("test_output.vortex")));

    // Verify the written file by opening it
    auto written_file = vortex::VortexFile::Open(GetTestDataPath("test_output.vortex"));
    ASSERT_EQ(written_file.RowCount(), 5);

    // Verify data integrity by scanning the written file
    auto [arrow, schema] = written_file.CreateScanBuilder().IntoArray();

    // Create nanoarrow UniqueArray and UniqueSchema from C ABI structs
    nanoarrow::UniqueArray struct_array;
    ArrowArrayMove(&arrow, struct_array.get());

    nanoarrow::UniqueSchema schema_obj;
    ArrowSchemaMove(&schema, schema_obj.get());

    ValidateStructArray(struct_array, schema_obj);
}
