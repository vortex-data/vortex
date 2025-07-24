// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <gtest/gtest.h>
#include <filesystem>

#include "vortex/file.hpp"
#include "vortex/scan.hpp"
#include "vortex/write_options.hpp"
#include "vortex/thread_pool.hpp"
#include "vortex_cxx_bridge/lib.h"
#include "vortex_cxx_bridge/gen_test_data.h"

#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow.h>

class VortexTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        vortex::ConfigureThreadPool(1);
        std::string test_data_path = GetTestDataPath("test_data.vortex");
        vortex::ffi::testing::generate_test_vortex_file(test_data_path.c_str());
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

    // Helper function to create nanoarrow objects from Arrow C ABI structs
    std::pair<nanoarrow::UniqueArray, nanoarrow::UniqueSchema> CreateNanoarrowFromCAPI(ArrowArray &arrow,
                                                                                       ArrowSchema &schema) {
        nanoarrow::UniqueArray array_obj;
        ArrowArrayMove(&arrow, array_obj.get());

        nanoarrow::UniqueSchema schema_obj;
        ArrowSchemaMove(&schema, schema_obj.get());

        return {std::move(array_obj), std::move(schema_obj)};
    }

    // Helper function to create and initialize array view
    nanoarrow::UniqueArrayView CreateArrayView(const nanoarrow::UniqueArray &array,
                                               const nanoarrow::UniqueSchema &schema) {
        nanoarrow::UniqueArrayView array_view;
        ArrowErrorCode init_result = ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), nullptr);
        EXPECT_EQ(init_result, NANOARROW_OK);
        ArrowErrorCode set_result = ArrowArrayViewSetArray(array_view.get(), array.get(), nullptr);
        EXPECT_EQ(set_result, NANOARROW_OK);
        return array_view;
    }

    // Helper function to create array stream and get schema
    std::pair<nanoarrow::UniqueArrayStream, nanoarrow::UniqueSchema>
    CreateArrayStreamWithSchema(ArrowArrayStream &stream) {
        nanoarrow::UniqueArrayStream array_stream;
        ArrowArrayStreamMove(&stream, array_stream.get());

        nanoarrow::UniqueSchema schema;
        int get_schema_result = array_stream->get_schema(array_stream.get(), schema.get());
        EXPECT_EQ(get_schema_result, 0);

        return {std::move(array_stream), std::move(schema)};
    }

    // Helper function to validate basic array properties
    void ValidateBasicArrayProperties(const nanoarrow::UniqueArray &array, int64_t expected_length,
                                      int64_t expected_null_count) {
        ASSERT_EQ(array->length, expected_length);
        ASSERT_EQ(array->null_count, expected_null_count);
    }

    // Helper function to extract field values
    std::vector<int32_t> ExtractFieldValues(ArrowArrayView *field_view, int64_t count) {
        EXPECT_EQ(field_view->array->length, count);
        EXPECT_EQ(field_view->array->null_count, 0);

        std::vector<int32_t> values(count);
        for (int64_t i = 0; i < count; ++i) {
            values[i] = static_cast<int32_t>(ArrowArrayViewGetIntUnsafe(field_view, i));
        }
        return values;
    }

    // Helper function to validate field values against expected values
    void ValidateFieldValues(ArrowArrayView *field_view, const std::vector<int32_t> &expected_values) {
        auto actual_values = ExtractFieldValues(field_view, static_cast<int64_t>(expected_values.size()));
        for (size_t i = 0; i < expected_values.size(); ++i) {
            ASSERT_EQ(actual_values[i], expected_values[i]);
        }
    }

    // Helper function to test scan builder with custom configuration
    void TestScanBuilderWithValidation(const std::function<void(vortex::ScanBuilder &)> &configureScanBuilder,
                                       const std::vector<int32_t> &expected_values_a,
                                       const std::vector<int32_t> &expected_values_b) {

        auto file = vortex::VortexFile::Open(GetTestDataPath("test_data.vortex"));
        auto scan_builder = file.CreateScanBuilder();
        configureScanBuilder(scan_builder);
        auto stream = scan_builder.IntoStream();

        // Create nanoarrow ArrayStream wrapper and get schema
        auto [array_stream, schema] = CreateArrayStreamWithSchema(stream);

        // Read the array from the stream
        nanoarrow::UniqueArray array;
        int get_next_result = array_stream->get_next(array_stream.get(), array.get());
        ASSERT_EQ(get_next_result, 0);

        // Validate array properties
        int64_t expected_length = static_cast<int64_t>(expected_values_a.size());
        ValidateBasicArrayProperties(array, expected_length, 0);

        // Create array view for validation
        auto array_view = CreateArrayView(array, schema);

        // Validate field values
        ValidateFieldValues(array_view->children[0], expected_values_a);
        ValidateFieldValues(array_view->children[1], expected_values_b);
    }

    // Helper function to validate struct array data
    // NOTE: This depends on the test data generated from `generate_test_vortex_file` in `src/lib.rs`
    void ValidateStructArray(const nanoarrow::UniqueArray &struct_array,
                             const nanoarrow::UniqueSchema &schema) {
        ValidateBasicArrayProperties(struct_array, 5, 0);
        ASSERT_EQ(schema->n_children, 2);

        auto array_view = CreateArrayView(struct_array, schema);

        // Test field "a" (first child)
        std::vector<int32_t> expected_values_a = {10, 20, 30, 40, 50};
        ValidateFieldValues(array_view->children[0], expected_values_a);

        // Test field "b" (second child)
        std::vector<int32_t> expected_values_b = {10, 20, 30, 40, 50};
        ValidateFieldValues(array_view->children[1], expected_values_b);
    }
};

TEST_F(VortexTest, ScanToArray) {
    auto file = vortex::VortexFile::Open(GetTestDataPath("test_data.vortex"));

    // Test scanning to Arrow C ABI
    auto [arrow, schema] = file.CreateScanBuilder().IntoArray();

    // Create nanoarrow objects from C ABI structs
    auto [struct_array, schema_obj] = CreateNanoarrowFromCAPI(arrow, schema);

    ValidateStructArray(struct_array, schema_obj);
}

TEST_F(VortexTest, ScanToStream) {
    auto file = vortex::VortexFile::Open(GetTestDataPath("test_data.vortex"));

    // Test scanning to ArrowArrayStream
    auto stream = file.CreateScanBuilder().IntoStream();

    // Create nanoarrow ArrayStream wrapper and get schema
    auto [array_stream, schema] = CreateArrayStreamWithSchema(stream);

    // Test that we can read arrays from the stream
    nanoarrow::UniqueArray array;
    int get_next_result = array_stream->get_next(array_stream.get(), array.get());
    ASSERT_EQ(get_next_result, 0);

    ValidateStructArray(array, schema);
}

TEST_F(VortexTest, ScanBuilderWithLimitWithRowRange) {
    // Test field "a" and "b" - should contain values 20, 30 (rows 1-2 from original data)
    std::vector<int32_t> expected_values_a = {20, 30};
    std::vector<int32_t> expected_values_b = {20, 30};

    TestScanBuilderWithValidation(
        [](vortex::ScanBuilder &scan_builder) { scan_builder.WithLimit(2).WithRowRange(1, 4); },
        expected_values_a, expected_values_b);
}

TEST_F(VortexTest, ScanBuilderWithIncludeByIndex) {
    std::vector<uint64_t> include_by_index = {1, 3};
    std::vector<int32_t> expected_values_a = {20, 40};
    std::vector<int32_t> expected_values_b = {20, 40};

    TestScanBuilderWithValidation(
        [&include_by_index](vortex::ScanBuilder &scan_builder) {
            scan_builder.WithIncludeByIndex(include_by_index.data(), include_by_index.size());
        },
        expected_values_a, expected_values_b);
}

TEST_F(VortexTest, ScanBuilderWithRowRangeWithIncludeByIndex) {
    std::vector<uint64_t> include_by_index = {1, 3, 4};
    std::vector<int32_t> expected_values_a = {40, 50};
    std::vector<int32_t> expected_values_b = {40, 50};

    TestScanBuilderWithValidation(
        [&include_by_index](vortex::ScanBuilder &scan_builder) {
            scan_builder.WithRowRange(2, 6);
            scan_builder.WithIncludeByIndex(include_by_index.data(), include_by_index.size());
        },
        expected_values_a, expected_values_b);
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

    // Create nanoarrow objects from C ABI structs
    auto [struct_array, schema_obj] = CreateNanoarrowFromCAPI(arrow, schema);

    ValidateStructArray(struct_array, schema_obj);
}
