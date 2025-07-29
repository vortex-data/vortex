// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <gtest/gtest.h>
#include <filesystem>
#include <thread>
#include <iostream>

#include "vortex/file.hpp"
#include "vortex/scan.hpp"
#include "vortex/write_options.hpp"
#include "vortex_cxx_bridge/lib.h"
#include "vortex_cxx_bridge/gen_test_data.h"

#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow.h>

class VortexTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
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
        ArrowError error;
        ArrowErrorCode init_result = ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), &error);
        if (init_result != NANOARROW_OK) {
            std::cerr << "Error: " << error.message << std::endl;
            std::abort();
        }
        ArrowErrorCode set_result = ArrowArrayViewSetArray(array_view.get(), array.get(), nullptr);
        if (set_result != NANOARROW_OK) {
            std::cerr << "Error: " << error.message << std::endl;
            std::abort();
        }
        return array_view;
    }

    std::pair<nanoarrow::UniqueArrayStream, nanoarrow::UniqueSchema>
    CreateArrayStreamWithSchema(ArrowArrayStream &stream) {
        nanoarrow::UniqueArrayStream array_stream;
        ArrowArrayStreamMove(&stream, array_stream.get());

        return {std::move(array_stream), GetSchemaFromArrayStream(array_stream.get())};
    }

    nanoarrow::UniqueSchema GetSchemaFromArrayStream(ArrowArrayStream *array_stream) {
        nanoarrow::UniqueSchema schema;
        int get_schema_result = array_stream->get_schema(array_stream, schema.get());
        if (get_schema_result != NANOARROW_OK) {
            std::cerr << "Error: " << array_stream->get_last_error(array_stream) << std::endl;
            std::abort();
        }
        return schema;
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
    auto out_stream = written_file.CreateScanBuilder().IntoStream();
    auto [array_stream, schema] = CreateArrayStreamWithSchema(out_stream);
    nanoarrow::UniqueArray array;
    int get_next_result = array_stream->get_next(array_stream.get(), array.get());
    ASSERT_EQ(get_next_result, 0);

    ValidateStructArray(array, schema);
}

TEST_F(VortexTest, ConcurrentMultiStreamRead) {
    std::string test_data_path_1m = GetTestDataPath("test_data_1m.vortex");
    vortex::ffi::testing::generate_test_vortex_file_1m(test_data_path_1m.c_str());

    auto file = vortex::VortexFile::Open(test_data_path_1m);
    auto stream_driver = file.CreateScanBuilder().IntoStreamDriver();

    // Structure to hold batch data with first ID and nanoarrow array
    struct BatchData {
        int64_t first_id;
        nanoarrow::UniqueArray array;

        BatchData(int64_t first_id, nanoarrow::UniqueArray array)
            : first_id(first_id), array(std::move(array)) {
        }
    };

    std::vector<BatchData> thread1_batches;
    std::vector<BatchData> thread2_batches;
    auto stream_for_schema = stream_driver.CreateArrayStream();
    auto schema = GetSchemaFromArrayStream(&stream_for_schema);

    // Helper function to read from a stream and collect batches
    auto read_stream = [&](std::vector<BatchData> &batches) {
        // Each thread creates its own stream
        auto stream = stream_driver.CreateArrayStream();
        auto [array_stream, _] = CreateArrayStreamWithSchema(stream);

        std::vector<BatchData> local_batches;

        while (true) {
            nanoarrow::UniqueArray array;
            int get_next_result = array_stream->get_next(array_stream.get(), array.get());

            if (get_next_result != 0) {
                std::cerr << "Error: " << array_stream->get_last_error(array_stream.get()) << std::endl;
                std::abort();
            }

            if (array->length == 0) {
                break; // Empty array indicates end
            }

            auto array_view = CreateArrayView(array, schema);

            int64_t first_id = ArrowArrayViewGetIntUnsafe(array_view->children[0], 0);

            local_batches.emplace_back(first_id, std::move(array));
        }
        batches = std::move(local_batches);
    };

    // Launch two threads
    std::thread thread1(read_stream, std::ref(thread1_batches));
    std::thread thread2(read_stream, std::ref(thread2_batches));

    // Wait for both threads to complete
    thread1.join();
    thread2.join();

    // Combine all batches from both threads
    std::vector<BatchData> all_batches;
    all_batches.insert(all_batches.end(), std::make_move_iterator(thread1_batches.begin()),
                       std::make_move_iterator(thread1_batches.end()));
    all_batches.insert(all_batches.end(), std::make_move_iterator(thread2_batches.begin()),
                       std::make_move_iterator(thread2_batches.end()));

    // Sort batches by first ID to ensure proper validation order
    std::sort(all_batches.begin(), all_batches.end(),
              [](const BatchData &a, const BatchData &b) { return a.first_id < b.first_id; });

    // Validate all data is sequential and correct
    constexpr size_t EXPECTED_ROWS = static_cast<size_t>(1024) * 1024;
    size_t total_rows_read = 0;
    int64_t expected_next_id = 0;

    for (const auto &batch : all_batches) {
        // Create array view for this batch
        auto array_view = CreateArrayView(batch.array, schema);

        for (int64_t i = 0; i < batch.array->length; ++i) {
            int64_t id = ArrowArrayViewGetIntUnsafe(array_view->children[0], i);
            int32_t value = static_cast<int32_t>(ArrowArrayViewGetIntUnsafe(array_view->children[1], i));

            ASSERT_EQ(id, expected_next_id) << "ID mismatch at position " << total_rows_read + i
                                            << ": expected " << expected_next_id << ", got " << id;

            ASSERT_EQ(value, static_cast<int32_t>(static_cast<size_t>(expected_next_id) * 2))
                << "Value mismatch at position " << total_rows_read + i << ": expected "
                << (expected_next_id * 2) << ", got " << value;

            expected_next_id++;
        }
        total_rows_read += batch.array->length;
    }

    // Verify we read all expected data
    ASSERT_EQ(total_rows_read, EXPECTED_ROWS)
        << "Expected to read " << EXPECTED_ROWS << " rows, but read " << total_rows_read << " rows";

    ASSERT_GT(all_batches.size(), 1) << "Expected multiple batches, but got " << all_batches.size();
}
