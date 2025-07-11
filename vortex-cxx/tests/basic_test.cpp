// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/type.h>
#include <gtest/gtest.h>

#include "vortex.hpp"

class VortexTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        // vortex::ConfigureRuntime(2);
    }

protected:
    // Helper function to validate struct array data
    // This depends on the data in `build.rs`
    void ValidateStructArray(const std::shared_ptr<arrow::StructArray> &struct_array) {
        ASSERT_EQ(struct_array->length(), 5);
        ASSERT_EQ(struct_array->null_count(), 0);
        ASSERT_EQ(struct_array->num_fields(), 2);

        // Test field "a"
        auto field_a = struct_array->field(0);
        auto int32_array_a = std::static_pointer_cast<arrow::Int32Array>(field_a);
        ASSERT_EQ(int32_array_a->length(), 5);
        ASSERT_EQ(int32_array_a->null_count(), 0);
        ASSERT_EQ(int32_array_a->Value(0), 10);
        ASSERT_EQ(int32_array_a->Value(1), 20);
        ASSERT_EQ(int32_array_a->Value(2), 30);
        ASSERT_EQ(int32_array_a->Value(3), 40);
        ASSERT_EQ(int32_array_a->Value(4), 50);

        // Test field "b"
        auto field_b = struct_array->field(1);
        auto int32_array_b = std::static_pointer_cast<arrow::Int32Array>(field_b);
        ASSERT_EQ(int32_array_b->length(), 5);
        ASSERT_EQ(int32_array_b->null_count(), 0);
        ASSERT_EQ(int32_array_b->Value(0), 10);
        ASSERT_EQ(int32_array_b->Value(1), 20);
        ASSERT_EQ(int32_array_b->Value(2), 30);
        ASSERT_EQ(int32_array_b->Value(3), 40);
        ASSERT_EQ(int32_array_b->Value(4), 50);
    }
};

TEST_F(VortexTest, ScanToArray) {
    auto file = vortex::VortexFile::Open("../target/debug/build/test_data.vortex");

    // Test scanning to Arrow C ABI
    auto [arrow, schema] = file.CreateScanBuilder().IntoArray();

    // Import the Arrow array using Arrow C++ API
    auto maybe_data_type = arrow::ImportType(&schema);
    ASSERT_TRUE(maybe_data_type.ok())
        << "Failed to import Arrow schema: " << maybe_data_type.status().message();
    auto data_type = maybe_data_type.ValueOrDie();

    auto maybe_imported_array = arrow::ImportArray(&arrow, data_type);
    ASSERT_TRUE(maybe_imported_array.ok())
        << "Failed to import Arrow array: " << maybe_imported_array.status().message();
    auto imported_array = maybe_imported_array.ValueOrDie();

    // Cast to StructArray to access struct fields
    auto struct_array = std::static_pointer_cast<arrow::StructArray>(imported_array);
    ValidateStructArray(struct_array);
}

TEST_F(VortexTest, ScanToStream) {
    auto file = vortex::VortexFile::Open("../target/debug/build/test_data.vortex");

    // Test scanning to Arrow RecordBatchReader
    auto maybe_reader = file.CreateScanBuilder().IntoStream();
    ASSERT_TRUE(maybe_reader.ok()) << "Failed to create RecordBatchReader: "
                                   << maybe_reader.status().message();

    auto reader = maybe_reader.ValueOrDie();
    ASSERT_NE(reader, nullptr);

    // Test that we can get the schema
    auto schema = reader->schema();
    ASSERT_NE(schema, nullptr);
    EXPECT_GT(schema->num_fields(), 0);

    // Test that we can read record batches
    auto maybe_batch = reader->Next();
    ASSERT_TRUE(maybe_batch.ok()) << "Failed to read first batch: " << maybe_batch.status().message();

    auto batch = maybe_batch.ValueOrDie();
    if (batch != nullptr) {
        auto struct_array = batch->ToStructArray().ValueOrDie();
        ValidateStructArray(struct_array);
    }
}

TEST_F(VortexTest, ScanOptionsWithLimit) {
    auto file = vortex::VortexFile::Open("../target/debug/build/test_data.vortex");

    auto maybe_reader = file.CreateScanBuilder().SetLimit(3).IntoStream();
    ASSERT_TRUE(maybe_reader.ok()) << "Failed to create RecordBatchReader: "
                                   << maybe_reader.status().message();

    auto reader = maybe_reader.ValueOrDie();
    ASSERT_NE(reader, nullptr);

    auto maybe_batch = reader->Next();
    ASSERT_TRUE(maybe_batch.ok()) << "Failed to read first batch: " << maybe_batch.status().message();

    auto batch = maybe_batch.ValueOrDie();
    if (batch != nullptr) {
        // Should have limited rows (3 instead of 5)
        ASSERT_EQ(batch->num_rows(), 3);

        auto struct_array = batch->ToStructArray().ValueOrDie();
        ASSERT_EQ(struct_array->length(), 3);
        ASSERT_EQ(struct_array->null_count(), 0);
        ASSERT_EQ(struct_array->num_fields(), 2);

        // Test field "a" - first 3 values
        auto field_a = struct_array->field(0);
        auto int32_array_a = std::static_pointer_cast<arrow::Int32Array>(field_a);
        ASSERT_EQ(int32_array_a->length(), 3);
        ASSERT_EQ(int32_array_a->null_count(), 0);
        ASSERT_EQ(int32_array_a->Value(0), 10);
        ASSERT_EQ(int32_array_a->Value(1), 20);
        ASSERT_EQ(int32_array_a->Value(2), 30);
    }
}

TEST_F(VortexTest, WriteArrayStream) {
    auto file = vortex::VortexFile::Open("../target/debug/build/test_data.vortex");

    // Create an Arrow RecordBatchReader by scanning the file
    auto maybe_reader = file.CreateScanBuilder().IntoStream();
    ASSERT_TRUE(maybe_reader.ok()) << "Failed to create RecordBatchReader: "
                                   << maybe_reader.status().message();

    auto reader = maybe_reader.ValueOrDie();
    ASSERT_NE(reader, nullptr);

    // Convert to Arrow C ABI stream
    ArrowArrayStream stream;
    ASSERT_EQ(arrow::ExportRecordBatchReader(reader, &stream), arrow::Status::OK());

    // Write the stream to a new Vortex file
    vortex::VortexWriteOptions write_options;
    ASSERT_NO_THROW(write_options.WriteArrayStream(stream, "../target/debug/build/test_output.vortex"));

    // Verify the written file by opening it
    auto written_file = vortex::VortexFile::Open("../target/debug/build/test_output.vortex");
    ASSERT_EQ(written_file.RowCount(), 5);

    // Verify data integrity by scanning the written file
    auto [arrow, schema] = written_file.CreateScanBuilder().IntoArray();

    auto maybe_data_type = arrow::ImportType(&schema);
    ASSERT_TRUE(maybe_data_type.ok())
        << "Failed to import Arrow schema: " << maybe_data_type.status().message();
    auto data_type = maybe_data_type.ValueOrDie();

    auto maybe_imported_array = arrow::ImportArray(&arrow, data_type);
    ASSERT_TRUE(maybe_imported_array.ok())
        << "Failed to import Arrow array: " << maybe_imported_array.status().message();
    auto imported_array = maybe_imported_array.ValueOrDie();

    auto struct_array = std::static_pointer_cast<arrow::StructArray>(imported_array);
    ValidateStructArray(struct_array);
}
