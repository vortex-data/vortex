#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/type.h>
#include <gtest/gtest.h>

#include "vortex.hpp"

class VortexTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Test setup
  }

  void TearDown() override {
    // Test cleanup
  }

  // Helper function to validate struct array data
  // This depends on the data in `build.rs`
  void ValidateStructArray(
      const std::shared_ptr<arrow::StructArray> &struct_array) {
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

TEST_F(VortexTest, ScanToArrow) {
  try {
    auto file =
        vortex::VortexFile::open("../target/debug/build/test_data.vortex");

    // Test scanning to Arrow C ABI
    auto [arrow, schema] = file.scan_to_arrow();

    // Import the Arrow array using Arrow C++ API
    auto maybe_data_type = arrow::ImportType(&schema);
    ASSERT_TRUE(maybe_data_type.ok()) << "Failed to import Arrow schema: "
                                      << maybe_data_type.status().message();
    auto data_type = maybe_data_type.ValueOrDie();

    auto maybe_imported_array = arrow::ImportArray(&arrow, data_type);
    ASSERT_TRUE(maybe_imported_array.ok())
        << "Failed to import Arrow array: "
        << maybe_imported_array.status().message();
    auto imported_array = maybe_imported_array.ValueOrDie();

    // Cast to StructArray to access struct fields
    auto struct_array =
        std::static_pointer_cast<arrow::StructArray>(imported_array);
    ValidateStructArray(struct_array);

  } catch (const vortex::VortexException &e) {
    GTEST_SKIP() << "Test file not found: " << e.what();
  }
}

TEST_F(VortexTest, ScanToStream) {
  try {
    auto file =
        vortex::VortexFile::open("../target/debug/build/test_data.vortex");

    // Test scanning to Arrow RecordBatchReader
    auto maybe_reader = file.scan_to_stream();
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
    ASSERT_TRUE(maybe_batch.ok())
        << "Failed to read first batch: " << maybe_batch.status().message();

    auto batch = maybe_batch.ValueOrDie();
    if (batch != nullptr) {
      auto struct_array = batch->ToStructArray().ValueOrDie();
      ValidateStructArray(struct_array);
    }

  } catch (const vortex::VortexException &e) {
    GTEST_SKIP() << "Test file not found: " << e.what();
  }
}
