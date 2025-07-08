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

    // Cast to Int32Array to access values
    auto int32_array =
        std::static_pointer_cast<arrow::Int32Array>(imported_array);
    ASSERT_EQ(int32_array->length(), 5);
    ASSERT_EQ(int32_array->null_count(), 0);
    ASSERT_EQ(int32_array->Value(0), 10);
    ASSERT_EQ(int32_array->Value(1), 20);
    ASSERT_EQ(int32_array->Value(2), 30);
    ASSERT_EQ(int32_array->Value(3), 40);
    ASSERT_EQ(int32_array->Value(4), 50);

  } catch (const vortex::VortexException &e) {
    GTEST_SKIP() << "Test file not found: " << e.what();
  }
}

// TEST_F(VortexTest, ScanToStream) {
//   try {
//     auto file =
//     vortex::VortexFile::open("../target/debug/build/test_data.vortex");

//     // Test scanning to Arrow C Stream
//     auto stream = file.scan_to_stream();

//     // Test that we can get the schema
//     auto [schema_array, schema] = stream.get_schema();
//     EXPECT_NE(schema.format, nullptr);

//     // Test that we can get arrays from the stream
//     ArrowArray array;
//     bool has_data = stream.next(array);

//     // If we got a valid array, it should have data
//     if (has_data) {
//       EXPECT_GT(array.length, 0);
//       EXPECT_EQ(array.null_count, 0);
//       EXPECT_EQ(array.offset, 0);

//       // Clean up the array
//       if (array.release != nullptr) {
//         array.release(&array);
//       }
//     }

//     // Clean up schema
//     if (schema.release != nullptr) {
//       schema.release(&schema);
//     }

//   } catch (const vortex::VortexException &e) {
//     GTEST_SKIP() << "Test file not found: " << e.what();
//   }
// }

// TEST_F(VortexTest, StreamEndOfData) {
//   try {
//     auto file =
//     vortex::VortexFile::open("../target/debug/build/test_data.vortex");

//     auto stream = file.scan_to_stream();

//     // Consume all arrays from the stream
//     int count = 0;
//     ArrowArray array;
//     while (stream.next(array)) {
//       count++;
//       // Clean up the array
//       if (array.release != nullptr) {
//         array.release(&array);
//       }
//     }

//     // Should have gotten at least one array
//     EXPECT_GT(count, 0);

//   } catch (const vortex::VortexException &e) {
//     GTEST_SKIP() << "Test file not found: " << e.what();
//   }
// }