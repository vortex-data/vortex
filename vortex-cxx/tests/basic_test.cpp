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

TEST_F(VortexTest, ArrayLength) {
  auto array = vortex::Array::create_dummy();
  EXPECT_EQ(array.len(), 3);  // Our dummy array has 3 elements
}

TEST_F(VortexTest, ArrayDType) {
  auto array = vortex::Array::create_dummy();
  auto dtype = array.dtype();

  vortex::DType dt(std::move(dtype));
  auto dtype_variant = dt.dtype_variant();
  EXPECT_EQ(dtype_variant, vortex::ffi::DType::Primitive);
  auto ptype_variant = dt.ptype_variant();
  EXPECT_EQ(ptype_variant, vortex::ffi::PType::I32);
}

TEST_F(VortexTest, ArrayScalarAccess) {
  auto array = vortex::Array::create_dummy();

  // Test accessing first element
  auto scalar = array.scalar_at(0);

  vortex::Scalar s(std::move(scalar));
  EXPECT_FALSE(s.is_null());
  EXPECT_EQ(s.as_i32(), 1);
}

TEST_F(VortexTest, ArraySlice) {
  auto array = vortex::Array::create_dummy();

  // Test slicing
  auto sliced = array.slice(0, 2);
  EXPECT_EQ(sliced.len(), 2);

  // Test accessing sliced element
  auto scalar = sliced.scalar_at(0);
  vortex::Scalar s(std::move(scalar));
  EXPECT_EQ(s.as_i32(), 1);
}

TEST_F(VortexTest, ScalarConversions) {
  auto array = vortex::Array::create_dummy();
  auto scalar = array.scalar_at(1);
  vortex::Scalar s(std::move(scalar));

  // Test different type conversions
  EXPECT_EQ(s.as_i32(), 2);
  EXPECT_EQ(s.as_i64(), 2);
  // Note: Direct conversion from i32 to f32/f64 may not be supported in all
  // cases This depends on Vortex's casting capabilities
}

TEST_F(VortexTest, ErrorHandling) {
  auto array = vortex::Array::create_dummy();

  // Test out of bounds access
  EXPECT_THROW(array.scalar_at(10), vortex::VortexException);

  // Test invalid slice
  EXPECT_THROW(array.slice(5, 10), vortex::VortexException);
}

TEST_F(VortexTest, ArrowConversion) {
  auto array = vortex::Array::create_dummy();

  // Test converting to native Arrow C ABI structures
  auto [native_arrow, native_schema] = array.to_arrow_c_abi();

  // Test basic properties of the Arrow array
  EXPECT_EQ(native_arrow.length, 3);      // Our dummy array has 3 elements
  EXPECT_EQ(native_arrow.null_count, 0);  // Null count should be 0
  EXPECT_EQ(native_arrow.offset, 0);      // No offset for this simple array

  EXPECT_NE(native_schema.format, nullptr);
  EXPECT_STREQ(native_schema.format, "i");
}

TEST_F(VortexTest, FileOperations) {
  // Test opening a non-existent file
  EXPECT_THROW(vortex::File::open("non_existent_file.vortex"),
               vortex::VortexException);

  // Test opening and reading from the generated test file
  try {
    auto file = vortex::File::open("../target/debug/build/test_data.vortex");

    // Test file.row_count()
    auto row_count = file.row_count();
    EXPECT_EQ(row_count, 5);  // Our test file has 5 elements

    // Test file.read_all()
    auto array = file.read_all();
    EXPECT_EQ(array.len(), 5);  // Should have 5 elements

    // Test accessing elements from the file
    auto scalar = array.scalar_at(0);
    vortex::Scalar s(std::move(scalar));
    EXPECT_EQ(s.as_i32(), 10);  // First element should be 10

    scalar = array.scalar_at(4);
    vortex::Scalar s2(std::move(scalar));
    EXPECT_EQ(s2.as_i32(), 50);  // Last element should be 50

    auto [arrow, schema] = array.to_arrow_c_abi();
    EXPECT_EQ(arrow.length, 5);

  } catch (const vortex::VortexException &e) {
    // If the test file doesn't exist, skip the test
    GTEST_SKIP() << "Test file not found: " << e.what();
  }
}