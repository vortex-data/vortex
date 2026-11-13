// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <rmm/device_scalar.hpp>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>

namespace {
using CudfEnvironment = cudf::test::BaseFixture;

__global__ void write_value(int32_t *value) {
    *value = 42;
}

TEST_F(CudfEnvironment, CompiledCudaKernel) {
    auto const stream = cudf::test::get_default_stream();
    rmm::device_scalar<int32_t> result {0, stream};
    write_value<<<1, 1, 0, stream.value()>>>(result.data());
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(result.value(stream), 42);
}

TEST_F(CudfEnvironment, MixedTypeBinaryOperation) {
    auto const stream = cudf::test::get_default_stream();
    cudf::test::fixed_width_column_wrapper<uint8_t> codes {0, 2, 1};
    cudf::numeric_scalar<int32_t> base {3, true, stream};
    auto result = cudf::binary_operation(codes,
                                         base,
                                         cudf::binary_operator::ADD,
                                         cudf::data_type {cudf::type_id::INT32},
                                         stream);
    cudf::test::fixed_width_column_wrapper<int32_t> expected {3, 5, 4};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}
} // namespace
