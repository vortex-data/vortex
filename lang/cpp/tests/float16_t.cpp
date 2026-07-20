// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/dtype.hpp"
#include <catch2/catch_test_macros.hpp>
#include <vortex/scalar.hpp>

using namespace vortex;

namespace {

static_assert(float(1.0f16) == 1.0f);

TEST_CASE("float16_t to_float", "[float]") {
    REQUIRE(float(1.0f16) == 1.0F);
    REQUIRE(float(-2.0f16) == -2.0F);
    REQUIRE(float(0.0f) == 0.0F);
    REQUIRE(float(-0.0f) == -0.0F);
}

#if __STDCPP_FLOAT16_T__ == 1
TEST_CASE("F16 scalar", "[scalar]") {
    std::float16_t float16t = 1.0f16;

    Scalar scalar = scalar::of(float16t);
    REQUIRE(scalar.dtype().variant() == DataTypeVariant::Primitive);
    REQUIRE(scalar.dtype().primitive_type() == vortex::PType::F16);

    _Float16 float16t_alias = 1.0f16;
    scalar = scalar::of(float16t_alias);
    REQUIRE(scalar.dtype().variant() == DataTypeVariant::Primitive);
    REQUIRE(scalar.dtype().primitive_type() == vortex::PType::F16);
}
#endif
} // namespace
