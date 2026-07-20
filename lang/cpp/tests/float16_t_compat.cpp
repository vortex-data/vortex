// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <cmath>
#include <limits>
// This is UB but no other way to test compatibility macro, fine
// as we're running tests
#undef __STDCPP_FLOAT16_T__
#include <catch2/catch_test_macros.hpp>
#include <vortex/array.hpp>

using namespace vortex;

namespace {

constexpr float16_t one {0x3C00};
static_assert(static_cast<float>(one) == 1.0f);

TEST_CASE("float16_t to_float (compatibility)", "[float]") {
    REQUIRE(float(float16_t {0x3C00}) == 1.0F);
    REQUIRE(float(float16_t {0xC000}) == -2.0F);
    REQUIRE(float(float16_t {0}) == 0.0F);
    REQUIRE(float(float16_t {0x8000}) == -0.0F);
    REQUIRE(float(float16_t {0x7C00}) == std::numeric_limits<float>::infinity());
    REQUIRE(float(float16_t {0xFC00}) == -std::numeric_limits<float>::infinity());
    REQUIRE(std::fpclassify(float(float16_t {0x7E01})) == FP_NAN);
    // Denormalized float16_t always gets to normal floats on conversion to float
    REQUIRE(std::fpclassify(float(float16_t {0x0001})) == FP_NORMAL);
    REQUIRE(std::fpclassify(float(float16_t {0x83FF})) == FP_NORMAL);
}

TEST_CASE("F16 scalar (compatibility)", "[scalar]") {
    const float16_t float16t {0x3C00};
    Scalar scalar = scalar::of(float16t);
    REQUIRE(scalar.dtype().variant() == DataTypeVariant::Primitive);
    REQUIRE(scalar.dtype().primitive_type() == vortex::PType::F16);
}
} // namespace
