// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <vortex/array.hpp>

using namespace vortex;

namespace {
TEST_CASE("Session copy and move", "[session]") {
    const Session s;
    Session other;
    other = s;
    Session moved;
    moved = std::move(other);
    Session other2(std::move(moved));
    const std::array<int32_t, 3> expected = {10, 20, 30};
    auto view = Array::primitive<int32_t>(expected).values<int32_t>(other2);
    REQUIRE(view.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(view.values()[i] == expected[i]);
    }
}
} // namespace
