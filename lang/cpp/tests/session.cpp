// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <string_view>
#include <vortex/array.hpp>
#include <vortex/error.hpp>
#include <vortex/session.hpp>

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

TEST_CASE("Session enables a named edition", "[session]") {
    Session session;
    const std::string_view text = VX_EDITION_CORE_2025_10_0 " suffix";
    REQUIRE_NOTHROW(session.enable_edition(text.substr(0, text.find(' '))));
    REQUIRE_THROWS_AS(session.enable_edition("unknown-edition"), VortexException);
}
} // namespace
