// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <catch2/catch_test_macros.hpp>
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
}
} // namespace
