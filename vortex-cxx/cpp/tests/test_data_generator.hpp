// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <nanoarrow/nanoarrow.h>

namespace vortex {
namespace testing {

    /// Create test data with structure {a: [10, 20, 30, 40, 50], b: [10, 20, 30, 40, 50]}
    /// This stream only has one Array
    ArrowArrayStream CreateTestDataStream();

    /// Create 1M rows of test data with structure {id: [0..1M], value: [0, 2, 4, ..., 2M]}
    /// This stream only has one Array
    ArrowArrayStream CreateTestData1MStream();

} // namespace testing
} // namespace vortex