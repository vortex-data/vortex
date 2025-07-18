// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <cstddef>

namespace vortex {

/// TODO(xinyu): This should be a builder/option pattern
/// Configure the thread pool with the specified number of worker threads
/// If the thread pool has already been initialized, this function will throw an exception.
void ConfigureThreadPool(size_t worker_threads);

} // namespace vortex