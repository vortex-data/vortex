// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/thread_pool.hpp"
#include "vortex/exception.hpp"

#include "rust/cxx.h"
#include "vortex_cxx_bridge/lib.h"

namespace vortex {

void ConfigureThreadPool(size_t worker_threads) {
    try {
        ffi::configure_thread_pool(worker_threads);
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

} // namespace vortex