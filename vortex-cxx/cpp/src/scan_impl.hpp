// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "vortex/scan.hpp"
#include "vortex_cxx_bridge/read.h"

namespace vortex {

struct ScanBuilder::Impl {
    rust::Box<ffi::VortexScanBuilder> rust_impl;

    explicit Impl(rust::Box<ffi::VortexScanBuilder> impl) : rust_impl(std::move(impl)) {
    }
};

} // namespace vortex