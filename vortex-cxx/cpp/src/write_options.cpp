// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/write_options.hpp"
#include "vortex/exception.hpp"

#include "rust/cxx.h"
#include "vortex_cxx_bridge/write.h"

namespace vortex {

struct VortexWriteOptions::Impl {
    rust::Box<ffi::VortexWriteOptions> rust_impl;

    explicit Impl(rust::Box<ffi::VortexWriteOptions> impl) : rust_impl(std::move(impl)) {
    }
};

VortexWriteOptions::VortexWriteOptions() : impl_(std::make_unique<Impl>(ffi::write_options_new())) {
}

VortexWriteOptions::VortexWriteOptions(VortexWriteOptions &&other) noexcept : impl_(std::move(other.impl_)) {
}

VortexWriteOptions &VortexWriteOptions::operator=(VortexWriteOptions &&other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

VortexWriteOptions::~VortexWriteOptions() = default;

void VortexWriteOptions::WriteArrayStream(ArrowArrayStream &stream, const std::string &path) {
    try {
        ffi::write_array_stream(std::move(impl_->rust_impl), reinterpret_cast<uint8_t *>(&stream), path);
    } catch (const rust::cxxbridge1::Error &e) {
        throw VortexException(e.what());
    }
}

} // namespace vortex