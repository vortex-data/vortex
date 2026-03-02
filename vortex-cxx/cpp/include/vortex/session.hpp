// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "vortex_cxx_bridge/lib.h"

namespace vortex {

class VortexSession {
public:
    VortexSession() : impl_(ffi::session_new()) {
    }
    VortexSession(VortexSession &&other) noexcept = default;
    VortexSession &operator=(VortexSession &&other) noexcept = default;
    ~VortexSession() = default;

    VortexSession(const VortexSession &) = delete;
    VortexSession &operator=(const VortexSession &) = delete;

    const rust::Box<ffi::VortexSession> &GetImpl() const {
        return impl_;
    }

private:
    rust::Box<ffi::VortexSession> impl_;
};

} // namespace vortex
