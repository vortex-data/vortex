// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/session.hpp"

#include <vortex.h>

namespace vortex {

void Session::Deleter::operator()(vx_session *ptr) const noexcept {
    vx_session_free(ptr);
}

Session::Session() : handle_(vx_session_new()) {
}

Session::Session(const Session &other) : handle_(vx_session_clone(other.handle_.get())) {
}

Session &Session::operator=(const Session &other) {
    if (this != &other) {
        handle_.reset(vx_session_clone(other.handle_.get()));
    }
    return *this;
}

} // namespace vortex
