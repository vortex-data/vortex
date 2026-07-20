// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include "vortex/common.hpp"

#include <vortex.h>

#include <memory>

namespace vortex {

/**
 * A handle to a Vortex session, registry of encodings and compute kernels.
 * Copying shares the underlying session.
 */
class Session {
public:
    Session();

    Session(const Session &other);
    Session(Session &&) noexcept = default;
    Session &operator=(const Session &other);
    Session &operator=(Session &&) noexcept = default;

private:
    friend struct detail::Access;

    struct Deleter {
        void operator()(vx_session *ptr) const noexcept;
    };
    std::unique_ptr<vx_session, Deleter> handle_;
};

} // namespace vortex
