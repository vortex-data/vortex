// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <stdexcept>
#include <string>

namespace vortex {

/// TODO(xinyu): better error handling
class VortexException : public std::runtime_error {
public:
    explicit VortexException(const std::string &message) : std::runtime_error(message) {
    }
};

} // namespace vortex