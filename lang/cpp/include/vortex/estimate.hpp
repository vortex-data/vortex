// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include "vortex/error.hpp"
#include <vortex.h>

#include <cstdint>

namespace vortex {

enum class EstimateType {
    Unknown = VX_ESTIMATE_UNKNOWN,
    Exact = VX_ESTIMATE_EXACT,
    Inexact = VX_ESTIMATE_INEXACT,
};

// Estimated count (rows in a partition, partitions in a scan)
class Estimate {
public:
    explicit Estimate(vx_estimate raw) : raw_(raw) {
    }

    inline EstimateType type() const noexcept {
        return static_cast<EstimateType>(raw_.type);
    }

    /**
     * Estimated count. Throws if type() is Unknown. For inexact estimates this
     * is an upper bound.
     */
    inline uint64_t value() const {
        if (type() == EstimateType::Unknown) {
            throw VortexException("estimate is unknown", ErrorCode::InvalidArgument);
        }
        return raw_.estimate;
    }

    inline uint64_t value_or(uint64_t fallback) const noexcept {
        return type() == EstimateType::Unknown ? fallback : raw_.estimate;
    }

private:
    vx_estimate raw_;
};

} // namespace vortex
