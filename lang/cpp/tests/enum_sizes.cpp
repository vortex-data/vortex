// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <magic_enum/magic_enum.hpp>

#include <vortex/array.hpp>
#include <vortex/estimate.hpp>

using magic_enum::enum_count;

static_assert(enum_count<vortex::ValidityType>() == enum_count<vx_validity_type>());
static_assert(enum_count<vortex::DataTypeVariant>() == enum_count<vx_dtype_variant>());
static_assert(enum_count<vortex::PType>() == enum_count<vx_ptype>());
static_assert(enum_count<vortex::BinaryOperator>() == enum_count<vx_binary_operator>());
static_assert(enum_count<vortex::EstimateType>() == enum_count<vx_estimate_type>());
static_assert(enum_count<vortex::ErrorCode>() == enum_count<vx_error_code>());
