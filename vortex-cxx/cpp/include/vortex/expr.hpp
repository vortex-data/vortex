// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "vortex/scalar.hpp"
#include "vortex_cxx_bridge/lib.h"
#include <vector>

namespace vortex::expr {
using scalar::Scalar;
class Expr {
public:
    Expr() = delete;
    explicit Expr(rust::Box<ffi::Expr> impl) : impl_(std::move(impl)) {
    }
    Expr(Expr &&other) noexcept = default;
    Expr &operator=(Expr &&other) noexcept = default;
    ~Expr() = default;

    Expr(const Expr &) = delete;
    Expr &operator=(const Expr &) = delete;

    rust::Box<ffi::Expr> IntoImpl() && {
        return std::move(impl_);
    }

private:
    rust::Box<ffi::Expr> impl_;
};

Expr literal(Scalar scalar);
Expr root();
Expr column(std::string_view name);
Expr get_item(std::string_view field, Expr expr);
Expr not_(Expr expr);
Expr is_null(Expr expr);
Expr eq(Expr lhs, Expr rhs);
Expr not_eq_(Expr lhs, Expr rhs);
Expr gt(Expr lhs, Expr rhs);
Expr gt_eq(Expr lhs, Expr rhs);
Expr lt(Expr lhs, Expr rhs);
Expr lt_eq(Expr lhs, Expr rhs);
Expr and_(Expr lhs, Expr rhs);
Expr or_(Expr lhs, Expr rhs);
Expr checked_add(Expr lhs, Expr rhs);
Expr select(const std::vector<std::string_view> &fields, Expr child);
} // namespace vortex::expr