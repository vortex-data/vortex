// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "vortex/scalar.hpp"
#include "vortex_cxx_bridge/lib.h"
#include <vector>

namespace vortex::expr {
class Expr {
public:
    Expr() = delete;
    explicit Expr(rust::Box<ffi::Expr> impl) : impl(std::move(impl)) {
    }
    Expr(Expr &&other) noexcept = default;
    Expr &operator=(Expr &&other) noexcept = default;
    ~Expr() = default;

    Expr(const Expr &) = delete;
    Expr &operator=(const Expr &) = delete;

    rust::Box<ffi::Expr> IntoImpl() && {
        return std::move(impl);
    }

    const ffi::Expr &Impl() const & {
        return *impl;
    }

private:
    rust::Box<ffi::Expr> impl;
};

Expr Literal(scalar::Scalar scalar);
Expr Root();
Expr Column(std::string_view name);
Expr GetItem(std::string_view field, Expr expr);
Expr Not(Expr expr);
Expr IsNull(Expr expr);
Expr Eq(Expr lhs, Expr rhs);
Expr NotEq(Expr lhs, Expr rhs);
Expr Gt(Expr lhs, Expr rhs);
Expr GtEq(Expr lhs, Expr rhs);
Expr Lt(Expr lhs, Expr rhs);
Expr LtEq(Expr lhs, Expr rhs);
Expr And(Expr lhs, Expr rhs);
Expr Or(Expr lhs, Expr rhs);
Expr CheckedAdd(Expr lhs, Expr rhs);
Expr Select(const std::vector<std::string_view> &fields, Expr child);
} // namespace vortex::expr