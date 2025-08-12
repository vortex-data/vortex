// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "vortex/scalar.hpp"
#include <memory>
#include "vortex_cxx_bridge/dtype_scalar_expr.h"

namespace vortex {

class Expr {
public:
    Expr() = delete;
    Expr(Expr &&other) noexcept;
    Expr &operator=(Expr &&other) noexcept;
    ~Expr();

    Expr(const Expr &) = delete;
    Expr &operator=(const Expr &) = delete;

    static Expr literal(Scalar scalar);
    static Expr root();
    static Expr column(std::string_view name);
    static Expr get_item(std::string_view field, Expr expr);
    static Expr not_(Expr expr);
    static Expr is_null(Expr expr);
    static Expr eq(Expr lhs, Expr rhs);
    static Expr not_eq_(Expr lhs, Expr rhs);
    static Expr gt(Expr lhs, Expr rhs);
    static Expr gt_eq(Expr lhs, Expr rhs);
    static Expr lt(Expr lhs, Expr rhs);
    static Expr lt_eq(Expr lhs, Expr rhs);
    static Expr and_(Expr lhs, Expr rhs);
    static Expr or_(Expr lhs, Expr rhs);
    static Expr checked_add(Expr lhs, Expr rhs);

private:
    friend class ScanBuilder;
    explicit Expr(rust::Box<ffi::Expr> impl);
    rust::Box<ffi::Expr> impl_;
};

} // namespace vortex