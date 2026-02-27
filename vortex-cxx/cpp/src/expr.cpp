// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/expr.hpp"
#include "vortex_cxx_bridge/lib.h"

namespace vortex::expr {

Expr Literal(scalar::Scalar scalar) {
    return Expr(ffi::literal(std::move(scalar).IntoImpl()));
}

Expr Root() {
    return Expr(ffi::root());
}

Expr Column(std::string_view name) {
    return Expr(ffi::column(rust::String(name.data(), name.length())));
}

Expr GetItem(std::string_view field, Expr child) {
    return Expr(ffi::get_item(rust::String(field.data(), field.length()), std::move(child).IntoImpl()));
}

Expr Not(Expr child) {
    return Expr(ffi::not_(std::move(child).IntoImpl()));
}

Expr IsNull(Expr child) {
    return Expr(ffi::is_null(std::move(child).IntoImpl()));
}

// Macro to define binary operator functions
#define DEFINE_BINARY_OP(name)                                                                               \
    Expr name(Expr lhs, Expr rhs) {                                                                          \
        return Expr(ffi::name(std::move(lhs).IntoImpl(), std::move(rhs).IntoImpl()));                        \
    }

DEFINE_BINARY_OP(Eq)
DEFINE_BINARY_OP(NotEq)
DEFINE_BINARY_OP(Gt)
DEFINE_BINARY_OP(GtEq)
DEFINE_BINARY_OP(Lt)
DEFINE_BINARY_OP(LtEq)
DEFINE_BINARY_OP(And)
DEFINE_BINARY_OP(Or)
DEFINE_BINARY_OP(CheckedAdd)

#undef DEFINE_BINARY_OP

Expr Select(const std::vector<std::string_view> &fields, Expr child) {
    ::rust::Vec<::rust::String> rs_fields;
    for (auto f : fields) {
        rs_fields.emplace_back(f.data(), f.length());
    }
    return Expr(ffi::select(rs_fields, std::move(child).IntoImpl()));
}

} // namespace vortex::expr