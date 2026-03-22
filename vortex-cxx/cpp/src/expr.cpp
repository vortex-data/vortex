// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/expr.hpp"
#include "vortex_cxx_bridge/lib.h"

namespace vortex::expr {

Expr literal(scalar::Scalar scalar) {
    return Expr(ffi::literal(std::move(scalar).IntoImpl()));
}

Expr root() {
    return Expr(ffi::root());
}

Expr column(std::string_view name) {
    return Expr(ffi::column(rust::String(name.data(), name.length())));
}

Expr get_item(std::string_view field, Expr child) {
    return Expr(ffi::get_item(rust::String(field.data(), field.length()), std::move(child).IntoImpl()));
}

Expr not_(Expr child) {
    return Expr(ffi::not_(std::move(child).IntoImpl()));
}

Expr is_null(Expr child) {
    return Expr(ffi::is_null(std::move(child).IntoImpl()));
}

// Macro to define binary operator functions
#define DEFINE_BINARY_OP(name)                                                                               \
    Expr name(Expr lhs, Expr rhs) {                                                                          \
        return Expr(ffi::name(std::move(lhs).IntoImpl(), std::move(rhs).IntoImpl()));                        \
    }

DEFINE_BINARY_OP(eq)
DEFINE_BINARY_OP(not_eq_)
DEFINE_BINARY_OP(gt)
DEFINE_BINARY_OP(gt_eq)
DEFINE_BINARY_OP(lt)
DEFINE_BINARY_OP(lt_eq)
DEFINE_BINARY_OP(and_)
DEFINE_BINARY_OP(or_)
DEFINE_BINARY_OP(checked_add)

#undef DEFINE_BINARY_OP

Expr select(const std::vector<std::string_view> &fields, Expr child) {
    ::rust::Vec<::rust::String> rs_fields;
    rs_fields.reserve(fields.size());
    for (std::string_view f : fields) {
        rs_fields.emplace_back(f.data(), f.length());
    }
    return Expr(ffi::select(rs_fields, std::move(child).IntoImpl()));
}

} // namespace vortex::expr