// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/expr.hpp"

#include "rust/cxx.h"
#include "vortex_cxx_bridge/expr.h"

#include <stdexcept>

namespace vortex {

struct Expr::Impl {
    rust::Box<ffi::Expr> rust_impl;

    explicit Impl(rust::Box<ffi::Expr> impl) : rust_impl(std::move(impl)) {
    }
};

Expr::Expr(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}

Expr::Expr(Expr &&other) noexcept : impl_(std::move(other.impl_)) {
}

Expr &Expr::operator=(Expr &&other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

Expr::~Expr() = default;

// Expr Expr::literal(std::string value, std::string dtype) {
//     auto rust_expr = ffi::literal(std::move(value), std::move(dtype));
//     if (rust_expr.has_error()) {
//         throw std::runtime_error(std::string(rust_expr.error()));
//     }
//     return Expr(std::make_unique<Expr::Impl>(std::move(rust_expr)));
// }

Expr Expr::root() {
    return Expr(std::make_unique<Expr::Impl>(ffi::root()));
}

Expr Expr::column(std::string_view name) {
    return Expr(std::make_unique<Expr::Impl>(ffi::column(rust::String(name.data(), name.length()))));
}

Expr Expr::get_item(std::string_view field, Expr child) {
    return Expr(std::make_unique<Expr::Impl>(
        ffi::get_item(rust::String(field.data(), field.length()), std::move(child.impl_->rust_impl))));
}

Expr Expr::not_(Expr child) {
    return Expr(std::make_unique<Expr::Impl>(ffi::not_(std::move(child.impl_->rust_impl))));
}

Expr Expr::is_null(Expr child) {
    return Expr(std::make_unique<Expr::Impl>(ffi::is_null(std::move(child.impl_->rust_impl))));
}

// Macro to define binary operator functions
#define DEFINE_BINARY_OP(name)                                                                               \
    Expr Expr::name(Expr lhs, Expr rhs) {                                                                    \
        auto rust_expr = ffi::name(std::move(lhs.impl_->rust_impl), std::move(rhs.impl_->rust_impl));        \
        return Expr(std::make_unique<Expr::Impl>(std::move(rust_expr)));                                     \
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

} // namespace vortex