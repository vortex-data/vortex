// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/expr.hpp"

#include "rust/cxx.h"
#include "vortex_cxx_bridge/expr.h"

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

Expr Expr::get_item(std::string field, Expr child) {
    auto rust_expr = ffi::get_item(field, std::move(child.impl_->rust_impl));
    return Expr(std::make_unique<Expr::Impl>(std::move(rust_expr)));
}

} // namespace vortex