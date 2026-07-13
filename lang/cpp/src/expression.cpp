// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/common.hpp"
#include "vortex/error.hpp"
#include "vortex/expression.hpp"

#include <vortex.h>

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vortex {

using detail::Access;
using detail::throw_on_error;
using detail::to_view;

void Expression::Deleter::operator()(const vx_expression *ptr) const noexcept {
    vx_expression_free(ptr);
}

Expression::Expression(const Expression &other) : handle_(vx_expression_clone(other.handle_.get())) {
}

Expression &Expression::operator=(const Expression &other) {
    if (this != &other) {
        handle_.reset(vx_expression_clone(other.handle_.get()));
    }
    return *this;
}

Expression Expression::operator[](std::string_view field) const {
    vx_expression *out = vx_expression_get_item(to_view(field), handle_.get());
    if (out == nullptr) {
        throw VortexException("get_item: field name is not valid UTF-8", ErrorCode::InvalidArgument);
    }
    return Access::adopt<Expression>(out);
}

Expression Expression::is_null() const {
    return expr::is_null(*this);
}

template <class T>
static Expression select_impl(std::span<T> names, const vx_expression *expr) {
    std::vector<vx_view> raw;
    raw.reserve(names.size());
    for (const auto &name : names) {
        raw.push_back(to_view(name));
    }
    return Access::adopt<Expression>(vx_expression_select(raw.data(), raw.size(), expr));
}

Expression Expression::select(std::span<const std::string> names) const {
    return select_impl(names, handle_.get());
}

Expression Expression::select(std::span<const std::string_view> names) const {
    return select_impl(names, handle_.get());
}

Expression Expression::select(std::initializer_list<std::string_view> names) const {
    std::span span {names.begin(), names.end()};
    return select_impl(span, handle_.get());
}

namespace expr {

Expression root() {
    return Access::adopt<Expression>(vx_expression_root());
}

Expression col(std::string_view name) {
    return root()[name];
}

Expression lit(const Scalar &value) {
    vx_error *error = nullptr;
    vx_expression *out = vx_expression_literal(Access::c_ptr(value), &error);
    throw_on_error(error);
    return Access::adopt<Expression>(out);
}

Expression binary_op(BinaryOperator op, const Expression &l, const Expression &r) {
    return Access::adopt<Expression>(
        vx_expression_binary(static_cast<vx_binary_operator>(op), Access::c_ptr(l), Access::c_ptr(r)));
}

Expression eq(const Expression &l, const Expression &r) {
    return binary_op(BinaryOperator::Eq, l, r);
}
Expression neq(const Expression &l, const Expression &r) {
    return binary_op(BinaryOperator::NotEq, l, r);
}
Expression lt(const Expression &l, const Expression &r) {
    return binary_op(BinaryOperator::Lt, l, r);
}
Expression lte(const Expression &l, const Expression &r) {
    return binary_op(BinaryOperator::Lte, l, r);
}
Expression gt(const Expression &l, const Expression &r) {
    return binary_op(BinaryOperator::Gt, l, r);
}
Expression gte(const Expression &l, const Expression &r) {
    return binary_op(BinaryOperator::Gte, l, r);
}
Expression add(const Expression &l, const Expression &r) {
    return binary_op(BinaryOperator::Add, l, r);
}
Expression sub(const Expression &l, const Expression &r) {
    return binary_op(BinaryOperator::Sub, l, r);
}
Expression mul(const Expression &l, const Expression &r) {
    return binary_op(BinaryOperator::Mul, l, r);
}
Expression div(const Expression &l, const Expression &r) {
    return binary_op(BinaryOperator::Div, l, r);
}

static Expression combine(vx_expression *(*combiner)(const vx_expression *const *, size_t),
                          std::span<const Expression> children) {
    std::vector<const vx_expression *> raw;
    raw.reserve(children.size());
    for (const auto &child : children) {
        raw.push_back(Access::c_ptr(child));
    }
    vx_expression *out = combiner(raw.data(), raw.size());
    if (out == nullptr) {
        throw VortexException("empty expression list", ErrorCode::InvalidArgument);
    }
    return Access::adopt<Expression>(out);
}

Expression and_all(std::span<const Expression> children) {
    return combine(vx_expression_and, children);
}

Expression or_all(std::span<const Expression> children) {
    return combine(vx_expression_or, children);
}

Expression logical_not(const Expression &child) {
    return Access::adopt<Expression>(vx_expression_not(Access::c_ptr(child)));
}

Expression is_null(const Expression &child) {
    return Access::adopt<Expression>(vx_expression_is_null(Access::c_ptr(child)));
}

Expression list_contains(const Expression &list, const Expression &value) {
    return Access::adopt<Expression>(vx_expression_list_contains(Access::c_ptr(list), Access::c_ptr(value)));
}

} // namespace expr

} // namespace vortex
