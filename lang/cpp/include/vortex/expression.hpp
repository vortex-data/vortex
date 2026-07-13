// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include "vortex/common.hpp"
#include "vortex/scalar.hpp"

#include <vortex.h>

#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace vortex {

// A node in an expression tree used for scan filters and projections.
class Expression {
public:
    Expression(const Expression &);
    Expression &operator=(const Expression &);
    Expression(Expression &&) noexcept = default;
    Expression &operator=(Expression &&) noexcept = default;

    /**
     * Extract field from a Struct. Output DataType is field's DataType.
     *
     * Errors at scan/apply time if field does not exist or if root() is
     * not a Struct.
     *
     * Example:
     *
     * Expression field = root()["age"]["nested"];
     */
    Expression operator[](std::string_view field) const;

    Expression is_null() const;

    /*
     * Extract fields from a Struct. Output DataType is a Struct.
     *
     * Errors at scan/apply time if any of fields does not exist or if root()
     * is not a Struct.
     */
    Expression select(std::span<const std::string> names) const;
    Expression select(std::span<const std::string_view> names) const;
    Expression select(std::initializer_list<std::string_view> names) const;

private:
    friend struct detail::Access;
    explicit Expression(const vx_expression *owned) : handle_(owned) {
    }
    const vx_expression *release() && noexcept {
        return handle_.release();
    }

    struct Deleter {
        void operator()(const vx_expression *ptr) const noexcept;
    };
    std::unique_ptr<const vx_expression, Deleter> handle_;
};

// Add, Sub, Mul, and Div error in runtime on overflow and underflow
enum class BinaryOperator {
    // x == y
    Eq = VX_OPERATOR_EQ,
    // x != y
    NotEq = VX_OPERATOR_NOT_EQ,
    // x > y
    Gt = VX_OPERATOR_GT,
    // x >= y
    Gte = VX_OPERATOR_GTE,
    // x < y
    Lt = VX_OPERATOR_LT,
    // x <= y
    Lte = VX_OPERATOR_LTE,
    // boolean x AND y, Kleene semantics
    KleeneAnd = VX_OPERATOR_KLEENE_AND,
    // boolean x OR y, Kleene semantics
    KleeneOr = VX_OPERATOR_KLEENE_OR,
    // x + y
    Add = VX_OPERATOR_ADD,
    // x - y
    Sub = VX_OPERATOR_SUB,
    // x * y
    Mul = VX_OPERATOR_MUL,
    // x / y
    Div = VX_OPERATOR_DIV,
};

namespace expr {

// scanned/applied array.
Expression root();

// root()'s named column.
Expression col(std::string_view name);

// Literal expression
Expression lit(const Scalar &value);

/*
 * Literal expression.
 *
 * Literal's DataType must match column it's compared against, otherwise scan
 * fails at runtime. No type coercion is performed.
 */
template <element_type T>
Expression lit(T value) {
    return lit(scalar::of<T>(value));
}

Expression eq(const Expression &l, const Expression &r);
Expression neq(const Expression &l, const Expression &r);
Expression lt(const Expression &l, const Expression &r);
Expression lte(const Expression &l, const Expression &r);
Expression gt(const Expression &l, const Expression &r);
Expression gte(const Expression &l, const Expression &r);
Expression add(const Expression &l, const Expression &r);
Expression sub(const Expression &l, const Expression &r);
Expression mul(const Expression &l, const Expression &r);
Expression div(const Expression &l, const Expression &r);
Expression binary_op(BinaryOperator op, const Expression &l, const Expression &r);

// Kleene AND of children. Errors on an empty list.
Expression and_all(std::span<const Expression> children);
// Kleene OR of children. Errors on an empty list.
Expression or_all(std::span<const Expression> children);

Expression logical_not(const Expression &child);
Expression is_null(const Expression &child);
Expression list_contains(const Expression &list, const Expression &value);

/**
 * Opt-in operator overloads like in Eigen.
 * Note && and || don't short-circuit.
 */
namespace ops {

inline Expression operator==(const Expression &l, const Expression &r) {
    return eq(l, r);
}
inline Expression operator!=(const Expression &l, const Expression &r) {
    return neq(l, r);
}
inline Expression operator<(const Expression &l, const Expression &r) {
    return lt(l, r);
}
inline Expression operator<=(const Expression &l, const Expression &r) {
    return lte(l, r);
}
inline Expression operator>(const Expression &l, const Expression &r) {
    return gt(l, r);
}
inline Expression operator>=(const Expression &l, const Expression &r) {
    return gte(l, r);
}
inline Expression operator&&(const Expression &l, const Expression &r) {
    return binary_op(BinaryOperator::KleeneAnd, l, r);
}
inline Expression operator||(const Expression &l, const Expression &r) {
    return binary_op(BinaryOperator::KleeneOr, l, r);
}
inline Expression operator!(const Expression &e) {
    return logical_not(e);
}
inline Expression operator+(const Expression &l, const Expression &r) {
    return add(l, r);
}
inline Expression operator-(const Expression &l, const Expression &r) {
    return sub(l, r);
}
inline Expression operator*(const Expression &l, const Expression &r) {
    return mul(l, r);
}
inline Expression operator/(const Expression &l, const Expression &r) {
    return div(l, r);
}

} // namespace ops
} // namespace expr
} // namespace vortex
