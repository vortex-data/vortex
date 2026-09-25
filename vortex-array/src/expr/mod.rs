// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex's expression language: scalar operations over [arrays](crate::ArrayRef).
//!
//! A [`BoundExpression`] is a typed tree of scalar operations rooted at a scope. Scans accept a
//! bound filter and projection, so integrations resolve field names, casts, and function overloads
//! before submission. User-authored expressions live in `vortex-expr`.
//!
//! # Scalar functions
//!
//! Each node references a scalar function defined by a
//! [`ScalarFnVTable`](crate::scalar_fn::ScalarFnVTable). The vtable declares the function signature,
//! properties such as strictness, and the logic that executes it over input arrays. Built-in
//! functions live in [`crate::scalar_fn`]; integration and plugin crates supply additional,
//! use-case-specific functions.
//!
//! # Deferred execution
//!
//! Applying an expression to an array does not compute the result eagerly. Instead it builds a
//! [`ScalarFnArray`](crate::arrays::ScalarFnArray) representing the deferred application, letting
//! downstream encodings push the computation into compressed data, or fuse several expressions
//! together, before any data is materialized. The deferred tree is executed toward canonical form
//! only when a result is actually required.
//!
//! # Type checking
//!
//! Expressions are type-checked when constructed. Callers perform required casts before building
//! a bound expression. Comparison permits differing nullability and an extension value paired
//! with its storage dtype, but rejects incompatible logical types.
//!
//! Filter expressions are decomposed into independent conjuncts with [`split_conjunction`] so that
//! scans can evaluate and reorder the most selective predicates first.
//!
//! The implementation takes inspiration from [Postgres] and [Apache Datafusion].
//!
//! [Postgres]: https://www.postgresql.org/docs/current/sql-expressions.html
//! [Apache Datafusion]: https://github.com/apache/datafusion/tree/5fac581efbaffd0e6a9edf931182517524526afd/datafusion/expr

use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::Operator;

pub mod aliases;
pub mod analysis;
#[cfg(feature = "arbitrary")]
pub mod arbitrary;
pub mod bound_expression;
pub mod display;
mod exprs;
pub(crate) mod field;
mod optimize;
mod reduce_node;
pub mod stats;
pub mod transform;
pub mod traversal;

pub use analysis::*;
pub use bound_expression::*;
pub use exprs::*;
pub use reduce_node::BoundExpressionReduceNode;

/// Split the top-level boolean conjunction into its operands.
pub fn split_conjunction(expr: &BoundExpression) -> Vec<BoundExpression> {
    let mut conjunctions = vec![];
    split_inner(expr, &mut conjunctions);
    conjunctions
}

fn split_inner(expr: &BoundExpression, exprs: &mut Vec<BoundExpression>) {
    match expr.as_opt::<Binary>() {
        Some(operator) if *operator == Operator::And => {
            split_inner(expr.child(0), exprs);
            split_inner(expr.child(1), exprs);
        }
        Some(_) | None => exprs.push(expr.clone()),
    }
}

#[cfg(feature = "_test-harness")]
pub mod test_harness {
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;

    pub fn struct_dtype() -> DType {
        DType::Struct(
            StructFields::new(
                ["a", "col1", "col2", "bool1", "bool2"].into(),
                vec![
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Primitive(PType::U16, Nullability::Nullable),
                    DType::Primitive(PType::U16, Nullability::Nullable),
                    DType::Bool(Nullability::NonNullable),
                    DType::Bool(Nullability::NonNullable),
                ],
            ),
            Nullability::NonNullable,
        )
    }
}
