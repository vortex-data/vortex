// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex's expression language: scalar operations over [arrays](crate::ArrayRef).
//!
//! An [`Expression`] is a tree of scalar operations rooted at a scope (see [`root`]). Expressions
//! are the common currency of scans: a scan takes a *filter* expression that resolves to a boolean
//! and a *projection* expression that shapes the output. All expressions are serializable and own
//! their own wire format, so they can be pushed down to remote sources and reconstructed on workers.
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
//! Expressions are strictly typed: an input array's dtype must match the function signature exactly,
//! so callers perform any required casts themselves before building the expression. The one
//! relaxation is nullability—for example, equality may compare a `u32` against a `u32?`, but never
//! a `u32` against an `i32`.
//!
//! Scans decompose bound filter expressions into independent conjuncts so that they can evaluate
//! and reorder the most selective predicates first.
//!
//! The implementation takes inspiration from [Postgres] and [Apache Datafusion].
//!
//! [Postgres]: https://www.postgresql.org/docs/current/sql-expressions.html
//! [Apache Datafusion]: https://github.com/apache/datafusion/tree/5fac581efbaffd0e6a9edf931182517524526afd/datafusion/expr

pub mod aliases;
#[cfg(feature = "arbitrary")]
pub mod arbitrary;
pub mod bound_expression;
pub mod display;
pub(crate) mod expression;
mod exprs;
pub mod lambda;
mod optimize;
pub mod proto;
pub mod scope;
pub mod stats;
pub mod traversal;

pub use bound_expression::*;
pub use expression::*;
pub use exprs::and;
pub use exprs::and_collect;
pub use exprs::binary;
pub use exprs::bound;
pub use exprs::cast;
pub use exprs::checked_add;
pub use exprs::col;
pub use exprs::eq;
pub use exprs::fill_null;
pub use exprs::get_item;
pub use exprs::gt;
pub use exprs::gt_eq;
pub use exprs::is_not_null;
pub use exprs::is_null;
pub use exprs::is_root;
pub use exprs::lit;
pub use exprs::lt;
pub use exprs::lt_eq;
pub use exprs::mask;
pub use exprs::not;
pub use exprs::not_eq;
pub use exprs::or;
pub use exprs::or_collect;
pub use exprs::pack;
pub use exprs::root;
pub use exprs::select;
pub use exprs::select_exclude;
pub use exprs::var;
pub use lambda::*;
pub use scope::*;

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

#[cfg(test)]
mod tests {
    use vortex_array::expr::eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;

    use crate::scalar_fn::builtin::literal::Literal;

    #[test]
    fn expr_contains() {
        let expression = &eq(root(), lit(3u64));
        assert!(expression.contains::<Literal>().unwrap());
        let expression = root();
        assert!(!expression.contains::<Literal>().unwrap());
    }
}
