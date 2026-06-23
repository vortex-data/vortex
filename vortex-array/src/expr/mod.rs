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
//! properties such as null-sensitivity, and the logic that executes it over input arrays. Built-in
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
//! # Typing and coercion
//!
//! Expressions are strictly typed: an input array's dtype must match the function signature exactly,
//! so callers perform any required type coercion themselves before building the expression (see the
//! [`transform`] passes). The one relaxation is null-coercion — for example, equality may compare a
//! `u32` against a `u32?`, but never a `u32` against an `i32`.
//!
//! Filter expressions are decomposed into independent conjuncts with [`split_conjunction`] so that
//! scans can evaluate and reorder the most selective predicates first.
//!
//! The implementation takes inspiration from [Postgres] and [Apache Datafusion].
//!
//! [Postgres]: https://www.postgresql.org/docs/current/sql-expressions.html
//! [Apache Datafusion]: https://github.com/apache/datafusion/tree/5fac581efbaffd0e6a9edf931182517524526afd/datafusion/expr

use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_utils::aliases::hash_set::HashSet;

use crate::dtype::FieldName;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::ReferenceCollector;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::Operator;

pub mod aliases;
pub mod analysis;
#[cfg(feature = "arbitrary")]
pub mod arbitrary;
pub mod display;
pub(crate) mod expression;
mod exprs;
pub(crate) mod field;
pub mod forms;
mod optimize;
pub mod proto;
pub mod stats;
pub mod transform;
pub mod traversal;

pub use analysis::*;
pub use expression::*;
pub use exprs::*;

pub trait VortexExprExt {
    /// Accumulate all field references from this expression and its children in a set
    fn field_references(&self) -> HashSet<FieldName>;
}

impl VortexExprExt for Expression {
    fn field_references(&self) -> HashSet<FieldName> {
        let mut collector = ReferenceCollector::new();
        // The collector is infallible, so we can unwrap the result
        self.accept(&mut collector)
            .vortex_expect("reference collector should never fail");
        collector.into_fields()
    }
}

/// Splits top level and operations into separate expressions.
pub fn split_conjunction(expr: &Expression) -> Vec<Expression> {
    let mut conjunctions = vec![];
    split_inner(expr, &mut conjunctions);
    conjunctions
}

fn split_inner(expr: &Expression, exprs: &mut Vec<Expression>) {
    match expr.as_opt::<Binary>() {
        Some(operator) if *operator == Operator::And => {
            split_inner(expr.child(0), exprs);
            split_inner(expr.child(1), exprs);
        }
        Some(_) | None => {
            exprs.push(expr.clone());
        }
    }
}

/// An expression wrapper that performs pointer equality on child expressions.
#[derive(Clone)]
pub struct ExactExpr(pub Expression);
impl PartialEq for ExactExpr {
    fn eq(&self, other: &Self) -> bool {
        self.0.scalar_fn() == other.0.scalar_fn()
            && Arc::ptr_eq(self.0.children(), other.0.children())
    }
}
impl Eq for ExactExpr {}

impl Hash for ExactExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::and;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::get_item;
    use crate::expr::gt;
    use crate::expr::gt_eq;
    use crate::expr::lit;
    use crate::expr::lt;
    use crate::expr::lt_eq;
    use crate::expr::not;
    use crate::expr::not_eq;
    use crate::expr::or;
    use crate::expr::root;
    use crate::expr::select;
    use crate::expr::select_exclude;
    use crate::scalar::Scalar;

    #[test]
    fn basic_expr_split_test() {
        let lhs = get_item("col1", root());
        let rhs = lit(1);
        let expr = eq(lhs, rhs);
        let conjunction = split_conjunction(&expr);
        assert_eq!(conjunction.len(), 1);
    }

    #[test]
    fn basic_conjunction_split_test() {
        let lhs = get_item("col1", root());
        let rhs = lit(1);
        let expr = and(lhs, rhs);
        let conjunction = split_conjunction(&expr);
        assert_eq!(conjunction.len(), 2, "Conjunction is {conjunction:?}");
    }

    #[test]
    fn expr_display() {
        assert_eq!(col("a").to_string(), "$.a");
        assert_eq!(root().to_string(), "$");

        let col1: Expression = col("col1");
        let col2: Expression = col("col2");
        assert_eq!(
            and(col1.clone(), col2.clone()).to_string(),
            "($.col1 and $.col2)"
        );
        assert_eq!(
            or(col1.clone(), col2.clone()).to_string(),
            "($.col1 or $.col2)"
        );
        assert_eq!(
            eq(col1.clone(), col2.clone()).to_string(),
            "($.col1 = $.col2)"
        );
        assert_eq!(
            not_eq(col1.clone(), col2.clone()).to_string(),
            "($.col1 != $.col2)"
        );
        assert_eq!(
            gt(col1.clone(), col2.clone()).to_string(),
            "($.col1 > $.col2)"
        );
        assert_eq!(
            gt_eq(col1.clone(), col2.clone()).to_string(),
            "($.col1 >= $.col2)"
        );
        assert_eq!(
            lt(col1.clone(), col2.clone()).to_string(),
            "($.col1 < $.col2)"
        );
        assert_eq!(
            lt_eq(col1.clone(), col2.clone()).to_string(),
            "($.col1 <= $.col2)"
        );

        assert_eq!(
            or(lt(col1.clone(), col2.clone()), not_eq(col1.clone(), col2),).to_string(),
            "(($.col1 < $.col2) or ($.col1 != $.col2))"
        );

        assert_eq!(not(col1).to_string(), "vortex.not($.col1)");

        assert_eq!(
            select(vec![FieldName::from("col1")], root()).to_string(),
            "${col1}"
        );
        assert_eq!(
            select(
                vec![FieldName::from("col1"), FieldName::from("col2")],
                root()
            )
            .to_string(),
            "${col1, col2}"
        );
        assert_eq!(
            select_exclude(
                vec![FieldName::from("col1"), FieldName::from("col2")],
                root()
            )
            .to_string(),
            "${~ col1, col2}"
        );

        assert_eq!(lit(Scalar::from(0u8)).to_string(), "0u8");
        assert_eq!(lit(Scalar::from(0.0f32)).to_string(), "0f32");
        assert_eq!(
            lit(Scalar::from(i64::MAX)).to_string(),
            "9223372036854775807i64"
        );
        assert_eq!(lit(Scalar::from(true)).to_string(), "true");
        assert_eq!(
            lit(Scalar::null(DType::Bool(Nullability::Nullable))).to_string(),
            "null"
        );

        assert_eq!(
            lit(Scalar::struct_(
                DType::Struct(
                    StructFields::new(
                        FieldNames::from(["dog", "cat"]),
                        vec![
                            DType::Primitive(PType::U32, Nullability::NonNullable),
                            DType::Utf8(Nullability::NonNullable)
                        ],
                    ),
                    Nullability::NonNullable
                ),
                vec![Scalar::from(32_u32), Scalar::from("rufus".to_string())]
            ))
            .to_string(),
            "{dog: 32u32, cat: \"rufus\"}"
        );
    }
}
