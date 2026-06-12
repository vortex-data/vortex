// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex's expression language.
//!
//! All expressions are serializable, and own their own wire format.
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
pub mod placeholder;
pub mod proto;
pub mod pruning;
pub mod stats;
pub mod transform;
pub mod traversal;

pub use analysis::*;
pub use expression::*;
pub use exprs::*;
pub use pruning::StatsCatalog;

pub trait VortexExprExt {
    /// Accumulate all field references from this expression and its children in a set
    fn field_references(&self) -> HashSet<FieldName>;
}

impl VortexExprExt for BoundExpr {
    fn field_references(&self) -> HashSet<FieldName> {
        let mut collector = ReferenceCollector::new();
        // The collector is infallible, so we can unwrap the result
        self.accept(&mut collector)
            .vortex_expect("reference collector should never fail");
        collector.into_fields()
    }
}

/// Splits top level and operations into separate expressions.
pub fn split_conjunction(expr: &BoundExpr) -> Vec<BoundExpr> {
    let mut conjunctions = vec![];
    split_inner(expr, &mut conjunctions);
    conjunctions
}

fn split_inner(expr: &BoundExpr, exprs: &mut Vec<BoundExpr>) {
    match expr
        .as_call()
        .and_then(|call| call.function().as_opt::<Binary>())
    {
        Some(operator) if *operator == Operator::And => {
            split_inner(expr.child(0), exprs);
            split_inner(expr.child(1), exprs);
        }
        Some(_) | None => {
            exprs.push(expr.clone());
        }
    }
}

/// An expression wrapper that performs pointer equality on call children.
///
/// Hashing remains structural. Equality uses `(function, Arc::ptr_eq(args))` for calls, while
/// leaves use structural equality because they do not have shared child storage.
#[derive(Clone)]
pub struct ExactExpr(pub BoundExpr);
impl PartialEq for ExactExpr {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (BoundExpr::Call(lhs), BoundExpr::Call(rhs)) => {
                lhs.function() == rhs.function() && Arc::ptr_eq(lhs.args_arc(), rhs.args_arc())
            }
            _ => self.0 == other.0,
        }
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
    use crate::expr::col as expr_col;
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
    use crate::expr::root as expr_root;
    use crate::expr::select;
    use crate::expr::select_exclude;
    use crate::scalar::Scalar;

    fn scope() -> DType {
        DType::Struct(
            StructFields::from_iter([
                ("a", DType::Bool(Nullability::NonNullable)),
                ("col1", DType::Bool(Nullability::NonNullable)),
                ("col2", DType::Bool(Nullability::NonNullable)),
            ]),
            Nullability::NonNullable,
        )
    }

    fn root() -> BoundExpr {
        expr_root(scope())
    }

    fn col(field: impl Into<FieldName>) -> BoundExpr {
        expr_col(field, &scope())
    }

    #[test]
    fn basic_expr_split_test() {
        let lhs = get_item("col1", root());
        let rhs = lit(true);
        let expr = eq(lhs, rhs);
        let conjunction = split_conjunction(&expr);
        assert_eq!(conjunction.len(), 1);
    }

    #[test]
    fn basic_conjunction_split_test() {
        let lhs = eq(get_item("col1", root()), lit(true));
        let rhs = lit(true);
        let expr = and(lhs, rhs);
        let conjunction = split_conjunction(&expr);
        assert_eq!(conjunction.len(), 2, "Conjunction is {conjunction:?}");
    }

    #[test]
    fn expr_display() {
        assert_eq!(col("a").to_string(), "$.a");
        assert_eq!(root().to_string(), "$");

        let col1: BoundExpr = col("col1");
        let col2: BoundExpr = col("col2");
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
