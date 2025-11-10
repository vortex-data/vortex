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

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arcref::ArcRef;
use vortex_dtype::FieldName;
use vortex_error::VortexUnwrap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::expr::traversal::{NodeExt, ReferenceCollector};

pub mod aliases;
mod analysis;
#[cfg(feature = "arbitrary")]
pub mod arbitrary;
pub mod display;
mod expression;
mod exprs;
mod field;
pub mod forms;
pub mod proto;
pub mod pruning;
pub mod session;
pub mod transform;
pub mod traversal;
mod view;
mod vtable;

pub use analysis::*;
pub use expression::*;
pub use exprs::*;
pub use view::*;
pub use vtable::*;

pub type ExprId = ArcRef<str>;

pub trait VortexExprExt {
    /// Accumulate all field references from this expression and its children in a set
    fn field_references(&self) -> HashSet<FieldName>;
}

impl VortexExprExt for Expression {
    fn field_references(&self) -> HashSet<FieldName> {
        let mut collector = ReferenceCollector::new();
        // The collector is infallible, so we can unwrap the result
        self.accept(&mut collector).vortex_unwrap();
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
        Some(bexp) if bexp.operator() == Operator::And => {
            split_inner(bexp.lhs(), exprs);
            split_inner(bexp.rhs(), exprs);
        }
        Some(_) | None => {
            exprs.push(expr.clone());
        }
    }
}

/// An expression wrapper that performs pointer equality.
#[derive(Clone)]
pub struct ExactExpr(pub Expression);

impl PartialEq for ExactExpr {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id() && Arc::ptr_eq(self.0.data(), other.0.data())
    }
}
impl Eq for ExactExpr {}

impl Hash for ExactExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

#[cfg(feature = "test-harness")]
pub mod test_harness {
    use vortex_dtype::{DType, Nullability, PType, StructFields};

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
    use vortex_dtype::{DType, FieldNames, Nullability, PType, StructFields};
    use vortex_scalar::Scalar;

    use super::*;
    use crate::expr::exprs::binary::{and, eq, gt, gt_eq, lt, lt_eq, not_eq, or};
    use crate::expr::exprs::get_item::{col, get_item};
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::not::not;
    use crate::expr::exprs::root::root;
    use crate::expr::exprs::select::{select, select_exclude};

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

        assert_eq!(not(col1).to_string(), "not($.col1)");

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
