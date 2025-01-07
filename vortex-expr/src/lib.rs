use std::any::Any;
use std::fmt::{Debug, Display};
use std::sync::Arc;

mod binary;
mod column;
pub mod datafusion;
mod identity;
mod like;
mod literal;
mod not;
mod operators;
mod project;
pub mod pruning;
mod row_filter;
mod select;
#[allow(dead_code)]
mod traversal;

pub use binary::*;
pub use column::*;
pub use identity::*;
pub use like::*;
pub use literal::*;
pub use not::*;
pub use operators::*;
pub use project::*;
pub use row_filter::*;
pub use select::*;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::ArrayData;
use vortex_dtype::Field;
use vortex_error::{VortexResult, VortexUnwrap};

use crate::traversal::{Node, ReferenceCollector};

pub type ExprRef = Arc<dyn VortexExpr>;

/// Represents logical operation on [`ArrayData`]s
pub trait VortexExpr: Debug + Send + Sync + DynEq + Display {
    /// Convert expression reference to reference of [`Any`] type
    fn as_any(&self) -> &dyn Any;

    /// Compute result of expression on given batch producing a new batch
    fn evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData>;

    fn children(&self) -> Vec<&ExprRef>;

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef;
}

pub trait VortexExprExt {
    /// Accumulate all field references from this expression and its children in a set
    fn references(&self) -> HashSet<&Field>;
}

impl VortexExprExt for ExprRef {
    fn references(&self) -> HashSet<&Field> {
        let mut collector = ReferenceCollector::new();
        // The collector is infallible, so we can unwrap the result
        self.accept(&mut collector).vortex_unwrap();
        collector.into_fields()
    }
}

/// Splits top level and operations into separate expressions
pub fn split_conjunction(expr: &ExprRef) -> Vec<ExprRef> {
    let mut conjunctions = vec![];
    split_inner(expr, &mut conjunctions);
    conjunctions
}

fn split_inner(expr: &ExprRef, exprs: &mut Vec<ExprRef>) {
    match expr.as_any().downcast_ref::<BinaryExpr>() {
        Some(bexp) if bexp.op() == Operator::And => {
            split_inner(bexp.lhs(), exprs);
            split_inner(bexp.rhs(), exprs);
        }
        Some(_) | None => {
            exprs.push(expr.clone());
        }
    }
}

// Adapted from apache/datafusion https://github.com/apache/datafusion/blob/f31ca5b927c040ce03f6a3c8c8dc3d7f4ef5be34/datafusion/physical-expr-common/src/physical_expr.rs#L156
/// [`VortexExpr`] can't be constrained by [`Eq`] directly because it must remain object
/// safe. To ease implementation blanket implementation is provided for [`Eq`] types.
pub trait DynEq {
    fn dyn_eq(&self, other: &dyn Any) -> bool;
}

impl<T: Eq + Any> DynEq for T {
    fn dyn_eq(&self, other: &dyn Any) -> bool {
        other.downcast_ref::<Self>() == Some(self)
    }
}

impl PartialEq for dyn VortexExpr {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other.as_any())
    }
}

impl Eq for dyn VortexExpr {}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Field, Nullability, PType, StructDType};
    use vortex_scalar::Scalar;

    use super::*;

    #[test]
    fn basic_expr_split_test() {
        let lhs = col("a");
        let rhs = lit(1);
        let expr = eq(lhs, rhs);
        let conjunction = split_conjunction(&expr);
        assert_eq!(conjunction.len(), 1);
    }

    #[test]
    fn basic_conjunction_split_test() {
        let lhs = col("a");
        let rhs = lit(1);
        let expr = and(lhs, rhs);
        let conjunction = split_conjunction(&expr);
        assert_eq!(conjunction.len(), 2, "Conjunction is {conjunction:?}");
    }

    #[test]
    fn expr_display() {
        assert_eq!(col("a").to_string(), "$a");
        assert_eq!(col(1).to_string(), "[1]");
        assert_eq!(Identity.to_string(), "[]");
        assert_eq!(Identity.to_string(), "[]");

        let col1: Arc<dyn VortexExpr> = col("col1");
        let col2: Arc<dyn VortexExpr> = col("col2");
        assert_eq!(
            and(col1.clone(), col2.clone()).to_string(),
            "($col1 and $col2)"
        );
        assert_eq!(
            or(col1.clone(), col2.clone()).to_string(),
            "($col1 or $col2)"
        );
        assert_eq!(
            eq(col1.clone(), col2.clone()).to_string(),
            "($col1 = $col2)"
        );
        assert_eq!(
            not_eq(col1.clone(), col2.clone()).to_string(),
            "($col1 != $col2)"
        );
        assert_eq!(
            gt(col1.clone(), col2.clone()).to_string(),
            "($col1 > $col2)"
        );
        assert_eq!(
            gt_eq(col1.clone(), col2.clone()).to_string(),
            "($col1 >= $col2)"
        );
        assert_eq!(
            lt(col1.clone(), col2.clone()).to_string(),
            "($col1 < $col2)"
        );
        assert_eq!(
            lt_eq(col1.clone(), col2.clone()).to_string(),
            "($col1 <= $col2)"
        );

        assert_eq!(
            or(
                lt(col1.clone(), col2.clone()),
                not_eq(col1.clone(), col2.clone()),
            )
            .to_string(),
            "(($col1 < $col2) or ($col1 != $col2))"
        );

        assert_eq!(Not::new_expr(col1.clone()).to_string(), "!$col1");

        assert_eq!(
            Select::include(vec![Field::from("col1")]).to_string(),
            "Include($col1)"
        );
        assert_eq!(
            Select::include(vec![Field::from("col1"), Field::from("col2")]).to_string(),
            "Include($col1,$col2)"
        );
        assert_eq!(
            Select::exclude(vec![
                Field::from("col1"),
                Field::from("col2"),
                Field::Index(1),
            ])
            .to_string(),
            "Exclude($col1,$col2,[1])"
        );

        assert_eq!(lit(Scalar::from(0_u8)).to_string(), "0_u8");
        assert_eq!(lit(Scalar::from(0.0_f32)).to_string(), "0_f32");
        assert_eq!(
            lit(Scalar::from(i64::MAX)).to_string(),
            "9223372036854775807_i64"
        );
        assert_eq!(lit(Scalar::from(true)).to_string(), "true");
        assert_eq!(
            lit(Scalar::null(DType::Bool(Nullability::Nullable))).to_string(),
            "null"
        );

        assert_eq!(
            lit(Scalar::struct_(
                DType::Struct(
                    StructDType::new(
                        Arc::from([Arc::from("dog"), Arc::from("cat")]),
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
            "{dog:32_u32,cat:rufus}"
        );
    }
}
