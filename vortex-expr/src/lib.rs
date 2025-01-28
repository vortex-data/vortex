extern crate core;

use std::any::Any;
use std::fmt::{Debug, Display};
use std::sync::Arc;

use dyn_hash::DynHash;

mod binary;

pub mod datafusion;
mod field;
pub mod forms;
mod get_item;
mod identity;
mod like;
mod literal;
mod merge;
mod not;
mod operators;
mod pack;
pub mod pruning;
mod select;
pub mod transform;
#[allow(dead_code)]
mod traversal;

pub use binary::*;
pub use get_item::*;
pub use identity::*;
pub use like::*;
pub use literal::*;
pub use merge::*;
pub use not::*;
pub use operators::*;
pub use pack::*;
pub use select::*;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::{ArrayDType as _, ArrayData, Canonical, IntoArrayData as _};
use vortex_dtype::{DType, FieldName};
use vortex_error::{VortexResult, VortexUnwrap};

use crate::traversal::{Node, ReferenceCollector};

pub type ExprRef = Arc<dyn VortexExpr>;

/// Represents logical operation on [`ArrayData`]s
pub trait VortexExpr: Debug + Send + Sync + DynEq + DynHash + Display {
    /// Convert expression reference to reference of [`Any`] type
    fn as_any(&self) -> &dyn Any;

    /// Compute result of expression on given batch producing a new batch
    ///
    fn evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
        let result = self.unchecked_evaluate(batch)?;
        debug_assert_eq!(result.dtype(), &self.return_dtype(batch.dtype())?);
        Ok(result)
    }

    /// Compute result of expression on given batch producing a new batch
    ///
    /// "Unchecked" means that this function lacks a debug assertion that the returned array matches
    /// the [VortexExpr::return_dtype] method. Use instead the [VortexExpr::evaluate] function which
    /// includes such an assertion.
    fn unchecked_evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData>;

    fn children(&self) -> Vec<&ExprRef>;

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef;

    /// Compute the type of the array returned by [VortexExpr::evaluate].
    fn return_dtype(&self, scope_dtype: &DType) -> VortexResult<DType> {
        let empty = Canonical::empty(scope_dtype)?.into_array();
        self.unchecked_evaluate(&empty)
            .map(|array| array.dtype().clone())
    }
}

pub trait VortexExprExt {
    /// Accumulate all field references from this expression and its children in a set
    fn references(&self) -> HashSet<FieldName>;
}

impl VortexExprExt for ExprRef {
    fn references(&self) -> HashSet<FieldName> {
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

dyn_hash::hash_trait_object!(VortexExpr);

#[cfg(feature = "test-harness")]
pub mod test_harness {
    use std::sync::Arc;

    use vortex_dtype::{DType, Nullability, PType, StructDType};

    pub fn struct_dtype() -> DType {
        DType::Struct(
            Arc::new(StructDType::new(
                [
                    "a".into(),
                    "col1".into(),
                    "col2".into(),
                    "bool1".into(),
                    "bool2".into(),
                ]
                .into(),
                vec![
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Primitive(PType::U16, Nullability::Nullable),
                    DType::Primitive(PType::U16, Nullability::Nullable),
                    DType::Bool(Nullability::NonNullable),
                    DType::Bool(Nullability::NonNullable),
                ],
            )),
            Nullability::NonNullable,
        )
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability, PType, StructDType};
    use vortex_scalar::Scalar;

    use super::*;

    #[test]
    fn basic_expr_split_test() {
        let lhs = get_item("col1", ident());
        let rhs = lit(1);
        let expr = eq(lhs, rhs);
        let conjunction = split_conjunction(&expr);
        assert_eq!(conjunction.len(), 1);
    }

    #[test]
    fn basic_conjunction_split_test() {
        let lhs = get_item("col1", ident());
        let rhs = lit(1);
        let expr = and(lhs, rhs);
        let conjunction = split_conjunction(&expr);
        assert_eq!(conjunction.len(), 2, "Conjunction is {conjunction:?}");
    }

    #[test]
    fn expr_display() {
        assert_eq!(col("a").to_string(), "[].$a");
        assert_eq!(Identity.to_string(), "[]");
        assert_eq!(Identity.to_string(), "[]");

        let col1: Arc<dyn VortexExpr> = col("col1");
        let col2: Arc<dyn VortexExpr> = col("col2");
        assert_eq!(
            and(col1.clone(), col2.clone()).to_string(),
            "([].$col1 and [].$col2)"
        );
        assert_eq!(
            or(col1.clone(), col2.clone()).to_string(),
            "([].$col1 or [].$col2)"
        );
        assert_eq!(
            eq(col1.clone(), col2.clone()).to_string(),
            "([].$col1 = [].$col2)"
        );
        assert_eq!(
            not_eq(col1.clone(), col2.clone()).to_string(),
            "([].$col1 != [].$col2)"
        );
        assert_eq!(
            gt(col1.clone(), col2.clone()).to_string(),
            "([].$col1 > [].$col2)"
        );
        assert_eq!(
            gt_eq(col1.clone(), col2.clone()).to_string(),
            "([].$col1 >= [].$col2)"
        );
        assert_eq!(
            lt(col1.clone(), col2.clone()).to_string(),
            "([].$col1 < [].$col2)"
        );
        assert_eq!(
            lt_eq(col1.clone(), col2.clone()).to_string(),
            "([].$col1 <= [].$col2)"
        );

        assert_eq!(
            or(
                lt(col1.clone(), col2.clone()),
                not_eq(col1.clone(), col2.clone()),
            )
            .to_string(),
            "(([].$col1 < [].$col2) or ([].$col1 != [].$col2))"
        );

        assert_eq!(not(col1.clone()).to_string(), "![].$col1");

        assert_eq!(
            select(vec![FieldName::from("col1")], ident()).to_string(),
            "select +($col1) []"
        );
        assert_eq!(
            select(
                vec![FieldName::from("col1"), FieldName::from("col2")],
                ident()
            )
            .to_string(),
            "select +($col1,$col2) []"
        );
        assert_eq!(
            select_exclude(
                vec![FieldName::from("col1"), FieldName::from("col2")],
                ident()
            )
            .to_string(),
            "select -($col1,$col2) []"
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
                    Arc::new(StructDType::new(
                        Arc::from([Arc::from("dog"), Arc::from("cat")]),
                        vec![
                            DType::Primitive(PType::U32, Nullability::NonNullable),
                            DType::Utf8(Nullability::NonNullable)
                        ],
                    )),
                    Nullability::NonNullable
                ),
                vec![Scalar::from(32_u32), Scalar::from("rufus".to_string())]
            ))
            .to_string(),
            "{dog:32_u32,cat:\"rufus\"}"
        );
    }
}
