// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use dyn_hash::DynHash;
pub use exprs::*;
pub mod aliases;
mod analysis;
#[cfg(feature = "arbitrary")]
pub mod arbitrary;
pub mod dyn_traits;
mod encoding;
mod exprs;
mod field;
pub mod forms;
pub mod proto;
pub mod pruning;
mod registry;
mod scope;
mod scope_vars;
pub mod transform;
pub mod traversal;
mod vtable;

pub use analysis::*;
pub use between::*;
pub use binary::*;
pub use cast::*;
pub use encoding::*;
pub use get_item::*;
pub use is_null::*;
pub use like::*;
pub use list_contains::*;
pub use literal::*;
pub use merge::*;
pub use not::*;
pub use operators::*;
pub use pack::*;
pub use registry::*;
pub use root::*;
pub use scope::*;
pub use select::*;
use vortex_array::{Array, ArrayRef, SerializeMetadata};
use vortex_dtype::{DType, FieldName, FieldPath};
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap, vortex_bail};
use vortex_utils::aliases::hash_set::HashSet;
pub use vtable::*;

use crate::dyn_traits::DynEq;
use crate::traversal::{NodeExt, ReferenceCollector};

pub trait IntoExpr {
    /// Convert this type into an expression reference.
    fn into_expr(self) -> ExprRef;
}

pub type ExprRef = Arc<dyn VortexExpr>;

/// Represents logical operation on [`ArrayRef`]s
pub trait VortexExpr:
    'static + Send + Sync + Debug + Display + DynEq + DynHash + private::Sealed + AnalysisExpr
{
    /// Convert expression reference to reference of [`Any`] type
    fn as_any(&self) -> &dyn Any;

    /// Convert the expression to an [`ExprRef`].
    fn to_expr(&self) -> ExprRef;

    /// Return the encoding of the expression.
    fn encoding(&self) -> ExprEncodingRef;

    /// Serialize the metadata of this expression into a bytes vector.
    ///
    /// Returns `None` if the expression does not support serialization.
    fn metadata(&self) -> Option<Vec<u8>> {
        None
    }

    /// Compute result of expression on given batch producing a new batch
    ///
    /// "Unchecked" means that this function lacks a debug assertion that the returned array matches
    /// the [VortexExpr::return_dtype] method. Use instead the
    /// [`VortexExpr::evaluate`](./trait.VortexExpr.html#method.evaluate).
    /// function which includes such an assertion.
    fn unchecked_evaluate(&self, ctx: &Scope) -> VortexResult<ArrayRef>;

    /// Returns the children of this expression.
    fn children(&self) -> Vec<&ExprRef>;

    /// Returns a new instance of this expression with the children replaced.
    fn with_children(self: Arc<Self>, children: Vec<ExprRef>) -> VortexResult<ExprRef>;

    /// Compute the type of the array returned by
    /// [`VortexExpr::evaluate`](./trait.VortexExpr.html#method.evaluate).
    fn return_dtype(&self, scope: &DType) -> VortexResult<DType>;
}

dyn_hash::hash_trait_object!(VortexExpr);

impl PartialEq for dyn VortexExpr {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other.as_any())
    }
}

impl Eq for dyn VortexExpr {}

impl dyn VortexExpr + '_ {
    pub fn id(&self) -> ExprId {
        self.encoding().id()
    }

    pub fn is<V: VTable>(&self) -> bool {
        self.as_opt::<V>().is_some()
    }

    pub fn as_<V: VTable>(&self) -> &V::Expr {
        self.as_opt::<V>()
            .vortex_expect("Expr is not of the expected type")
    }

    pub fn as_opt<V: VTable>(&self) -> Option<&V::Expr> {
        VortexExpr::as_any(self)
            .downcast_ref::<ExprAdapter<V>>()
            .map(|e| &e.0)
    }

    /// Compute result of expression on given batch producing a new batch
    pub fn evaluate(&self, scope: &Scope) -> VortexResult<ArrayRef> {
        let result = self.unchecked_evaluate(scope)?;
        assert_eq!(
            result.dtype(),
            &self.return_dtype(scope.dtype())?,
            "Expression {} returned dtype {} but declared return_dtype of {}",
            self,
            result.dtype(),
            self.return_dtype(scope.dtype())?,
        );
        Ok(result)
    }
}

pub trait VortexExprExt {
    /// Accumulate all field references from this expression and its children in a set
    fn field_references(&self) -> HashSet<FieldName>;
}

impl VortexExprExt for ExprRef {
    fn field_references(&self) -> HashSet<FieldName> {
        let mut collector = ReferenceCollector::new();
        // The collector is infallible, so we can unwrap the result
        self.accept(&mut collector).vortex_unwrap();
        collector.into_fields()
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct ExprAdapter<V: VTable>(V::Expr);

impl<V: VTable> VortexExpr for ExprAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn to_expr(&self) -> ExprRef {
        Arc::new(ExprAdapter::<V>(self.0.clone()))
    }

    fn encoding(&self) -> ExprEncodingRef {
        V::encoding(&self.0)
    }

    fn metadata(&self) -> Option<Vec<u8>> {
        V::metadata(&self.0).map(|m| m.serialize())
    }

    fn unchecked_evaluate(&self, ctx: &Scope) -> VortexResult<ArrayRef> {
        V::evaluate(&self.0, ctx)
    }

    fn children(&self) -> Vec<&ExprRef> {
        V::children(&self.0)
    }

    fn with_children(self: Arc<Self>, children: Vec<ExprRef>) -> VortexResult<ExprRef> {
        if self.children().len() != children.len() {
            vortex_bail!(
                "Expected {} children, got {}",
                self.children().len(),
                children.len()
            );
        }
        Ok(V::with_children(&self.0, children)?.to_expr())
    }

    fn return_dtype(&self, scope: &DType) -> VortexResult<DType> {
        V::return_dtype(&self.0, scope)
    }
}

impl<V: VTable> Debug for ExprAdapter<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl<V: VTable> Display for ExprAdapter<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl<V: VTable> PartialEq for ExprAdapter<V> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&self.0, &other.0)
    }
}

impl<V: VTable> Eq for ExprAdapter<V> {}

impl<V: VTable> Hash for ExprAdapter<V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&self.0, state);
    }
}

impl<V: VTable> AnalysisExpr for ExprAdapter<V> {
    fn stat_falsification(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        <V::Expr as AnalysisExpr>::stat_falsification(&self.0, catalog)
    }

    fn max(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        <V::Expr as AnalysisExpr>::max(&self.0, catalog)
    }

    fn min(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        <V::Expr as AnalysisExpr>::min(&self.0, catalog)
    }

    fn nan_count(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        <V::Expr as AnalysisExpr>::nan_count(&self.0, catalog)
    }

    fn field_path(&self) -> Option<FieldPath> {
        <V::Expr as AnalysisExpr>::field_path(&self.0)
    }
}

mod private {
    use super::*;

    pub trait Sealed {}

    impl<V: VTable> Sealed for ExprAdapter<V> {}
}

/// Splits top level and operations into separate expressions
pub fn split_conjunction(expr: &ExprRef) -> Vec<ExprRef> {
    let mut conjunctions = vec![];
    split_inner(expr, &mut conjunctions);
    conjunctions
}

fn split_inner(expr: &ExprRef, exprs: &mut Vec<ExprRef>) {
    match expr.as_opt::<BinaryVTable>() {
        Some(bexp) if bexp.op() == Operator::And => {
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
pub struct ExactExpr(pub ExprRef);

impl PartialEq for ExactExpr {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ExactExpr {}

impl Hash for ExactExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state)
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

        let col1: Arc<dyn VortexExpr> = col("col1");
        let col2: Arc<dyn VortexExpr> = col("col2");
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
            or(
                lt(col1.clone(), col2.clone()),
                not_eq(col1.clone(), col2.clone()),
            )
            .to_string(),
            "(($.col1 < $.col2) or ($.col1 != $.col2))"
        );

        assert_eq!(not(col1.clone()).to_string(), "(!$.col1)");

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
            "$~{col1, col2}"
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
