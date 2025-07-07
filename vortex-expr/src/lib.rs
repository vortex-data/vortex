// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::sync::Arc;

use dyn_hash::DynHash;
pub use exprs::*;
mod analysis;
#[cfg(feature = "arbitrary")]
pub mod arbitrary;
mod exprs;
mod field;
pub mod forms;
pub mod pruning;
#[cfg(feature = "proto")]
mod registry;
mod scope;
mod scope_vars;
pub mod transform;
pub mod traversal;

pub use analysis::*;
pub use between::*;
pub use binary::*;
pub use cast::*;
pub use get_item::*;
pub use is_null::*;
pub use like::*;
pub use list_contains::*;
pub use literal::*;
pub use merge::*;
pub use not::*;
pub use operators::*;
pub use pack::*;
#[cfg(feature = "proto")]
pub use registry::deserialize_expr;
pub use scope::*;
pub use select::*;
pub use var::*;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::{DType, FieldName, FieldPath};
use vortex_error::{VortexResult, VortexUnwrap};
#[cfg(feature = "proto")]
use vortex_proto::expr;
#[cfg(feature = "proto")]
use vortex_proto::expr::{Expr, kind};
use vortex_utils::aliases::hash_set::HashSet;

use crate::traversal::{Node, ReferenceCollector, VarsCollector};

pub type ExprRef = Arc<dyn VortexExpr>;

#[cfg(feature = "proto")]
pub trait Id {
    fn id(&self) -> &'static str;
}

#[cfg(feature = "proto")]
pub trait ExprDeserialize: Id + Sync {
    fn deserialize(&self, kind: &kind::Kind, children: Vec<ExprRef>) -> VortexResult<ExprRef>;
}

#[cfg(feature = "proto")]
pub trait ExprSerializable {
    fn id(&self) -> &'static str;

    fn serialize_kind(&self) -> VortexResult<kind::Kind>;
}

#[cfg(not(feature = "proto"))]
pub trait ExprSerializable {}
#[cfg(not(feature = "proto"))]
impl<T> ExprSerializable for T {}
/// Represents logical operation on [`ArrayRef`]s
pub trait VortexExpr:
    Debug + Send + Sync + DynEq + DynHash + Display + ExprSerializable + AnalysisExpr
{
    /// Convert expression reference to reference of [`Any`] type
    fn as_any(&self) -> &dyn Any;

    /// Compute result of expression on given batch producing a new batch
    fn evaluate(&self, scope: &Scope) -> VortexResult<ArrayRef> {
        let result = self.unchecked_evaluate(scope)?;
        assert_eq!(
            result.dtype(),
            &self.return_dtype(&scope.into())?,
            "Expression {} returned dtype {} but declared return_dtype of {}",
            self,
            result.dtype(),
            self.return_dtype(&scope.into())?,
        );
        Ok(result)
    }

    /// Compute result of expression on given batch producing a new batch
    ///
    /// "Unchecked" means that this function lacks a debug assertion that the returned array matches
    /// the [VortexExpr::return_dtype] method. Use instead the [VortexExpr::evaluate] function which
    /// includes such an assertion.
    fn unchecked_evaluate(&self, ctx: &Scope) -> VortexResult<ArrayRef>;

    fn children(&self) -> Vec<&ExprRef>;

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef;

    /// Compute the type of the array returned by [VortexExpr::evaluate].
    fn return_dtype(&self, scope: &ScopeDType) -> VortexResult<DType>;
}

pub trait VortexExprExt {
    /// Accumulate all field references from this expression and its children in a set
    fn field_references(&self) -> HashSet<FieldName>;

    fn vars(&self) -> HashSet<Identifier>;

    #[cfg(feature = "proto")]
    fn serialize(&self) -> VortexResult<Expr>;
}

impl VortexExprExt for ExprRef {
    fn field_references(&self) -> HashSet<FieldName> {
        let mut collector = ReferenceCollector::new();
        // The collector is infallible, so we can unwrap the result
        self.accept(&mut collector).vortex_unwrap();
        collector.into_fields()
    }

    fn vars(&self) -> HashSet<Identifier> {
        let mut collector = VarsCollector::new();
        // The collector is infallible, so we can unwrap the result
        self.accept(&mut collector).vortex_unwrap();
        collector.into_vars()
    }

    #[cfg(feature = "proto")]
    fn serialize(&self) -> VortexResult<Expr> {
        let children = self
            .children()
            .iter()
            .map(|e| e.serialize())
            .collect::<VortexResult<_>>()?;

        Ok(Expr {
            id: self.id().to_string(),
            children,
            kind: Some(expr::Kind {
                kind: Some(self.serialize_kind()?),
            }),
        })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct AccessPath {
    field_path: FieldPath,
    identifier: Identifier,
}

impl AccessPath {
    pub fn root_field(path: FieldName) -> Self {
        Self {
            field_path: FieldPath::from_name(path),
            identifier: Identifier::Identity,
        }
    }

    pub fn new(path: FieldPath, identifier: Identifier) -> Self {
        Self {
            field_path: path,
            identifier,
        }
    }

    pub fn identifier(&self) -> &Identifier {
        &self.identifier
    }

    pub fn field_path(&self) -> &FieldPath {
        &self.field_path
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
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
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

        assert_eq!(not(col1.clone()).to_string(), "!$.col1");

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

    #[cfg(feature = "proto")]
    mod tests_proto {
        use crate::{VortexExprExt, deserialize_expr, eq, lit, root};

        #[test]
        fn round_trip_serde() {
            let expr = eq(root(), lit(1));
            let res = expr.serialize().unwrap();
            let final_ = deserialize_expr(&res).unwrap();

            assert_eq!(&expr, &final_);
        }
    }
}
