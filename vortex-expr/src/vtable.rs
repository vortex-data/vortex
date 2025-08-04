// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::Deref;

use vortex_array::{ArrayRef, DeserializeMetadata, SerializeMetadata};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{
    AnalysisExpr, ExprEncoding, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, VortexExpr,
};

pub trait VTable: 'static + Sized + Send + Sync + Debug {
    type Expr: 'static
        + Send
        + Sync
        + Clone
        + Debug
        + Display
        + PartialEq
        + Hash
        + Deref<Target = dyn VortexExpr>
        + IntoExpr
        + AnalysisExpr;
    type Encoding: 'static + Send + Sync + Deref<Target = dyn ExprEncoding>;
    type Metadata: SerializeMetadata + DeserializeMetadata + Debug;

    /// Returns the ID of the expr encoding.
    fn id(encoding: &Self::Encoding) -> ExprId;

    /// Returns the encoding for the expr.
    fn encoding(expr: &Self::Expr) -> ExprEncodingRef;

    /// Returns the serialize-able metadata for the expr, or `None` if serialization is not
    /// supported.
    fn metadata(expr: &Self::Expr) -> Option<Self::Metadata>;

    /// Returns the children of the expr.
    fn children(expr: &Self::Expr) -> Vec<&ExprRef>;

    /// Return a new instance of the expression with the children replaced.
    ///
    /// ## Preconditions
    ///
    /// The number of children will match the current number of children in the expression.
    fn with_children(expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr>;

    /// Construct a new [`VortexExpr`] from the provided parts.
    fn build(
        encoding: &Self::Encoding,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr>;

    /// Evaluate the expression in the given scope.
    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef>;

    /// Compute the return [`DType`] of the expression if evaluated in the given scope.
    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType>;
}

#[macro_export]
macro_rules! vtable {
    ($V:ident) => {
        $crate::aliases::paste::paste! {
            #[derive(Debug)]
            pub struct [<$V VTable>];

            impl AsRef<dyn $crate::VortexExpr> for [<$V Expr>] {
                fn as_ref(&self) -> &dyn $crate::VortexExpr {
                    // We can unsafe cast ourselves to a ExprAdapter.
                    unsafe { &*(self as *const [<$V Expr>] as *const $crate::ExprAdapter<[<$V VTable>]>) }
                }
            }

            impl std::ops::Deref for [<$V Expr>] {
                type Target = dyn $crate::VortexExpr;

                fn deref(&self) -> &Self::Target {
                    // We can unsafe cast ourselves to an ExprAdapter.
                    unsafe { &*(self as *const [<$V Expr>] as *const $crate::ExprAdapter<[<$V VTable>]>) }
                }
            }

            impl $crate::IntoExpr for [<$V Expr>] {
                fn into_expr(self) -> $crate::ExprRef {
                    // We can unsafe transmute ourselves to an ExprAdapter.
                    std::sync::Arc::new(unsafe { std::mem::transmute::<[<$V Expr>], $crate::ExprAdapter::<[<$V VTable>]>>(self) })
                }
            }

            impl From<[<$V Expr>]> for $crate::ExprRef {
                fn from(value: [<$V Expr>]) -> $crate::ExprRef {
                    use $crate::IntoExpr;
                    value.into_expr()
                }
            }

            impl AsRef<dyn $crate::ExprEncoding> for [<$V ExprEncoding>] {
                fn as_ref(&self) -> &dyn $crate::ExprEncoding {
                    // We can unsafe cast ourselves to an ExprEncodingAdapter.
                    unsafe { &*(self as *const [<$V ExprEncoding>] as *const $crate::ExprEncodingAdapter<[<$V VTable>]>) }
                }
            }

            impl std::ops::Deref for [<$V ExprEncoding>] {
                type Target = dyn $crate::ExprEncoding;

                fn deref(&self) -> &Self::Target {
                    // We can unsafe cast ourselves to an ExprEncodingAdapter.
                    unsafe { &*(self as *const [<$V ExprEncoding>] as *const $crate::ExprEncodingAdapter<[<$V VTable>]>) }
                }
            }
        }
    };
}

#[cfg(test)]
mod tests {

    use rstest::{fixture, rstest};

    use super::*;
    use crate::proto::{ExprSerializeProtoExt, deserialize_expr_proto};
    use crate::*;

    #[fixture]
    #[once]
    fn registry() -> ExprRegistry {
        ExprRegistry::default()
    }

    #[rstest]
    // Root and selection expressions
    #[case(root())]
    #[case(select(["hello", "world"], root()))]
    #[case(select_exclude(["world", "hello"], root()))]
    // Literal expressions
    #[case(lit(42i32))]
    #[case(lit(std::f64::consts::PI))]
    #[case(lit(true))]
    #[case(lit("hello"))]
    // Column access expressions
    #[case(col("column_name"))]
    #[case(get_item("field", root()))]
    // Binary comparison expressions
    #[case(eq(col("a"), lit(10)))]
    #[case(not_eq(col("a"), lit(10)))]
    #[case(gt(col("a"), lit(10)))]
    #[case(gt_eq(col("a"), lit(10)))]
    #[case(lt(col("a"), lit(10)))]
    #[case(lt_eq(col("a"), lit(10)))]
    // Logical expressions
    #[case(and(col("a"), col("b")))]
    #[case(or(col("a"), col("b")))]
    #[case(not(col("a")))]
    // Arithmetic expressions
    #[case(checked_add(col("a"), lit(5)))]
    // Null check expressions
    #[case(is_null(col("nullable_col")))]
    // Type casting expressions
    #[case(cast(
        col("a"),
        DType::Primitive(vortex_dtype::PType::I64, vortex_dtype::Nullability::NonNullable)
    ))]
    // Between expressions
    #[case(between(col("a"), lit(10), lit(20), vortex_array::compute::BetweenOptions { lower_strict: vortex_array::compute::StrictComparison::NonStrict, upper_strict: vortex_array::compute::StrictComparison::NonStrict }))]
    // List contains expressions
    #[case(list_contains(col("list_col"), lit("item")))]
    // Pack expressions - creating struct from fields
    #[case(pack([("field1", col("a")), ("field2", col("b"))], vortex_dtype::Nullability::NonNullable))]
    // Merge expressions - merging struct expressions
    #[case(merge([col("struct1"), col("struct2")], vortex_dtype::Nullability::NonNullable))]
    // Complex nested expressions
    #[case(and(gt(col("a"), lit(0)), lt(col("a"), lit(100))))]
    #[case(or(is_null(col("a")), eq(col("a"), lit(0))))]
    #[case(not(and(eq(col("status"), lit("active")), gt(col("age"), lit(18)))))]
    fn text_expr_serde_round_trip(
        registry: &ExprRegistry,
        #[case] expr: ExprRef,
    ) -> anyhow::Result<()> {
        let serialized_pb = expr.serialize_proto()?;
        let deserialized_expr = deserialize_expr_proto(&serialized_pb, registry)?;

        assert_eq!(&expr, &deserialized_expr);

        Ok(())
    }
}
