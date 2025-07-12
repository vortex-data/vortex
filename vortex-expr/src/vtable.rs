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
