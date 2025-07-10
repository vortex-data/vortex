// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Display, Formatter};

use vortex_array::{ArrayRef, DeserializeMetadata, EmptyMetadata};
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_expr::{
    AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, ScopeDType, VTable, vtable,
};

vtable!(RowIdx);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RowIdxExpr;

impl AnalysisExpr for RowIdxExpr {}

impl Display for RowIdxExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "#row_idx")
    }
}

#[derive(Clone)]
pub struct RowIdxExprEncoding;

impl VTable for RowIdxVTable {
    type Expr = RowIdxExpr;
    type Encoding = RowIdxExprEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("vortex.row_idx")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(RowIdxExprEncoding.as_ref())
    }

    fn metadata(_expr: &Self::Expr) -> Option<Self::Metadata> {
        // Serializable, but with no metadata
        Some(EmptyMetadata)
    }

    fn children(_expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![]
    }

    fn with_children(expr: &Self::Expr, _children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        Ok(expr.clone())
    }

    fn build(
        _encoding: &Self::Encoding,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        if !children.is_empty() {
            vortex_bail!(
                "RowIdxExpr does not expect any children, got {}",
                children.len()
            );
        }
        Ok(RowIdxExpr)
    }

    fn evaluate(_expr: &Self::Expr, _scope: &Scope) -> VortexResult<ArrayRef> {
        vortex_bail!(
            "RowIdxExpr should not be evaluated directly, use it in the context of a Vortex scan and it will be substituted for a row index array"
        );
    }

    fn return_dtype(_expr: &Self::Expr, _scope: &ScopeDType) -> VortexResult<DType> {
        Ok(DType::Primitive(PType::U64, Nullability::NonNullable))
    }
}

pub fn row_idx() -> ExprRef {
    RowIdxExpr.into_expr()
}
