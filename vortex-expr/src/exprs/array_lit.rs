// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};

use vortex_array::{ArrayRef, DeserializeMetadata, EmptyMetadata};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::display::{DisplayAs, DisplayFormat};
use crate::{AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, VTable, vtable};

#[derive(Clone)]
pub struct ArrayLitExpr {
    array: ArrayRef,
}

impl ArrayLitExpr {
    pub fn new(array: ArrayRef) -> Self {
        Self { array }
    }

    pub fn new_expr(array: ArrayRef) -> ExprRef {
        Self::new(array).into_expr()
    }

    pub fn num_rows(&self) -> usize {
        self.array.len()
    }

    pub fn array(&self) -> &ArrayRef {
        &self.array
    }
}

impl Debug for ArrayLitExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ArrayLit(length={})", self.num_rows())
    }
}

impl DisplayAs for ArrayLitExpr {
    fn fmt_as(&self, df: DisplayFormat, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match df {
            DisplayFormat::Compact => write!(f, "array_lit(length={})", self.num_rows()),
            DisplayFormat::Tree => write!(
                f,
                "ArrayLit(length={}, dtype={}, encoding={})",
                self.num_rows(),
                self.array.dtype(),
                self.array.encoding_id()
            ),
        }
    }
}

impl PartialEq<Self> for ArrayLitExpr {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl Eq for ArrayLitExpr {}

impl Hash for ArrayLitExpr {
    fn hash<H: Hasher>(&self, _state: &mut H) {
        todo!()
    }
}

impl AnalysisExpr for ArrayLitExpr {}

pub struct ArrayLitExprEncoding;

vtable!(ArrayLit);

impl VTable for ArrayLitVTable {
    type Expr = ArrayLitExpr;

    type Encoding = ArrayLitExprEncoding;

    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("spql.array")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(ArrayLitExprEncoding.as_ref())
    }

    fn metadata(_expr: &Self::Expr) -> Option<Self::Metadata> {
        None
    }

    fn children(_expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![]
    }

    fn with_children(_expr: &Self::Expr, _children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        vortex_bail!("Can't replace children of a ArrayLitExpr")
    }

    fn build(
        _encoding: &Self::Encoding,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        unimplemented!()
    }

    fn evaluate(expr: &Self::Expr, _scope: &Scope) -> VortexResult<ArrayRef> {
        Ok(expr.array.clone())
    }

    fn return_dtype(expr: &Self::Expr, _scope: &DType) -> VortexResult<DType> {
        Ok(expr.array.dtype().clone())
    }
}
