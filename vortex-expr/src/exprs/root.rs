// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;

use vortex_array::stats::Stat;
use vortex_array::{ArrayRef, DeserializeMetadata, EmptyMetadata};
use vortex_dtype::{DType, FieldPath};
use vortex_error::{VortexResult, vortex_bail};

use crate::{
    AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, StatsCatalog, VTable, vtable,
};

vtable!(Root);

/// An expression that returns the full scope of the expression evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RootExpr;

pub struct RootExprEncoding;

impl VTable for RootVTable {
    type Expr = RootExpr;
    type Encoding = RootExprEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("root")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(RootExprEncoding.as_ref())
    }

    fn metadata(_expr: &Self::Expr) -> Option<Self::Metadata> {
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
                "Root expression does not have children, got: {:?}",
                children
            );
        }
        Ok(RootExpr)
    }

    fn evaluate(_expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        Ok(scope.root().clone())
    }

    fn return_dtype(_expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        Ok(scope.clone())
    }
}

impl Display for RootExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "$")
    }
}

impl AnalysisExpr for RootExpr {
    fn max(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        catalog.stats_ref(&self.field_path()?, Stat::Max)
    }

    fn min(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        catalog.stats_ref(&self.field_path()?, Stat::Min)
    }

    fn nan_count(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        catalog.stats_ref(&self.field_path()?, Stat::NaNCount)
    }

    fn field_path(&self) -> Option<FieldPath> {
        Some(FieldPath::root())
    }
}

/// Return a global pointer to the identity token.
/// This is the name of the data found in a vortex array or file.
pub fn root() -> ExprRef {
    RootExpr.into_expr()
}

/// Return whether the expression is a root expression.
pub fn is_root(expr: &ExprRef) -> bool {
    expr.is::<RootVTable>()
}
