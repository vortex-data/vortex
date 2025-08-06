// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use datafusion::physical_plan::{PhysicalExpr, expressions::DynamicFilterPhysicalExpr};
use vortex::ArrayRef;
use vortex::DeserializeMetadata;
use vortex::EmptyMetadata;
use vortex::dtype::DType;
use vortex::expr::IntoExpr;

use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::expr::AnalysisExpr;
use vortex::expr::ExprEncodingRef;
use vortex::expr::ExprId;
use vortex::expr::ExprRef;
use vortex::expr::Scope;
use vortex::expr::VTable;

use crate::make_vortex_predicate;

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Debug, Hash, Clone)]
pub struct DFDynamicExpr {
    inner: Arc<dyn PhysicalExpr>,
}

vortex::expr::vtable!(DFDynamic);

impl DFDynamicExpr {
    pub fn new(df_expr: Arc<dyn PhysicalExpr>) -> Self {
        Self { inner: df_expr }
    }

    pub fn new_expr(df_expr: Arc<dyn PhysicalExpr>) -> ExprRef {
        Self::new(df_expr).into_expr()
    }

    fn inner_expr(&self) -> &DynamicFilterPhysicalExpr {
        self.inner
            .as_any()
            .downcast_ref::<DynamicFilterPhysicalExpr>()
            .vortex_expect("Verified on creation")
    }
}

impl PartialEq for DFDynamicExpr {
    fn eq(&self, other: &Self) -> bool {
        self.inner.dyn_eq(other)
    }
}

impl std::fmt::Display for DFDynamicExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DynamicDFExpr({})", self.inner)
    }
}

impl Eq for DFDynamicExpr {}

impl AnalysisExpr for DFDynamicExpr {}

pub struct DFDynamicExprEncoding;

impl VTable for DFDynamicVTable {
    type Expr = DFDynamicExpr;

    type Encoding = DFDynamicExprEncoding;

    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("datafusion.vortex.dynamic")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(DFDynamicExprEncoding.as_ref())
    }

    fn metadata(_expr: &Self::Expr) -> Option<Self::Metadata> {
        None
    }

    fn children(_expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![]
    }

    fn with_children(expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        if !children.is_empty() {
            vortex_bail!("DFDynamicExpr ")
        }

        Ok(expr.clone())
    }

    fn build(
        _encoding: &Self::Encoding,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        unreachable!()
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        let inner = expr.inner_expr();
        let current = inner.current()?;

        match make_vortex_predicate(&[&current]) {
            Some(expr) => expr.evaluate(scope),
            None => Ok(scope.root().clone()),
        }
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        let inner = expr.inner_expr();
        let current = inner.current()?;

        match make_vortex_predicate(&[&current]) {
            Some(expr) => expr.return_dtype(scope),
            None => Ok(scope.clone()),
        }
    }
}
