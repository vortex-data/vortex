// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_plan::expressions::DynamicFilterPhysicalExpr;

use vortex::arrays::BoolArray;
use vortex::dtype::{DType, Nullability};
use vortex::error::{VortexExpect, VortexResult, vortex_bail};
use vortex::expr::{AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, VTable};
use vortex::{ArrayRef, DeserializeMetadata, EmptyMetadata, IntoArray};

use crate::make_vortex_predicate;

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Debug, Clone)]
pub struct DFDynamicExpr {
    inner: Arc<dyn PhysicalExpr>,
    initial_vortex_children: Vec<ExprRef>,
}

impl std::hash::Hash for DFDynamicExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.initial_vortex_children.hash(state);
    }
}

impl PartialEq for DFDynamicExpr {
    fn eq(&self, other: &Self) -> bool {
        self.initial_vortex_children == other.initial_vortex_children
    }
}

impl Eq for DFDynamicExpr {}

vortex::expr::vtable!(DFDynamic);

impl DFDynamicExpr {
    pub fn try_new(df_expr: Arc<dyn PhysicalExpr>) -> VortexResult<Self> {
        match df_expr.as_any().downcast_ref::<DynamicFilterPhysicalExpr>() {
            Some(dynamic_expr) => {
                let initial_vortex_children = dynamic_expr
                    .children()
                    .into_iter()
                    .filter_map(|e| make_vortex_predicate(&[e]))
                    .collect::<Vec<_>>();
                if initial_vortex_children.len() != dynamic_expr.children().len() {
                    vortex_bail!("Couldn't convert all expressions")
                }
                Ok(Self {
                    inner: df_expr,
                    initial_vortex_children,
                })
            }
            None => vortex_bail!(""),
        }
    }

    pub fn try_new_expr(df_expr: Arc<dyn PhysicalExpr>) -> VortexResult<ExprRef> {
        Ok(Self::try_new(df_expr)?.into_expr())
    }

    fn inner_expr(&self) -> &DynamicFilterPhysicalExpr {
        self.inner
            .as_any()
            .downcast_ref::<DynamicFilterPhysicalExpr>()
            .vortex_expect("Verified on creation")
    }
}

impl std::fmt::Display for DFDynamicExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DynamicDFExpr({})", self.inner)
    }
}

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

    fn children(expr: &Self::Expr) -> Vec<&ExprRef> {
        expr.initial_vortex_children.iter().collect()
    }

    fn with_children(expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        if children.len() != expr.children().len() {
            panic!("oops")
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

        if current == datafusion::physical_expr::expressions::lit(true) {
            return Ok(BoolArray::from_iter(vec![true; scope.root().len()]).into_array());
        }

        match make_vortex_predicate(&[&current]) {
            Some(expr) => expr.evaluate(scope),
            None => Ok(scope.root().clone()),
        }
    }

    fn return_dtype(_expr: &Self::Expr, _scope: &DType) -> VortexResult<DType> {
        Ok(DType::Bool(Nullability::NonNullable))
        // return Ok(scope.clone());
    }
}
