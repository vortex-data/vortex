// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_plan::expressions::DynamicFilterPhysicalExpr;

use vortex::arrays::BoolArray;
use vortex::compute::cast;
use vortex::dtype::DType;
use vortex::error::{VortexExpect, VortexResult, vortex_bail};
use vortex::expr::traversal::{Node, Transformed};
use vortex::expr::{
    AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, GetItemVTable, IntoExpr, Scope, VTable, root,
};
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
                let mut initial_vortex_children = dynamic_expr
                    .children()
                    .into_iter()
                    .map(|e| make_vortex_predicate(&[e]).vortex_expect("must work"))
                    .collect::<Vec<_>>();

                if initial_vortex_children.len() != dynamic_expr.children().len() {
                    vortex_bail!("Couldn't convert all expressions")
                }

                if initial_vortex_children.len() == 1 {
                    let arr = initial_vortex_children.pop().vortex_expect("validated");
                    initial_vortex_children = vec![vortex::expr::and(arr.clone(), arr)];
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

impl AnalysisExpr for DFDynamicExpr {
    fn stat_falsification(&self, catalog: &mut dyn vortex::expr::StatsCatalog) -> Option<ExprRef> {
        let current = self.inner_expr().current().ok()?;
        let vx_predicate = make_vortex_predicate(&[&current])?;
        vx_predicate.stat_falsification(catalog)
        // _ = catalog;
        // None
    }
}

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
        // expr.inner_expr()
        //     .children()
        //     .into_iter()
        //     .map(|e| &make_vortex_predicate(&[e]).vortex_expect("must work"))
        //     .collect::<Vec<_>>()
    }

    fn with_children(expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        if children.len() != expr.children().len() {
            panic!("oops")
        }

        DFDynamicExpr::try_new(expr.inner.clone())
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
            let o = BoolArray::from_iter(vec![true; scope.root().len()]).into_array();
            return cast(
                o.as_ref(),
                &o.dtype().with_nullability(scope.dtype().nullability()),
            );
        }

        match make_vortex_predicate(&[&current]) {
            Some(expr) => {
                let expr = expr
                    .transform_up(|node| {
                        if node.is::<GetItemVTable>() {
                            Ok(Transformed::yes(root()))
                        } else {
                            Ok(Transformed::no(node))
                        }
                    })?
                    .into_inner();
                let o = expr.unchecked_evaluate(scope)?;
                cast(
                    o.as_ref(),
                    &o.dtype().with_nullability(scope.dtype().nullability()),
                )
            }
            None => panic!("Oops"),
        }
    }

    fn return_dtype(_expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        Ok(DType::Bool(scope.nullability()))
        // return Ok(scope.clone());
    }
}
