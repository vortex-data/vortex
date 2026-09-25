// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::expr::BoundExpression;
use crate::expr::BoundExpressionReduceNode;
use crate::expr::transform::match_between::find_between_bound;
use crate::scalar_fn::SimplifyCtx;

struct BoundSimplifyCtx;

impl SimplifyCtx for BoundSimplifyCtx {
    fn return_dtype(&self, expr: &BoundExpression) -> VortexResult<DType> {
        Ok(expr.dtype().clone())
    }
}

impl BoundExpression {
    /// Simplify a typed tree after its children and function overloads have been chosen.
    pub fn optimize_recursive(&self) -> VortexResult<Self> {
        let optimized = self.optimize_recursive_inner()?;
        Ok(find_between_bound(optimized))
    }

    fn optimize_recursive_inner(&self) -> VortexResult<Self> {
        let mut current = self.optimize_node()?;
        if !current.children().is_empty() {
            let children = current
                .children()
                .iter()
                .map(Self::optimize_recursive_inner)
                .collect::<VortexResult<Vec<_>>>()?;
            current = current.with_children(children)?.optimize_node()?;
        }
        Ok(current)
    }

    fn optimize_node(&self) -> VortexResult<Self> {
        let mut current = self.clone();
        for _ in 0..100 {
            let Some(scalar_fn) = current.as_scalar() else {
                return Ok(current);
            };
            let next = scalar_fn.simplify(&current, &BoundSimplifyCtx)?;
            let next = if next.is_some() {
                next
            } else {
                let node = BoundExpressionReduceNode::new(&current);
                scalar_fn
                    .reduce_bound_expression(&node)?
                    .map(BoundExpressionReduceNode::into_expression)
            };
            match next {
                Some(next) if next != current => current = next,
                _ => return Ok(current),
            }
        }
        vortex_error::vortex_bail!("Exceeded maximum bound-expression optimization iterations")
    }
}
