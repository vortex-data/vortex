// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::expr::BoundExpr;
use crate::expr::transform::match_between::find_between;
use crate::scalar_fn::ReduceCtx;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeRef;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::SimplifyCtx;

impl BoundExpr {
    /// Optimize the root expression node only, iterating to convergence.
    ///
    /// This applies optimization rules repeatedly until no more changes occur:
    /// 1. `simplify_untyped` - type-independent simplifications
    /// 2. `simplify` - type-aware simplifications
    /// 3. `reduce` - abstract reduction rules via `ReduceNode`/`ReduceCtx`
    pub fn optimize(&self) -> VortexResult<BoundExpr> {
        let cache = SimplifyCache;
        Ok(self
            .clone()
            .try_optimize(&cache)?
            .unwrap_or_else(|| self.clone()))
    }

    /// Try to optimize the root expression node only, returning None if no optimizations applied.
    fn try_optimize(&self, cache: &SimplifyCache) -> VortexResult<Option<BoundExpr>> {
        let reduce_ctx = ExpressionReduceCtx;

        let mut current = self.clone();
        let mut any_optimizations = false;
        let mut loop_counter = 0;

        loop {
            if loop_counter > 100 {
                vortex_error::vortex_bail!(
                    "Exceeded maximum optimization iterations (possible infinite loop)"
                );
            }
            loop_counter += 1;

            // Each counted iteration applies all three rule kinds in sequence, each seeing the
            // previous step's output. Leaf nodes have no rules to apply.
            let mut changed = false;

            // Try simplify_untyped
            let simplified = match current.as_call() {
                Some(call) => call.function().simplify_untyped(call)?,
                None => None,
            };
            if let Some(simplified) = simplified {
                current = simplified;
                changed = true;
            }

            // Try simplify (typed)
            let simplified = match current.as_call() {
                Some(call) => call.function().simplify(call, cache)?,
                None => None,
            };
            if let Some(simplified) = simplified {
                current = simplified;
                changed = true;
            }

            // Try reduce via ReduceNode/ReduceCtx
            if let Some(call) = current.as_call() {
                let reduce_node = ExpressionReduceNode {
                    expression: current.clone(),
                };
                if let Some(reduced) = call.function().reduce(&reduce_node, &reduce_ctx)? {
                    let reduced_expr = reduced
                        .as_any()
                        .downcast_ref::<ExpressionReduceNode>()
                        .vortex_expect("ReduceNode not an ExpressionReduceNode")
                        .expression
                        .clone();
                    current = reduced_expr;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
            any_optimizations = true;
        }

        if any_optimizations {
            Ok(Some(current))
        } else {
            Ok(None)
        }
    }

    /// Optimize the entire expression tree recursively.
    ///
    /// Optimizes children first (bottom-up), then optimizes the root.
    pub fn optimize_recursive(&self) -> VortexResult<BoundExpr> {
        Ok(self
            .clone()
            .try_optimize_recursive()?
            .unwrap_or_else(|| self.clone()))
    }

    /// Try to optimize the entire expression tree recursively.
    pub fn try_optimize_recursive(&self) -> VortexResult<Option<BoundExpr>> {
        let cache = SimplifyCache;
        let result = self.try_optimize_recursive_inner(&cache)?;

        // Apply the between optimization once at the top level only.
        // TODO(ngates): remove the "between" optimization, or rewrite it to not always convert
        //  to CNF?
        Ok(Some(find_between(result.unwrap_or_else(|| self.clone()))))
    }

    fn try_optimize_recursive_inner(
        &self,
        cache: &SimplifyCache,
    ) -> VortexResult<Option<BoundExpr>> {
        let mut current = self.clone();
        let mut any_optimizations = false;

        // First optimize the root
        if let Some(optimized) = current.clone().try_optimize(cache)? {
            current = optimized;
            any_optimizations = true;
        }

        // Then recursively optimize children
        let mut new_children = Vec::with_capacity(current.children().len());
        let mut any_child_optimized = false;
        for child in current.children().iter() {
            if let Some(optimized) = child.try_optimize_recursive_inner(cache)? {
                new_children.push(optimized);
                any_child_optimized = true;
            } else {
                new_children.push(child.clone());
            }
        }

        if any_child_optimized {
            current = current.with_children(new_children)?;
            any_optimizations = true;

            // After updating children, try to optimize root again
            if let Some(optimized) = current.clone().try_optimize(cache)? {
                current = optimized;
            }
        }

        if any_optimizations {
            Ok(Some(current))
        } else {
            Ok(None)
        }
    }

    /// Simplify the expression, returning a potentially new expression.
    ///
    /// Deprecated: Use [`BoundExpr::optimize_recursive`] instead, which iterates to convergence.
    #[deprecated(note = "Use BoundExpr::optimize_recursive instead")]
    pub fn simplify(&self) -> VortexResult<BoundExpr> {
        self.optimize_recursive()
    }

    /// Simplify the expression without type information.
    ///
    /// Deprecated: Use [`BoundExpr::optimize_recursive`] instead.
    #[deprecated(note = "Use BoundExpr::optimize_recursive instead")]
    pub fn simplify_untyped(&self) -> VortexResult<BoundExpr> {
        // For backwards compat, do a single bottom-up pass of untyped simplification
        fn inner(expr: &BoundExpr) -> VortexResult<Option<BoundExpr>> {
            let children: Vec<_> = expr.children().iter().map(inner).try_collect()?;

            if children.iter().any(|c| c.is_some()) {
                let new_children: Vec<_> = children
                    .into_iter()
                    .zip(expr.children().iter())
                    .map(|(new_c, old_c)| new_c.unwrap_or_else(|| old_c.clone()))
                    .collect();

                let new_expr = expr.clone().with_children(new_children)?;
                if let Some(call) = new_expr.as_call() {
                    Ok(Some(
                        call.function().simplify_untyped(call)?.unwrap_or(new_expr),
                    ))
                } else {
                    Ok(Some(new_expr))
                }
            } else {
                let Some(call) = expr.as_call() else {
                    return Ok(None);
                };
                call.function().simplify_untyped(call)
            }
        }

        let simplified = if let Some(call) = self.as_call() {
            call.function()
                .simplify_untyped(call)?
                .unwrap_or_else(|| self.clone())
        } else {
            self.clone()
        };

        let simplified = inner(&simplified)?.unwrap_or(simplified);
        let simplified = find_between(simplified);

        Ok(simplified)
    }
}

struct SimplifyCache;

impl SimplifyCtx for SimplifyCache {
    fn return_dtype(&self, expr: &BoundExpr) -> VortexResult<crate::dtype::DType> {
        Ok(expr.dtype().clone())
    }
}

struct ExpressionReduceNode {
    expression: BoundExpr,
}

impl ReduceNode for ExpressionReduceNode {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn node_dtype(&self) -> VortexResult<crate::dtype::DType> {
        Ok(self.expression.dtype().clone())
    }

    fn scalar_fn(&self) -> Option<&ScalarFnRef> {
        self.expression.as_call().map(|call| call.function())
    }

    fn child(&self, idx: usize) -> ReduceNodeRef {
        Arc::new(ExpressionReduceNode {
            expression: self.expression.child(idx).clone(),
        })
    }

    fn child_count(&self) -> usize {
        self.expression.children().len()
    }
}

struct ExpressionReduceCtx;
impl ReduceCtx for ExpressionReduceCtx {
    fn new_node(
        &self,
        scalar_fn: ScalarFnRef,
        children: &[ReduceNodeRef],
    ) -> VortexResult<ReduceNodeRef> {
        let expression = BoundExpr::try_new(
            scalar_fn,
            children
                .iter()
                .map(|c| {
                    c.as_any()
                        .downcast_ref::<ExpressionReduceNode>()
                        .vortex_expect("ReduceNode not an ExpressionReduceNode")
                        .expression
                        .clone()
                })
                .collect::<Vec<_>>(),
        )?;

        Ok(Arc::new(ExpressionReduceNode { expression }))
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::eq;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::or;
    use crate::expr::root;

    #[test]
    fn optimize_or_chain_correctness() -> VortexResult<()> {
        let scope = DType::Struct(
            StructFields::new(
                ["x"].into(),
                vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );
        let expr = or(
            eq(get_item("x", root(scope.clone())), lit(1i32)),
            eq(get_item("x", root(scope)), lit(2i32)),
        );
        let optimized = expr.optimize_recursive()?;

        let s = optimized.to_string();
        assert!(s.contains("$.x"), "expected $.x in {s}");
        assert!(s.contains("1i32") || s.contains('1'), "expected 1 in {s}");
        assert!(s.contains("2i32") || s.contains('2'), "expected 2 in {s}");
        Ok(())
    }
}
