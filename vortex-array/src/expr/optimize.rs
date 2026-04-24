// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::cell::RefCell;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::expression::ExprReturnInfo;
use crate::expr::expression::expr_return_dtype_info;
use crate::expr::transform::match_between::find_between;
use crate::scalar_fn::ReduceCtx;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeRef;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::SimplifyCtx;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::root::Root;

impl Expression {
    /// Optimize the root expression node only, iterating to convergence.
    ///
    /// This applies optimization rules repeatedly until no more changes occur:
    /// 1. `simplify_untyped` - type-independent simplifications
    /// 2. `simplify` - type-aware simplifications
    /// 3. `reduce` - abstract reduction rules via `ReduceNode`/`ReduceCtx`
    pub fn optimize(&self, scope: &DType) -> VortexResult<Expression> {
        Ok(self
            .clone()
            .try_optimize(scope)?
            .unwrap_or_else(|| self.clone()))
    }

    /// Try to optimize the root expression node only, returning None if no optimizations applied.
    pub fn try_optimize(&self, scope: &DType) -> VortexResult<Option<Expression>> {
        let cache = SimplifyCache {
            scope,
            dtype_cache: RefCell::new(HashMap::new()),
        };
        let reduce_ctx = ExpressionReduceCtx {
            scope: scope.clone(),
        };

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

            let mut changed = false;

            // Try simplify_untyped
            if let Some(simplified) = current.scalar_fn().simplify_untyped(&current)? {
                current = simplified;
                changed = true;
                any_optimizations = true;
            }

            // Try simplify (typed)
            if let Some(simplified) = current.scalar_fn().simplify(&current, &cache)? {
                current = simplified;
                changed = true;
                any_optimizations = true;
            }

            // Try reduce via ReduceNode/ReduceCtx
            let reduce_node = ExpressionReduceNode {
                expression: current.clone(),
                scope: scope.clone(),
            };
            if let Some(reduced) = current.scalar_fn().reduce(&reduce_node, &reduce_ctx)? {
                let reduced_expr = reduced
                    .as_any()
                    .downcast_ref::<ExpressionReduceNode>()
                    .vortex_expect("ReduceNode not an ExpressionReduceNode")
                    .expression
                    .clone();
                current = reduced_expr;
                changed = true;
                any_optimizations = true;
            }

            if !changed {
                break;
            }
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
    pub fn optimize_recursive(&self, scope: &DType) -> VortexResult<Expression> {
        Ok(self
            .clone()
            .try_optimize_recursive(scope)?
            .unwrap_or_else(|| self.clone()))
    }

    /// Try to optimize the entire expression tree recursively.
    pub fn try_optimize_recursive(&self, scope: &DType) -> VortexResult<Option<Expression>> {
        let mut current = self.clone();
        let mut any_optimizations = false;

        // First optimize the root
        if let Some(optimized) = current.clone().try_optimize(scope)? {
            current = optimized;
            any_optimizations = true;
        }

        // Then recursively optimize children
        let mut new_children = Vec::with_capacity(current.children().len());
        let mut any_child_optimized = false;
        for child in current.children().iter() {
            if let Some(optimized) = child.try_optimize_recursive(scope)? {
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
            if let Some(optimized) = current.clone().try_optimize(scope)? {
                current = optimized;
            }
        }

        // TODO(ngates): remove the "between" optimization, or rewrite it to not always convert
        //  to CNF?
        let current = find_between(current);

        if any_optimizations {
            Ok(Some(current))
        } else {
            Ok(None)
        }
    }

    /// Simplify the expression, returning a potentially new expression.
    ///
    /// Deprecated: Use [`Expression::optimize_recursive`] instead, which iterates to convergence.
    #[deprecated(note = "Use Expression::optimize_recursive instead")]
    pub fn simplify(&self, scope: &DType) -> VortexResult<Expression> {
        self.optimize_recursive(scope)
    }

    /// Simplify the expression without type information.
    ///
    /// Deprecated: Use [`Expression::optimize_recursive`] instead.
    #[deprecated(note = "Use Expression::optimize_recursive instead")]
    pub fn simplify_untyped(&self) -> VortexResult<Expression> {
        // For backwards compat, do a single bottom-up pass of untyped simplification
        fn inner(expr: &Expression) -> VortexResult<Option<Expression>> {
            let children: Vec<_> = expr.children().iter().map(inner).try_collect()?;

            if children.iter().any(|c| c.is_some()) {
                let new_children: Vec<_> = children
                    .into_iter()
                    .zip(expr.children().iter())
                    .map(|(new_c, old_c)| new_c.unwrap_or_else(|| old_c.clone()))
                    .collect();

                let new_expr = expr.clone().with_children(new_children)?;
                Ok(Some(
                    new_expr
                        .scalar_fn()
                        .simplify_untyped(&new_expr)?
                        .unwrap_or(new_expr),
                ))
            } else {
                expr.scalar_fn().simplify_untyped(expr)
            }
        }

        let simplified = self
            .scalar_fn()
            .simplify_untyped(self)?
            .unwrap_or_else(|| self.clone());

        let simplified = inner(&simplified)?.unwrap_or(simplified);
        let simplified = find_between(simplified);

        Ok(simplified)
    }
}

struct SimplifyCache<'a> {
    scope: &'a DType,
    dtype_cache: RefCell<HashMap<Expression, ExprReturnInfo>>,
}

impl SimplifyCache<'_> {
    fn return_dtype_info(&self, expr: &Expression) -> VortexResult<ExprReturnInfo> {
        if expr.is::<Root>() {
            return Ok(ExprReturnInfo::storage_chain_repr_wrapper(
                self.scope.clone(),
            ));
        }

        if let Some(literal) = expr.as_opt::<Literal>() {
            return Ok(ExprReturnInfo::storage_chain_repr_wrapper(
                literal.dtype().clone(),
            ));
        }

        if let Some(dtype) = self.dtype_cache.borrow().get(expr) {
            return Ok(dtype.clone());
        }

        // Otherwise, compute dtype from children
        let mut input_dtypes: Vec<_> = expr
            .children()
            .iter()
            .map(|c| self.return_dtype_info(c))
            .try_collect()?;
        let dtype = expr_return_dtype_info(expr.scalar_fn(), &mut input_dtypes)?;
        self.dtype_cache
            .borrow_mut()
            .insert(expr.clone(), dtype.clone());

        Ok(dtype)
    }
}

impl SimplifyCtx for SimplifyCache<'_> {
    fn return_dtype(&self, expr: &Expression) -> VortexResult<DType> {
        Ok(self.return_dtype_info(expr)?.dtype)
    }
}

struct ExpressionReduceNode {
    expression: Expression,
    scope: DType,
}

impl ReduceNode for ExpressionReduceNode {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn node_dtype(&self) -> VortexResult<DType> {
        self.expression.return_dtype(&self.scope)
    }

    fn scalar_fn(&self) -> Option<&ScalarFnRef> {
        Some(self.expression.scalar_fn())
    }

    fn child(&self, idx: usize) -> ReduceNodeRef {
        Arc::new(ExpressionReduceNode {
            expression: self.expression.child(idx).clone(),
            scope: self.scope.clone(),
        })
    }

    fn child_count(&self) -> usize {
        self.expression.children().len()
    }
}

struct ExpressionReduceCtx {
    scope: DType,
}
impl ReduceCtx for ExpressionReduceCtx {
    fn new_node(
        &self,
        scalar_fn: ScalarFnRef,
        children: &[ReduceNodeRef],
    ) -> VortexResult<ReduceNodeRef> {
        let expression = Expression::try_new(
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

        Ok(Arc::new(ExpressionReduceNode {
            expression,
            scope: self.scope.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::extension::ExtDType;
    use crate::expr::checked_add;
    use crate::expr::fill_null;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::extension::tests::divisible_int::DivisibleInt;
    use crate::extension::tests::divisible_int::Divisor;

    fn divisible_int_dtype(divisor: u64) -> DType {
        DType::Extension(
            ExtDType::<DivisibleInt>::try_new(
                Divisor(divisor),
                DType::Primitive(PType::U64, Nullability::NonNullable),
            )
            .unwrap()
            .erased(),
        )
    }

    #[test]
    fn optimize_recursive_uses_refinement_fallback_for_typed_simplify() {
        let scope = divisible_int_dtype(3);
        let add = checked_add(root(), root());
        let expr = fill_null(add.clone(), lit(0u64));

        let optimized = expr.optimize_recursive(&scope).unwrap();

        assert_eq!(optimized, add);
        assert_eq!(
            optimized.return_dtype(&scope).unwrap(),
            DType::Primitive(PType::U64, Nullability::NonNullable),
        );
    }
}
