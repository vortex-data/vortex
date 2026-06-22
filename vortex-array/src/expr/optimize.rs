// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::RefCell;
use std::ops::Deref;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::transform::match_between::find_between;
use crate::scalar_fn::ReduceCtx;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeRef;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::SimplifyCtx;
use crate::scalar_fn::fns::root::Root;

impl Expression {
    /// Optimize the root expression node only, iterating to convergence.
    ///
    /// This applies optimization rules repeatedly until no more changes occur:
    /// 1. `simplify_untyped` - type-independent simplifications
    /// 2. `simplify` - type-aware simplifications
    /// 3. `reduce` - abstract reduction rules via `ReduceNode`/`ReduceCtx`
    pub fn optimize(&self, scope: &DType) -> VortexResult<Expression> {
        let cache = SimplifyCache {
            scope,
            dtype_cache: RefCell::new(HashMap::new()),
        };
        Ok(self
            .clone()
            .try_optimize(scope, &cache)?
            .unwrap_or_else(|| self.clone()))
    }

    /// Try to optimize the root expression node only, returning None if no optimizations applied.
    fn try_optimize(
        &self,
        scope: &DType,
        cache: &SimplifyCache<'_>,
    ) -> VortexResult<Option<Expression>> {
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
            if let Some(simplified) = current.scalar_fn().simplify(&current, cache)? {
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
                let reduced_expr = reduced.as_expression().expression;
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
        let cache = SimplifyCache {
            scope,
            dtype_cache: RefCell::new(HashMap::new()),
        };
        let result = self.try_optimize_recursive_inner(scope, &cache)?;

        // Apply the between optimization once at the top level only.
        // TODO(ngates): remove the "between" optimization, or rewrite it to not always convert
        //  to CNF?
        Ok(Some(find_between(result.unwrap_or_else(|| self.clone()))))
    }

    fn try_optimize_recursive_inner(
        &self,
        scope: &DType,
        cache: &SimplifyCache<'_>,
    ) -> VortexResult<Option<Expression>> {
        let mut current = self.clone();
        let mut any_optimizations = false;

        // First optimize the root
        if let Some(optimized) = current.clone().try_optimize(scope, cache)? {
            current = optimized;
            any_optimizations = true;
        }

        // Then recursively optimize children
        let mut new_children = Vec::with_capacity(current.children().len());
        let mut any_child_optimized = false;
        for child in current.children().iter() {
            if let Some(optimized) = child.try_optimize_recursive_inner(scope, cache)? {
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
            if let Some(optimized) = current.clone().try_optimize(scope, cache)? {
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
    dtype_cache: RefCell<HashMap<Expression, DType>>,
}

impl SimplifyCtx for SimplifyCache<'_> {
    fn return_dtype(&self, expr: &Expression) -> VortexResult<DType> {
        // If the expression is "root", return the scope dtype
        if expr.is::<Root>() {
            return Ok(self.scope.clone());
        }

        if let Some(dtype) = self.dtype_cache.borrow().get(expr) {
            return Ok(dtype.clone());
        }

        // Otherwise, compute dtype from children
        let input_dtypes: Vec<_> = expr
            .children()
            .iter()
            .map(|c| self.return_dtype(c))
            .try_collect()?;
        let dtype = expr.deref().return_dtype(&input_dtypes)?;
        self.dtype_cache
            .borrow_mut()
            .insert(expr.clone(), dtype.clone());

        Ok(dtype)
    }
}

#[derive(Clone)]
pub struct ExpressionReduceNode {
    expression: Expression,
    scope: DType,
}

impl ReduceNode for ExpressionReduceNode {
    fn as_array(&self) -> ArrayRef {
        vortex_panic!("Cannot produce ArrayRef out of Expression node")
    }

    fn as_expression(&self) -> ExpressionReduceNode {
        self.clone()
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
                .map(|c| c.as_expression().expression)
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
        let expr = or(
            eq(get_item("x", root()), lit(1i32)),
            eq(get_item("x", root()), lit(2i32)),
        );
        let scope = DType::Struct(
            StructFields::new(
                ["x"].into(),
                vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );
        let optimized = expr.optimize_recursive(&scope)?;

        let s = optimized.to_string();
        assert!(s.contains("$.x"), "expected $.x in {s}");
        assert!(s.contains("1i32") || s.contains('1'), "expected 1 in {s}");
        assert!(s.contains("2i32") || s.contains('2'), "expected 2 in {s}");
        Ok(())
    }
}
