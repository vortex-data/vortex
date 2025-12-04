// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::RefCell;
use std::ops::Deref;

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::expr::Expression;
use crate::expr::Root;
use crate::expr::SimplifyCtx;
use crate::expr::transform::match_between::find_between;

impl Expression {
    /// Simplify the expression, returning a potentially new expression.
    pub fn simplify(&self, scope: &DType) -> VortexResult<Expression> {
        // Recursive inner function to simplify an expression
        fn inner(expr: &Expression, cache: &SimplifyCache) -> VortexResult<Option<Expression>> {
            // Recurse into the expression and simplify from the bottom up.
            let children: Vec<_> = expr
                .children()
                .iter()
                .map(|c| inner(c, cache))
                .try_collect()?;

            if children.iter().any(|c| c.is_some()) {
                // If any child changed, we need to create a new expression node
                let new_children: Vec<_> = children
                    .into_iter()
                    .zip(expr.children().iter())
                    .map(|(new_c, old_c)| new_c.unwrap_or_else(|| old_c.clone()))
                    .collect();

                let new_expr = expr.clone().with_children(new_children)?;

                // Then we simplify the new expression, and since we rewrote the expression we must
                // always return a new expression (even if simplification returns None)
                Ok(Some(
                    new_expr
                        .vtable()
                        .as_dyn()
                        .simplify(&new_expr, cache)?
                        .unwrap_or(new_expr),
                ))
            } else {
                // Otherwise, we attempt to simplify the current expression
                expr.vtable().as_dyn().simplify(expr, cache)
            }
        }

        let cache = SimplifyCache {
            scope,
            dtype_cache: RefCell::new(HashMap::new()),
        };

        let simplified = inner(self, &cache)?.unwrap_or_else(|| self.clone());

        // TODO(ngates): remove the "between" optimization, or rewrite it to not always convert
        //  to CNF?
        let simplified = find_between(simplified);

        // TODO(ngates): perform constant folding by executing expressions with all-literal
        //  children here
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
