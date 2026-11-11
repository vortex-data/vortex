// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::SmallVec;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::expr::BoundExpression;
use crate::expr::Expression;
use crate::expr::expression::unbound_variable;
use crate::scalar_fn::fns::literal::Literal;

impl ArrayRef {
    /// Apply a bound expression to this array, producing a new array in constant time.
    pub fn apply_bound(self, expr: &BoundExpression) -> VortexResult<ArrayRef> {
        // See `apply`: exercise the optimizer's dtype-preservation assertions in debug builds.
        if cfg!(debug_assertions) {
            drop(expr.optimize_recursive());
        }
        self.apply_bound_tree(expr)
    }

    fn apply_bound_tree(self, expr: &BoundExpression) -> VortexResult<ArrayRef> {
        let BoundExpression::Scalar {
            scalar_fn,
            children,
            ..
        } = expr
        else {
            return Ok(self);
        };

        if let Some(scalar) = scalar_fn.as_opt::<Literal>() {
            return Ok(ConstantArray::new(scalar.clone(), self.len()).into_array());
        }

        let len = self.len();
        let children = children
            .iter()
            .map(|child| self.clone().apply_bound_tree(child))
            .collect::<VortexResult<SmallVec<[ArrayRef; 4]>>>()?;

        Ok(ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, len)?.into_array())
    }

    /// Apply the expression to this array, producing a new array in constant time.
    pub fn apply(self, expr: &Expression) -> VortexResult<ArrayRef> {
        // Exercise the optimizer on every applied expression in debug builds, so its
        // dtype-preservation assertions run across the whole test suite. Binding fails for
        // expressions that are invalid for this array; `apply` reports those lazily, so they are
        // skipped here, as are optimizer errors, which execution of the bound form would surface.
        if cfg!(debug_assertions)
            && let Ok(bound) = expr.bind(self.dtype())
        {
            drop(bound.optimize_recursive());
        }
        self.apply_tree(expr)
    }

    fn apply_tree(self, expr: &Expression) -> VortexResult<ArrayRef> {
        // If the expression is a root, return self.
        if expr.is_root() {
            return Ok(self);
        }
        if let Expression::Variable(name) = expr {
            return Err(unbound_variable(name));
        }

        // Manually convert literals to ConstantArray.
        if let Some(scalar) = expr.as_opt::<Literal>() {
            return Ok(ConstantArray::new(scalar.clone(), self.len()).into_array());
        }

        // Otherwise, collect the child arrays.
        let len = self.len();
        let children = expr
            .children()
            .iter()
            .map(|e| self.clone().apply_tree(e))
            .collect::<VortexResult<SmallVec<[ArrayRef; 4]>>>()?;

        // And wrap the scalar function up in an array.
        let scalar_fn = expr
            .as_scalar()
            .vortex_expect("leaves and literals were handled above, so this is a scalar node");
        Ok(ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, len)?.into_array())
    }
}
