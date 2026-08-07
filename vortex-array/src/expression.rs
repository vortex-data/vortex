// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::expr::BoundExpression;
use crate::expr::Expression;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::fns::literal::Literal;

impl ArrayRef {
    /// Apply a bound expression to this array, producing a new array in constant time.
    pub fn apply_bound(self, expr: &BoundExpression) -> VortexResult<ArrayRef> {
        let (scalar_fn, children) = match expr {
            // Root evaluates to the scope, which is this array.
            BoundExpression::Root { .. } => return Ok(self),
            // Execution has no variable environment, so a variable cannot be evaluated. It must be
            // substituted by whatever bound it before the tree reaches the array layer.
            BoundExpression::Variable { variable, .. } => vortex_bail!(
                "cannot evaluate variable '{variable}': execution has no variable environment"
            ),
            BoundExpression::Lambda(lambda) => {
                vortex_bail!("cannot evaluate a lambda ({lambda:?}); it must be applied first")
            }
            BoundExpression::Scalar {
                scalar_fn,
                children,
                ..
            } => (scalar_fn, children),
        };

        if let Some(scalar) = scalar_fn.as_opt::<Literal>() {
            return Ok(ConstantArray::new(scalar.clone(), self.len()).into_array());
        }

        let children: Vec<_> = children
            .iter()
            .map(|child| self.clone().apply_bound(child))
            .try_collect()?;

        let array =
            ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, self.len())?.into_array();

        array.optimize()
    }

    /// Apply the expression to this array, producing a new array in constant time.
    pub fn apply(self, expr: &Expression) -> VortexResult<ArrayRef> {
        let scalar_fn = match expr {
            // Root evaluates to the scope, which is this array.
            Expression::Root => return Ok(self),
            Expression::Variable(variable) => vortex_bail!(
                "cannot evaluate variable '{variable}': execution has no variable environment"
            ),
            Expression::Lambda(lambda) => {
                vortex_bail!("cannot evaluate a lambda ({lambda}); it must be applied first")
            }
            Expression::Scalar { scalar_fn, .. } => scalar_fn,
        };

        // Manually convert literals to ConstantArray.
        if let Some(scalar) = expr.as_opt::<Literal>() {
            return Ok(ConstantArray::new(scalar.clone(), self.len()).into_array());
        }

        // Otherwise, collect the child arrays.
        let children: Vec<_> = expr
            .children()
            .iter()
            .map(|e| self.clone().apply(e))
            .try_collect()?;
        let array =
            ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, self.len())?.into_array();

        // Optimize the resulting array's root.
        array.optimize()
    }
}
