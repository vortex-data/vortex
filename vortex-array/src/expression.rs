// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ListTransformArray;
use crate::arrays::ScalarFnArray;
use crate::expr::BoundExpression;
use crate::expr::Expression;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::fns::literal::Literal;

impl ArrayRef {
    /// Apply a bound expression to this array, producing a new array in constant time.
    pub fn apply_bound(self, expr: &BoundExpression) -> VortexResult<ArrayRef> {
        match expr {
            BoundExpression::Root { .. } => Ok(self),
            BoundExpression::Lambda(_) => {
                vortex_bail!("cannot apply a lambda outside a higher-order function")
            }
            BoundExpression::ListTransform {
                lambda, children, ..
            } => {
                let list = self.clone().apply_bound(&children[0])?;
                let captures = children[1..]
                    .iter()
                    .map(|capture| self.clone().apply_bound(capture))
                    .collect::<VortexResult<Vec<_>>>()?;
                Ok(ListTransformArray::try_new(list, lambda.clone(), captures)?.into_array())
            }
            BoundExpression::Variable(variable) => {
                vortex_bail!("cannot apply variable '{variable}' without a provided value")
            }
            BoundExpression::Scalar {
                scalar_fn,
                children,
                ..
            } => apply_bound_scalar_fn(self, scalar_fn, children),
        }
    }

    /// Apply the expression to this array, producing a new array in constant time.
    pub fn apply(self, expr: &Expression) -> VortexResult<ArrayRef> {
        match expr {
            Expression::Root => Ok(self),
            Expression::Lambda(_) => {
                vortex_bail!("cannot apply a lambda outside a higher-order function")
            }
            Expression::ListTransform { .. } => {
                let bound = expr.bind(self.dtype())?;
                self.apply_bound(&bound)
            }
            Expression::Variable(variable) => {
                vortex_bail!("cannot apply unbound variable '{variable}'")
            }
            Expression::Scalar {
                scalar_fn,
                children,
            } => apply_scalar_fn(self, scalar_fn, children),
        }
    }
}

fn apply_bound_scalar_fn(
    root: ArrayRef,
    scalar_fn: &ScalarFnRef,
    children: &[BoundExpression],
) -> VortexResult<ArrayRef> {
    if let Some(scalar) = scalar_fn.as_opt::<Literal>() {
        return Ok(ConstantArray::new(scalar.clone(), root.len()).into_array());
    }

    let children: Vec<_> = children
        .iter()
        .map(|child| root.clone().apply_bound(child))
        .try_collect()?;
    let array =
        ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, root.len())?.into_array();
    array.optimize()
}

fn apply_scalar_fn(
    root: ArrayRef,
    scalar_fn: &ScalarFnRef,
    children: &[Expression],
) -> VortexResult<ArrayRef> {
    if let Some(scalar) = scalar_fn.as_opt::<Literal>() {
        return Ok(ConstantArray::new(scalar.clone(), root.len()).into_array());
    }

    let children: Vec<_> = children
        .iter()
        .map(|child| root.clone().apply(child))
        .try_collect()?;
    let array =
        ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, root.len())?.into_array();
    array.optimize()
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::expr::Scope;
    use crate::expr::Variable;
    use crate::expr::lambda;
    use crate::expr::var;

    #[test]
    fn variable_application_requires_a_runtime_binding() -> VortexResult<()> {
        let root = buffer![1_i32, 2, 3].into_array();
        let expression = var("value");
        assert!(root.clone().apply(&expression).is_err());

        let scope = Scope::new(root.dtype().clone())
            .with_bindings([(Variable::new("value"), root.dtype().clone())])?;
        let bound = expression.bind_scope(&scope)?;
        assert!(root.apply_bound(&bound).is_err());
        Ok(())
    }

    #[test]
    fn lambda_application_requires_a_higher_order_function() -> VortexResult<()> {
        let root = buffer![1_i32, 2, 3].into_array();
        let expression = lambda(["value"], var("value"))?;

        assert!(root.apply(&expression).is_err());
        Ok(())
    }
}
