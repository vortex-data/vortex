// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::SmallVec;
use vortex_error::VortexExpect;
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
        let BoundExpression::Scalar {
            scalar_fn,
            children,
            ..
        } = expr
        else {
            if let Some(variable) = expr.as_variable() {
                vortex_bail!("cannot apply variable '{variable}' without a provided value");
            }
            return Ok(self);
        };

        if let Some(scalar) = scalar_fn.as_opt::<Literal>() {
            return Ok(ConstantArray::new(scalar.clone(), self.len()).into_array());
        }

        let len = self.len();
        let children = children
            .iter()
            .map(|child| self.clone().apply_bound(child))
            .collect::<VortexResult<SmallVec<[ArrayRef; 4]>>>()?;

        let array = ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, len)?.into_array();

        array.optimize()
    }

    /// Apply the expression to this array, producing a new array in constant time.
    pub fn apply(self, expr: &Expression) -> VortexResult<ArrayRef> {
        // If the expression is a root, return self.
        if expr.is_root() {
            return Ok(self);
        }

        if let Some(variable) = expr.as_variable() {
            vortex_bail!("cannot apply unbound variable '{variable}'");
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
            .map(|e| self.clone().apply(e))
            .collect::<VortexResult<SmallVec<[ArrayRef; 4]>>>()?;

        // And wrap the scalar function up in an array.
        let scalar_fn = expr.as_scalar().vortex_expect(
            "root, variable and literal were handled above, so this is a scalar node",
        );
        let array = ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, len)?.into_array();

        // Optimize the resulting array's root.
        array.optimize()
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::expr::Scope;
    use crate::expr::Variable;
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
}
