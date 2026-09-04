// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::expr::BoundExpression;
use crate::expr::Expression;
use crate::expr::Scope;
use crate::expr::VariableRef;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::fns::literal::Literal;

/// Dynamic lexical state shared while an expression is applied to an array.
pub(crate) struct ApplyCtx {
    bindings: HashMap<VariableRef, ArrayRef>,
}

impl ApplyCtx {
    fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    fn binding(&self, variable_ref: VariableRef) -> Option<&ArrayRef> {
        self.bindings.get(&variable_ref)
    }
}

impl ArrayRef {
    /// Apply a bound expression to this array.
    pub fn apply_bound(self, expr: &BoundExpression) -> VortexResult<ArrayRef> {
        apply(self, expr, &mut ApplyCtx::new())?.optimize()
    }

    /// Apply the expression to this array, producing a new array in constant time.
    pub fn apply(self, expr: &Expression) -> VortexResult<ArrayRef> {
        let scope = Scope::new(self.dtype().clone());
        let bound = expr.bind(&scope)?;
        self.apply_bound(&bound)
    }
}

/// Lower `expr` into an array in `root`'s row domain using the current lexical bindings.
pub(crate) fn apply(
    root: ArrayRef,
    expr: &BoundExpression,
    apply_ctx: &mut ApplyCtx,
) -> VortexResult<ArrayRef> {
    let (scalar_fn, children) = match expr {
        BoundExpression::Root { dtype } => {
            vortex_ensure!(
                root.dtype() == dtype,
                "expression root dtype {dtype} does not match input array dtype {}",
                root.dtype()
            );
            return Ok(root);
        }
        BoundExpression::Variable(variable) => {
            let array = apply_ctx
                .binding(variable.variable_ref())
                .cloned()
                .ok_or_else(|| vortex_err!("cannot apply unbound variable '{variable}'"))?;
            vortex_ensure!(
                array.dtype() == variable.dtype(),
                "binding for variable '{variable}' has dtype {}, expected {}",
                array.dtype(),
                variable.dtype(),
            );
            vortex_ensure!(
                array.len() == root.len(),
                "binding for variable '{variable}' has length {}, expected {}",
                array.len(),
                root.len()
            );
            return Ok(array);
        }
        BoundExpression::Lambda(_) => {
            return Err(vortex_err!(
                "a lambda can be applied only by the binder that established its scope"
            ));
        }
        BoundExpression::Scalar {
            scalar_fn,
            children,
            ..
        } => (scalar_fn, children),
    };

    if let Some(scalar) = scalar_fn.as_opt::<Literal>() {
        return Ok(ConstantArray::new(scalar.clone(), root.len()).into_array());
    }

    let children = children
        .iter()
        .map(|child| apply(root.clone(), child, apply_ctx))
        .try_collect()?;

    Ok(ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, root.len())?.into_array())
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::root;

    #[test]
    fn bound_application_checks_root_dtype() -> VortexResult<()> {
        let expected = DType::Primitive(PType::I32, Nullability::NonNullable);
        let bound = root().bind(&expected)?;
        let root = buffer![0_i64, 0].into_array();

        assert!(root.apply_bound(&bound).is_err());
        Ok(())
    }
}
