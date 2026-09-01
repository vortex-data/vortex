// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::dtype::FieldName;
use crate::dtype::Nullability;
use crate::expr::BoundExpression;
use crate::expr::BoundLambda;
use crate::expr::Expression;
use crate::expr::VariableRef;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::pack::Pack;
use crate::scalar_fn::fns::pack::PackOptions;

impl ArrayRef {
    /// Apply a bound expression to this array, producing a new array in constant time.
    pub fn apply_bound(self, expr: &BoundExpression) -> VortexResult<ArrayRef> {
        BoundApplyCtx {
            root: &self,
            bindings: None,
        }
        .apply(expr)
    }

    /// Apply the expression to this array, producing a new array in constant time.
    pub fn apply(self, expr: &Expression) -> VortexResult<ArrayRef> {
        match expr {
            Expression::Root => Ok(self),
            Expression::Lambda(_) => {
                vortex_bail!("cannot apply a lambda outside a higher-order function")
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

impl BoundLambda {
    /// Apply this lambda to arrays in a common invocation row domain.
    ///
    /// Parameters and captures are packed into a lazy non-nullable struct. Variable projections
    /// reduce through that pack to the original arrays, leaving only the lambda body's lazy scalar
    /// function array tree.
    pub fn apply(
        &self,
        root: ArrayRef,
        parameters: &[ArrayRef],
        captures: &[ArrayRef],
    ) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            parameters.len() == self.param_dtypes().len(),
            "lambda takes {} parameters but was applied with {} arguments",
            self.param_dtypes().len(),
            parameters.len()
        );
        vortex_ensure!(
            captures.len() == self.captures().len(),
            "lambda requires {} captures but was applied with {}",
            self.captures().len(),
            captures.len()
        );
        vortex_ensure!(
            self.body().is_root_bound_to(root.dtype()),
            "lambda root expects a different dtype than {}",
            root.dtype()
        );

        for (index, (expected_dtype, parameter)) in
            self.param_dtypes().iter().zip(parameters).enumerate()
        {
            vortex_ensure!(
                parameter.dtype() == expected_dtype,
                "lambda parameter {index} expects dtype {expected_dtype}, got {}",
                parameter.dtype()
            );
            vortex_ensure!(
                parameter.len() == root.len(),
                "lambda parameter {index} has length {}, expected {}",
                parameter.len(),
                root.len()
            );
        }
        for (index, (capture, array)) in self.captures().iter().zip(captures).enumerate() {
            vortex_ensure!(
                array.dtype() == capture.dtype(),
                "lambda capture {index} expects dtype {}, got {}",
                capture.dtype(),
                array.dtype()
            );
            vortex_ensure!(
                array.len() == root.len(),
                "lambda capture {index} has length {}, expected {}",
                array.len(),
                root.len()
            );
        }

        let names = self
            .param_refs()
            .iter()
            .copied()
            .chain(self.captures().iter().map(|capture| capture.variable_ref()))
            .map(binding_name)
            .collect::<Vec<_>>()
            .into();
        let fields = parameters.iter().chain(captures).cloned().collect();
        let bindings = ScalarFnArray::try_new_with_len(
            Pack.bind(PackOptions {
                names,
                nullability: Nullability::NonNullable,
            }),
            fields,
            root.len(),
        )?
        .into_array();

        let result = BoundApplyCtx {
            root: &root,
            bindings: Some(&bindings),
        }
        .apply(self.body())?;
        vortex_ensure!(
            result.dtype() == self.body_dtype(),
            "lambda produced dtype {}, expected {}",
            result.dtype(),
            self.body_dtype()
        );
        vortex_ensure!(
            result.len() == root.len(),
            "lambda produced {} rows, expected {}",
            result.len(),
            root.len()
        );
        Ok(result)
    }
}

struct BoundApplyCtx<'a> {
    root: &'a ArrayRef,
    bindings: Option<&'a ArrayRef>,
}

impl BoundApplyCtx<'_> {
    fn apply(&self, expr: &BoundExpression) -> VortexResult<ArrayRef> {
        match expr {
            BoundExpression::Root { .. } => Ok(self.root.clone()),
            BoundExpression::Lambda(_) => {
                vortex_bail!("cannot apply a lambda outside a higher-order function")
            }
            BoundExpression::Variable(variable) => {
                let Some(bindings) = self.bindings else {
                    vortex_bail!("cannot apply variable '{variable}' without a provided value");
                };
                GetItem::try_new(bindings.clone(), binding_name(variable.variable_ref()))?
                    .into_array()
                    .optimize()
            }
            BoundExpression::Scalar {
                scalar_fn,
                children,
                ..
            } => apply_bound_scalar_fn(self, scalar_fn, children),
        }
    }
}

fn binding_name(variable_ref: VariableRef) -> FieldName {
    FieldName::from(format!(
        "frame[{}].slot[{}]",
        variable_ref.frame(),
        variable_ref.slot()
    ))
}

fn apply_bound_scalar_fn(
    ctx: &BoundApplyCtx<'_>,
    scalar_fn: &ScalarFnRef,
    children: &[BoundExpression],
) -> VortexResult<ArrayRef> {
    if let Some(scalar) = scalar_fn.as_opt::<Literal>() {
        return Ok(ConstantArray::new(scalar.clone(), ctx.root.len()).into_array());
    }

    let children: Vec<_> = children
        .iter()
        .map(|child| ctx.apply(child))
        .try_collect()?;
    let array =
        ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, ctx.root.len())?.into_array();
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
    use vortex_error::vortex_bail;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::arrays::ScalarFn;
    use crate::arrays::scalar_fn::ScalarFnArrayExt;
    use crate::expr::Lambda;
    use crate::expr::Scope;
    use crate::expr::Variable;
    use crate::expr::binary;
    use crate::expr::lambda;
    use crate::expr::var;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::operators::Operator;

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

    #[test]
    fn bound_lambda_pack_is_eliminated() -> VortexResult<()> {
        let parameter = buffer![1_i32, 2, 3].into_array();
        let capture = buffer![10_i32, 20, 30].into_array();
        let scope = Scope::new(parameter.dtype().clone())
            .with_bindings([(Variable::new("capture"), capture.dtype().clone())])?
            .with_bindings([(Variable::new("x"), parameter.dtype().clone())])?;

        let identity = crate::expr::BoundLambda::bind(&Lambda::try_new(["x"], var("x"))?, &scope)?
            .apply(parameter.clone(), std::slice::from_ref(&parameter), &[])?;
        assert!(ArrayRef::ptr_eq(&identity, &parameter));

        let lambda = crate::expr::BoundLambda::bind(
            &Lambda::try_new(["x"], binary(Operator::Add, var("x"), var("capture")))?,
            &scope,
        )?;
        let result = lambda.apply(
            parameter.clone(),
            std::slice::from_ref(&parameter),
            std::slice::from_ref(&capture),
        )?;
        let Some(result) = result.as_opt::<ScalarFn>() else {
            vortex_bail!("bound lambda did not produce a ScalarFnArray");
        };
        assert!(result.scalar_fn().is::<Binary>());
        assert!(ArrayRef::ptr_eq(result.child_at(0), &parameter));
        assert!(ArrayRef::ptr_eq(result.child_at(1), &capture));
        Ok(())
    }
}
