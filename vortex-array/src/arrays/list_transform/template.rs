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
use crate::arrays::TemplateInputArray;
use crate::arrays::template::TemplateInputArrayExt;
use crate::arrays::template::TemplateScope;
use crate::expr::BoundExpression;
use crate::expr::BoundLambda;
use crate::scalar_fn::fns::literal::Literal;

/// Convert one bound lambda into its zero-length, lexically scoped body template.
pub(super) fn build_template(lambda: &BoundLambda) -> VortexResult<ArrayRef> {
    let scope = TemplateScope::fresh();
    // Slots 0 and 1 are permanently reserved for the element and optional local index. Captures
    // always start at 2, which keeps a one-parameter lambda's first capture distinct from the
    // index slot after the bound lambda itself has been discarded.
    let mut inputs = vec![None; 2 + lambda.captures().len()];
    inputs[0] =
        Some(TemplateInputArray::new(scope, 0, lambda.param_dtypes()[0].clone()).into_array());
    if lambda.param_dtypes().len() == 2 {
        inputs[1] =
            Some(TemplateInputArray::new(scope, 1, lambda.param_dtypes()[1].clone()).into_array());
    }
    for (index, capture) in lambda.captures().iter().enumerate() {
        inputs[index + 2] =
            Some(TemplateInputArray::new(scope, index + 2, capture.dtype().clone()).into_array());
    }
    build_expression(lambda.body(), lambda, scope, &inputs)
}

fn build_expression(
    expression: &BoundExpression,
    lambda: &BoundLambda,
    scope: TemplateScope,
    inputs: &[Option<ArrayRef>],
) -> VortexResult<ArrayRef> {
    match expression {
        BoundExpression::Root { .. } => template_input(inputs, 0),
        BoundExpression::Variable(variable) => {
            let slot = lambda
                .param_refs()
                .iter()
                .position(|reference| *reference == variable.variable_ref())
                .or_else(|| {
                    lambda
                        .captures()
                        .iter()
                        .position(|capture| capture.variable_ref() == variable.variable_ref())
                        .map(|index| 2 + index)
                })
                .ok_or_else(|| {
                    vortex_error::vortex_err!(
                        "variable '{}' is unresolved while building a template",
                        variable
                    )
                })?;
            let input = template_input(inputs, slot)?;
            vortex_ensure!(
                input.as_::<crate::arrays::TemplateInput>().scope() == scope,
                "template builder mixed scopes"
            );
            Ok(input)
        }
        BoundExpression::Scalar {
            scalar_fn,
            children,
            ..
        } => {
            if let Some(value) = scalar_fn.as_opt::<Literal>() {
                return Ok(ConstantArray::new(value.clone(), 0).into_array());
            }
            let children = children
                .iter()
                .map(|child| build_expression(child, lambda, scope, inputs))
                .try_collect()?;
            Ok(ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, 0)?.into_array())
        }
        BoundExpression::Lambda(_) => {
            vortex_bail!("a detached lambda cannot appear in a template body")
        }
        BoundExpression::ListTransform {
            lambda: nested_lambda,
            children,
            ..
        } => {
            let list = build_expression(&children[0], lambda, scope, inputs)?;
            let captures = children[1..]
                .iter()
                .map(|capture| build_expression(capture, lambda, scope, inputs))
                .collect::<VortexResult<Vec<_>>>()?;
            crate::arrays::ListTransformArray::try_new(list, nested_lambda.clone(), captures)
                .map(IntoArray::into_array)
        }
    }
}

fn template_input(inputs: &[Option<ArrayRef>], slot: usize) -> VortexResult<ArrayRef> {
    inputs
        .get(slot)
        .and_then(Option::as_ref)
        .cloned()
        .ok_or_else(|| vortex_error::vortex_err!("template input slot {slot} is not bound"))
}
