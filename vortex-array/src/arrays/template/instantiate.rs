// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ListTransformArrayExt;
use crate::arrays::ScalarFn;
use crate::arrays::ScalarFnArray;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::template::TemplateInput;
use crate::arrays::template::TemplateInputArrayExt;
use crate::arrays::template::TemplateScope;

/// Rebuild a template body in an invocation row domain.
///
/// Only the encodings a lambda builder can emit are accepted.  In particular, this intentionally
/// does not attempt generic arbitrary-array substitution.  Nested list transforms are boundaries:
/// their list and capture children are rebuilt, while their own zero-length body is left sealed.
pub(crate) fn instantiate(
    body: &ArrayRef,
    scope: TemplateScope,
    inputs: &[ArrayRef],
) -> VortexResult<ArrayRef> {
    let invocation_len = inputs.first().map_or(0, ArrayRef::len);
    vortex_ensure!(
        inputs.iter().all(|input| input.len() == invocation_len),
        "template invocation inputs must have a common length"
    );
    instantiate_inner(body, scope, inputs, invocation_len)
}

fn instantiate_inner(
    body: &ArrayRef,
    scope: TemplateScope,
    inputs: &[ArrayRef],
    invocation_len: usize,
) -> VortexResult<ArrayRef> {
    if let Some(input) = body.as_opt::<TemplateInput>() {
        vortex_ensure!(
            input.scope() == scope,
            "unresolved template input from a different template scope"
        );
        let actual = inputs.get(input.slot()).ok_or_else(|| {
            vortex_error::vortex_err!("template input slot {} is unresolved", input.slot())
        })?;
        vortex_ensure!(
            actual.dtype() == body.dtype(),
            "template input slot {} expects dtype {}, got {}",
            input.slot(),
            body.dtype(),
            actual.dtype()
        );
        return Ok(actual.clone());
    }

    if let Some(constant) = body.as_opt::<Constant>() {
        return Ok(ConstantArray::new(constant.scalar().clone(), invocation_len).into_array());
    }

    if let Some(scalar_fn) = body.as_opt::<ScalarFn>() {
        let children = scalar_fn
            .iter_children()
            .map(|child| instantiate_inner(child, scope, inputs, invocation_len))
            .collect::<VortexResult<Vec<_>>>()?;
        return Ok(ScalarFnArray::try_new_with_len(
            scalar_fn.scalar_fn().clone(),
            children,
            invocation_len,
        )?
        .into_array());
    }

    if let Some(transform) = body.as_opt::<crate::arrays::ListTransform>() {
        let list = instantiate_inner(transform.list(), scope, inputs, invocation_len)?;
        let captures = transform
            .captures()
            .map(|capture| instantiate_inner(capture, scope, inputs, invocation_len))
            .collect::<VortexResult<Vec<_>>>()?;
        return crate::arrays::ListTransformArray::try_new_from_parts(
            list,
            transform.body().clone(),
            captures,
        )
        .map(IntoArray::into_array);
    }

    vortex_bail!(
        "unsupported symbolic encoding {} in a template body",
        body.encoding_id()
    )
}

/// Infer the outer scope represented by a template body.
///
/// An all-constant body has no symbolic scope and needs no substitution.  The walk deliberately
/// stops at a nested transform body so an inner lambda cannot be captured by its outer lambda.
pub(crate) fn template_scope(body: &ArrayRef) -> VortexResult<Option<TemplateScope>> {
    fn visit(body: &ArrayRef, found: &mut Option<TemplateScope>) -> VortexResult<()> {
        if let Some(input) = body.as_opt::<TemplateInput>() {
            if let Some(scope) = found {
                vortex_ensure!(
                    *scope == input.scope(),
                    "template body contains inputs from more than one scope"
                );
            } else {
                *found = Some(input.scope());
            }
            return Ok(());
        }
        if let Some(scalar_fn) = body.as_opt::<ScalarFn>() {
            for child in scalar_fn.iter_children() {
                visit(child, found)?;
            }
            return Ok(());
        }
        if let Some(transform) = body.as_opt::<crate::arrays::ListTransform>() {
            visit(transform.list(), found)?;
            for capture in transform.captures() {
                visit(capture, found)?;
            }
        }
        Ok(())
    }

    let mut scope = None;
    visit(body, &mut scope)?;
    Ok(scope)
}
