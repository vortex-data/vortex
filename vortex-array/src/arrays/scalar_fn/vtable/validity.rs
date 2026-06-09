// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::vtable::ArrayExpr;
use crate::arrays::scalar_fn::vtable::FakeEq;
use crate::arrays::scalar_fn::vtable::ScalarFn;
use crate::expr::Expression;
use crate::expr::lit;
use crate::scalar::Scalar;
use crate::scalar_fn::TypedScalarFnInstance;
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::root::Root;
use crate::validity::Validity;

/// Execute an expression tree recursively.
///
/// This assumes all leaf expressions are either ArrayExpr (wrapping actual arrays) or Literals.
fn execute_expr(expr: &Expression, row_count: usize) -> VortexResult<ArrayRef> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    // Handle Root expression - this should not happen in validity expressions
    if expr.is::<Root>() {
        vortex_error::vortex_bail!("Root expression cannot be executed in validity context");
    }

    // Handle Literal expression - create a constant array
    if expr.is::<Literal>() {
        let scalar = expr.as_::<Literal>();
        return Ok(crate::arrays::ConstantArray::new(scalar.clone(), row_count).into_array());
    }

    // Recursively execute child expressions to get input arrays
    let inputs: Vec<ArrayRef> = expr
        .children()
        .iter()
        .map(|child| execute_expr(child, row_count))
        .collect::<VortexResult<_>>()?;

    let args = VecExecutionArgs::new(inputs, row_count);

    Ok(expr.scalar_fn().execute(&args, &mut ctx)?.into_array())
}

/// Collapse a constant boolean validity scalar into the most specific [`Validity`] variant.
///
/// Returns `None` if the scalar is not a definite boolean (e.g. a null), in which case the caller
/// should keep the validity as an array.
fn constant_validity(scalar: &Scalar) -> Option<Validity> {
    scalar.as_bool().value().map(|valid| {
        if valid {
            Validity::AllValid
        } else {
            Validity::AllInvalid
        }
    })
}

impl ValidityVTable<ScalarFn> for ScalarFn {
    fn validity(array: ArrayView<'_, ScalarFn>) -> VortexResult<Validity> {
        // A non-nullable result dtype guarantees there are no nulls, so we can skip building and
        // evaluating any validity expression entirely. This also keeps downstream `no_nulls`
        // fast-paths intact instead of handing them a constant-true validity array.
        if !array.dtype().is_nullable() {
            return Ok(Validity::NonNullable);
        }

        let inputs: Vec<_> = array
            .iter_children()
            .map(|child| {
                if let Some(scalar) = child.as_constant() {
                    return Ok(lit(scalar));
                }
                Expression::try_new(
                    TypedScalarFnInstance::new(ArrayExpr, FakeEq(child.clone())).erased(),
                    [],
                )
            })
            .collect::<VortexResult<_>>()?;

        let expr = Expression::try_new(array.scalar_fn().clone(), inputs)?;
        let validity_expr = array.scalar_fn().validity(&expr)?;

        // A literal validity expression collapses to a constant validity without evaluating
        // anything (e.g. functions whose result is always valid return `lit(true)`).
        if let Some(scalar) = validity_expr.as_opt::<Literal>()
            && let Some(validity) = constant_validity(scalar)
        {
            return Ok(validity);
        }

        // Otherwise evaluate the validity expression. All leaves are ArrayExpr or Literal nodes.
        let validity_array = execute_expr(&validity_expr, array.len())?;

        // Collapse a constant result into the most specific variant so that fast-paths keying off
        // `Validity::NonNullable | AllValid | AllInvalid` are not defeated by a constant array.
        if let Some(scalar) = validity_array.as_constant()
            && let Some(validity) = constant_validity(&scalar)
        {
            return Ok(validity);
        }

        Ok(Validity::Array(validity_array))
    }
}
