// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::array::child_to_validity;
use crate::arrays::ConstantArray;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::vtable::ArrayExpr;
use crate::arrays::scalar_fn::vtable::FakeEq;
use crate::arrays::scalar_fn::vtable::ScalarFn;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::expr::lit;
use crate::scalar_fn::TypedScalarFnInstance;
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::root::Root;
use crate::validity::Validity;

/// Convert an expression tree into a lazy array DAG without executing it.
///
/// This assumes all leaf expressions are either ArrayExpr (wrapping actual arrays) or Literals.
fn expr_to_lazy_array(expr: &Expression, row_count: usize) -> VortexResult<ArrayRef> {
    // Handle Root expression - this should not happen in validity expressions
    if expr.is::<Root>() {
        vortex_bail!("Root expression cannot be converted in validity context");
    }

    // Handle Literal expression - create a constant array
    if expr.is::<Literal>() {
        let scalar = expr.as_::<Literal>();
        return Ok(ConstantArray::new(scalar.clone(), row_count).into_array());
    }

    // Handle ArrayExpr leaves - unwrap the array they hold
    if expr.is::<ArrayExpr>() {
        return Ok(expr.as_::<ArrayExpr>().0.clone());
    }

    // Recursively convert child expressions into lazy input arrays
    let children: Vec<ArrayRef> = expr
        .children()
        .iter()
        .map(|child| expr_to_lazy_array(child, row_count))
        .collect::<VortexResult<_>>()?;

    Ok(Array::<ScalarFn>::try_new(expr.scalar_fn().clone(), children, row_count)?.into_array())
}

/// Execute an expression tree recursively.
///
/// This assumes all leaf expressions are either ArrayExpr (wrapping actual arrays) or Literals.
fn execute_expr(expr: &Expression, row_count: usize) -> VortexResult<ArrayRef> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    // Handle Root expression - this should not happen in validity expressions
    if expr.is::<Root>() {
        vortex_bail!("Root expression cannot be executed in validity context");
    }

    // Handle Literal expression - create a constant array
    if expr.is::<Literal>() {
        let scalar = expr.as_::<Literal>();
        return Ok(ConstantArray::new(scalar.clone(), row_count).into_array());
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

impl ValidityVTable<ScalarFn> for ScalarFn {
    fn validity(array: ArrayView<'_, ScalarFn>) -> VortexResult<Validity> {
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

        match array.scalar_fn().validity_opt(&expr)? {
            Some(validity_expr) => {
                // The function defines its validity as an expression over its inputs, so we can
                // represent it as a lazy array DAG without executing anything. If the expression
                // is already a constant it is folded back into AllValid/AllInvalid.
                let validity_array = expr_to_lazy_array(&validity_expr, array.len())?;
                Ok(child_to_validity(
                    Some(&validity_array),
                    Nullability::Nullable,
                ))
            }
            None => {
                // The function's validity can only be determined by executing the function
                // itself (e.g. Kleene logic and/or). Representing that lazily would create a
                // self-referential array (is_not_null over this very expression), so execute it
                // eagerly instead.
                let validity_expr = array.scalar_fn().validity(&expr)?;
                Ok(Validity::Array(execute_expr(&validity_expr, array.len())?))
            }
        }
    }
}
