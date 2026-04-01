// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::scalar_fn::vtable::ArrayExpr;
use crate::arrays::scalar_fn::vtable::FakeEq;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::expr::Expression;
use crate::expr::lit;
use crate::scalar_fn::ScalarFn;
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

impl ValidityVTable<ScalarFnVTable> for ScalarFnVTable {
    fn validity(array: ArrayView<'_, ScalarFnVTable>) -> VortexResult<Validity> {
        let inputs: Vec<_> = array
            .iter_children()
            .map(|child| {
                if let Some(scalar) = child.as_constant() {
                    return Ok(lit(scalar));
                }
                Expression::try_new(ScalarFn::new(ArrayExpr, FakeEq(child.clone())).erased(), [])
            })
            .collect::<VortexResult<_>>()?;

        let expr = Expression::try_new(array.scalar_fn().clone(), inputs)?;
        let validity_expr = array.scalar_fn().validity(&expr)?;

        // Execute the validity expression. All leaves are ArrayExpr nodes.
        Ok(Validity::Array(execute_expr(&validity_expr, array.len())?))
    }
}
