// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution of an array for the rows a caller demands.
//!
//! A demand is a mask of the rows whose values the caller reads. The values of the other rows are
//! unspecified, so a scalar function may ignore the demand and compute every row. A function can
//! also use it to skip work: `AND` and `OR` demand of their right operand only the rows their left
//! operand does not decide.

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::Operator;
use crate::scalar_fn::fns::switch::Switch;

/// Execute `array` to its canonical form for the rows of `demand`, or for every row without one.
///
/// The values of rows outside the demand are unspecified. A computation that uses the demand, such
/// as `AND` and `OR`, receives it through its [`ExecutionArgs`]. Every other array is executed
/// whole, as the executor executes it: a comparison of an encoded array, for example, runs the
/// comparison kernel of the encoding instead of decoding the array first.
///
/// [`ExecutionArgs`]: crate::scalar_fn::ExecutionArgs
pub fn execute_demanded(
    array: ArrayRef,
    demand: Option<&Mask>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let Some(demand) = demand.filter(|demand| !demand.all_true()) else {
        return Ok(array.execute::<Canonical>(ctx)?.into_array());
    };
    if demand.all_false() {
        return Ok(
            ConstantArray::new(Scalar::default_value(array.dtype()), array.len()).into_array(),
        );
    }
    let Some(scalar_fn_array) = array
        .as_opt::<ScalarFn>()
        .filter(|array| uses_demand(array.scalar_fn()))
    else {
        return Ok(array.execute::<Canonical>(ctx)?.into_array());
    };

    let args =
        VecExecutionArgs::new(scalar_fn_array.children(), array.len()).with_demand(demand.clone());
    Ok(scalar_fn_array
        .scalar_fn()
        .execute(&args, ctx)?
        .execute::<Canonical>(ctx)?
        .into_array())
}

/// Whether `scalar_fn` evaluates less of its inputs for a demand. Calling another function with
/// its inputs directly would skip the kernels the executor finds for them.
fn uses_demand(scalar_fn: &ScalarFnRef) -> bool {
    scalar_fn
        .as_opt::<Binary>()
        .is_some_and(|operator| matches!(operator, Operator::And | Operator::Or))
        || scalar_fn.is::<Switch>()
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::arrays::Constant;
    use crate::arrays::PrimitiveArray;
    use crate::builtins::ArrayBuiltins;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// The demanded rows hold the values of a full evaluation.
    #[test]
    fn demanded_rows_match_full_evaluation() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let len = 4096;
        let step = 64;
        let values = PrimitiveArray::from_iter(0..len as u64).into_array();
        let comparison = values.binary(
            ConstantArray::new(Scalar::from(1000u64), len).into_array(),
            Operator::Lt,
        )?;
        let demand = Mask::from_indices(len, (0..len).step_by(step).collect::<Vec<_>>());

        let full = comparison
            .clone()
            .execute::<Canonical>(&mut ctx)?
            .into_array();
        let demanded = execute_demanded(comparison, Some(&demand), &mut ctx)?;
        for idx in (0..len).step_by(step) {
            assert_eq!(
                demanded.execute_scalar(idx, &mut ctx)?,
                full.execute_scalar(idx, &mut ctx)?
            );
        }
        Ok(())
    }

    #[test]
    fn nothing_demanded_is_not_evaluated() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter([1u64, 2, 3]).into_array();
        let demanded = execute_demanded(values, Some(&Mask::new_false(3)), &mut ctx)?;
        assert!(demanded.is::<Constant>());
        Ok(())
    }
}
