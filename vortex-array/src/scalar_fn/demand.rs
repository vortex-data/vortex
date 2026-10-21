// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution of an array for the rows a caller demands.
//!
//! A demand is a mask of the rows whose values the caller reads. The values of the other rows are
//! unspecified, so a scalar function may ignore the demand and compute every row. A function can
//! also use it to skip work: `AND` and `OR` demand of their right operand only the rows their left
//! operand does not decide.

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::scalar::Scalar;
use crate::scalar_fn::VecExecutionArgs;
use crate::validity::Validity;

/// The largest fraction of demanded rows for which a computation is evaluated on the demanded rows
/// alone, rather than on every row.
const SPARSE_DEMAND_DENSITY: f64 = 0.25;

/// The fewest rows for which evaluating a computation on its demanded rows pays for filtering its
/// inputs and spreading its results back.
const MIN_SPARSE_DEMAND_LEN: usize = 1024;

/// Execute `array` to its canonical form for the rows of `demand`, or for every row without one.
///
/// The values of rows outside the demand are unspecified. A computation that is not evaluated yet,
/// such as a comparison, receives the demand through its [`ExecutionArgs`], and with a sparse
/// demand is evaluated on the demanded rows alone. An encoded or canonical array is executed
/// whole, as its own kernels would be lost to a filter.
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
    let Some(scalar_fn_array) = array.as_opt::<ScalarFn>() else {
        return Ok(array.execute::<Canonical>(ctx)?.into_array());
    };

    if array.len() >= MIN_SPARSE_DEMAND_LEN && demand.density() <= SPARSE_DEMAND_DENSITY {
        let demanded = array
            .filter(demand.clone())?
            .execute::<Canonical>(ctx)?
            .into_array();
        return spread(demanded, demand)?
            .execute::<Canonical>(ctx)
            .map(Canonical::into_array);
    }

    let args =
        VecExecutionArgs::new(scalar_fn_array.children(), array.len()).with_demand(demand.clone());
    Ok(scalar_fn_array
        .scalar_fn()
        .execute(&args, ctx)?
        .execute::<Canonical>(ctx)?
        .into_array())
}

/// Spread `values`, one for each demanded row, back to the rows of `demand`. The rows outside the
/// demand repeat a demanded value.
fn spread(values: ArrayRef, demand: &Mask) -> VortexResult<ArrayRef> {
    let AllOr::Some(bits) = demand.bit_buffer() else {
        return Ok(values);
    };
    let last = u32::try_from(values.len().saturating_sub(1))?;
    let mut indices = BufferMut::<u32>::with_capacity(demand.len());
    let mut position = 0u32;
    for demanded in bits.iter() {
        indices.push(position.min(last));
        position += u32::from(demanded);
    }
    values.take(PrimitiveArray::new(indices.freeze(), Validity::NonNullable).into_array())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::arrays::Constant;
    use crate::builtins::ArrayBuiltins;
    use crate::scalar_fn::fns::operators::Operator;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// The demanded rows hold the values of a full evaluation, whether the demand is sparse enough
    /// to evaluate them alone or not.
    #[rstest]
    #[case::sparse(64)]
    #[case::dense(2)]
    fn demanded_rows_match_full_evaluation(#[case] step: usize) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let len = 4096;
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
