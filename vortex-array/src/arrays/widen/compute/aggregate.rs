// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::expr::stats::Stat;
use crate::scalar::Scalar;

/// The longest batch for which [`WidenArrayAggregate`] serves [`Sum`].
///
/// `Sum` adds integers narrower than 64 bits in blocks and checks for overflow once per block, but
/// checks 64-bit integers after every value. No sum of at most 2^32 values of at most 32 bits
/// leaves the 64-bit range, so both checks agree. In a longer batch, a sum can overflow and come
/// back inside one block, which only the check after every value sees.
const MAX_EXACT_SUM_LEN: u64 = 1 << 32;

/// Runs an aggregate of a [`Widen`] on its values and widens the partial to the logical dtype.
///
/// The values run the same accumulator code as the canonical array, so edge cases such as an empty
/// batch or a single null match by construction, and the values keep their own cached statistics
/// and kernels. [`Sum`] is served only when [`is_exact_sum`] holds.
#[derive(Debug)]
pub(crate) struct WidenArrayAggregate;

impl DynAggregateKernel for WidenArrayAggregate {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(widen) = batch.as_opt::<Widen>() else {
            return Ok(None);
        };

        if aggregate_fn.is::<Sum>() && !is_exact_sum(aggregate_fn, batch) {
            return Ok(None);
        }

        let values = widen.values();
        let mut accumulator = aggregate_fn.accumulator(values.dtype())?;
        accumulator.accumulate(values, ctx)?;

        let partial_dtype = aggregate_fn.state_dtype(batch.dtype()).vortex_expect(
            "an aggregate that accepts the values accepts the logical dtype, which is a wider \
             integer of the same signedness",
        );

        Ok(Some(accumulator.flush()?.cast(&partial_dtype)?))
    }
}

/// Returns whether the [`Sum`] of the values of `batch` equals the sum of the canonical batch.
///
/// - The accumulator **must** sum each batch from zero and then add the total to the running sum.
///   It does so for an aggregate with the legacy [`Stat::Sum`] slot. Otherwise it adds each value
///   to the running sum, which a batch total cannot match near the 64-bit bounds: from a running
///   sum of `i64::MAX - 5`, the values `[10, -10]` overflow, but their total of zero does not.
/// - `batch` **must** have at most [`MAX_EXACT_SUM_LEN`] rows.
///
/// Only `Sum` with `skip_nans` has the slot. Integers have no NaNs, so `skip_nans` matters here
/// only because it selects the slot.
fn is_exact_sum(aggregate_fn: &AggregateFnRef, batch: &ArrayRef) -> bool {
    Stat::from_aggregate_fn(aggregate_fn) == Some(Stat::Sum)
        && batch.len() as u64 <= MAX_EXACT_SUM_LEN
}
