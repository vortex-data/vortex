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
/// [`Sum`] adds integers narrower than 64 bits in blocks and checks for overflow once per block,
/// but checks `i64` and `u64` after every value. So for a 64-bit logical dtype, the sum of the
/// values and the sum of the canonical batch check at different points. The values have at most 32
/// bits, so every running sum of at most 2^32 of them fits in 64 bits, and neither check fires. In
/// a longer batch, the sum can overflow and come back inside one block, which only the check after
/// every value catches.
const MAX_EXACT_SUM_LEN: u64 = 1 << 32;

/// Runs an aggregate of a [`Widen`] on its values, and casts the partial to the logical dtype.
///
/// [`AggregateFnSession::default`] registers this kernel only for aggregates whose partial is the
/// same on the values and on the `Widen`, which the [module documentation](super) explains.
/// [`Sum`] needs more, so the kernel serves it only when [`is_exact_sum`] holds.
///
/// [`AggregateFnSession::default`]: crate::aggregate_fn::session::AggregateFnSession::default
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

/// Returns whether the [`Sum`] of `batch` on its values equals its sum on the canonical batch.
///
/// Two conditions make this hold.
///
/// - `aggregate_fn` has the legacy [`Stat::Sum`] slot. The accumulator then sums each batch from
///   zero, caches the total as that statistic, and adds the total to the running sum. Without the
///   slot, the accumulator adds each value of the batch to the running sum, and the kernel cannot
///   match that near the 64-bit bounds. For example, from a running sum of `i64::MAX - 5`, the
///   batch `[10, -10]` overflows at its first value, but adding its total of zero does not.
/// - `batch` has at most [`MAX_EXACT_SUM_LEN`] rows, which that constant explains.
///
/// Only `Sum` with `skip_nans` has the slot, because the statistic skips NaNs. NaNs do not occur
/// in integers, so `skip_nans` matters here only because it selects the slot. [`SumV2`] has no
/// slot, so the session does not register this kernel for it.
///
/// [`SumV2`]: crate::aggregate_fn::fns::sum_v2::SumV2
fn is_exact_sum(aggregate_fn: &AggregateFnRef, batch: &ArrayRef) -> bool {
    Stat::from_aggregate_fn(aggregate_fn) == Some(Stat::Sum)
        && batch.len() as u64 <= MAX_EXACT_SUM_LEN
}
