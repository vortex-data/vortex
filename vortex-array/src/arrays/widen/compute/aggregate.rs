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
use crate::dtype::DType;
use crate::expr::stats::Stat;
use crate::scalar::Scalar;

/// The longest batch with a 64-bit logical dtype for which [`WidenArrayAggregate`] serves [`Sum`].
///
/// `Sum` adds integers narrower than 64 bits in blocks and checks for overflow once per block, but
/// checks 64-bit integers after every value. So only a 64-bit logical dtype is checked at other
/// points than its values. Its values have at most 32 bits, and no sum of at most 2^32 of them
/// leaves the 64-bit range, so neither check fires. In a longer batch, a sum can overflow and come
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

        // NB: Exactness rests on integer decoding. A decimal `Widen` (CD4 in #10733) needs its own
        // argument, because `Sum` checks a decimal for overflow against the precision of its input.
        if !batch.dtype().is_int() {
            return Ok(None);
        }

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
/// - The overflow checks **must** agree. A logical dtype narrower than 64 bits is summed in the
///   same blocks as its values, so they always agree. A 64-bit logical dtype needs a batch of at
///   most [`MAX_EXACT_SUM_LEN`] rows.
///
/// Only `Sum` with `skip_nans` has the slot. Integers have no NaNs, so `skip_nans` matters here
/// only because it selects the slot.
fn is_exact_sum(aggregate_fn: &AggregateFnRef, batch: &ArrayRef) -> bool {
    let is_64_bit = matches!(batch.dtype(), DType::Primitive(ptype, _) if ptype.byte_width() == 8);

    Stat::from_aggregate_fn(aggregate_fn) == Some(Stat::Sum)
        && (!is_64_bit || batch.len() as u64 <= MAX_EXACT_SUM_LEN)
}
