// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::fns::sum::accumulate_primitive;
use crate::aggregate_fn::fns::sum::make_zero_state;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Constant;
use crate::arrays::PrimitiveArray;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::dtype::Nullability;
use crate::scalar::Scalar;

/// The longest batch that [`WidenSumKernel`] sums. The `NB:` comment at its check explains the
/// bound.
const MAX_EXACT_SUM_LEN: u64 = 1 << 32;

/// Computes [`Sum`] with `skip_nans` of a [`Widen`] by summing its values into the sum state of
/// the logical dtype.
///
/// Each value widens exactly into the state, so the partial equals the partial of the canonical
/// batch. The accumulator also sums every batch of this aggregate from zero before it merges it,
/// through the statistics bridge. [`SumV2`] and `Sum` without `skip_nans` have no bridge: they
/// add each value to the running state and check every 64-bit value for overflow, which the
/// partial of one batch cannot match near the bounds of `i64`. So they keep the default path.
///
/// [`SumV2`]: crate::aggregate_fn::fns::sum_v2::SumV2
#[derive(Debug)]
pub(crate) struct WidenSumKernel;

impl DynAggregateKernel for WidenSumKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(options) = aggregate_fn
            .as_opt::<Sum>()
            .filter(|options| options.skip_nans)
        else {
            return Ok(None);
        };

        let Some(widen) = batch.as_opt::<Widen>() else {
            return Ok(None);
        };

        let values = widen.values();

        // On the default path the cast moves into values with a sum kernel of their own, so that
        // kernel sums at the logical dtype. The kernel can be registered for `Sum` exactly, as for
        // `RunEnd` or `Sparse`, or for every aggregate, as for `Chunked`, whose kernel lets each
        // chunk use its own kernel or cached statistic.
        if ctx
            .aggregate_kernels()
            .find_aggregate_kernel(values.encoding_id(), Sum.id())
            .is_some()
        {
            return Ok(None);
        }

        // The default path sums a constant without decoding it.
        if values.is::<Constant>() {
            return Ok(None);
        }

        // NB: The values are at most 32 bits wide, so no prefix sum of at most 2^32 values leaves
        // the 64-bit range. Then neither this kernel, which checks for overflow after each chunk,
        // nor the canonical sum, which checks after each 64-bit value, overflows within the batch.
        // A longer batch can overflow and recover within one chunk, which only the canonical sum
        // sees.
        if batch.len() as u64 > MAX_EXACT_SUM_LEN {
            return Ok(None);
        }

        let return_dtype = Sum.return_dtype(options, batch.dtype()).vortex_expect(
            "the logical dtype of a Widen is an integer, which validate_dtypes checks, and Sum \
             supports every integer",
        );
        let values = values.clone().execute::<PrimitiveArray>(ctx)?;
        let mut state = make_zero_state(&return_dtype);
        let overflowed = accumulate_primitive(&mut state, &values, ctx, options.skip_nans)?;

        if overflowed {
            vortex_panic!(
                "a Widen sum of at most 2^32 values of at most 32 bits cannot overflow, got an \
                 overflow over {} values",
                batch.len()
            );
        }

        Ok(Some(state.to_scalar(&return_dtype, Nullability::Nullable)))
    }
}
