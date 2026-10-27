// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::is_sorted;
use crate::aggregate_fn::fns::is_sorted::is_strict_sorted;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::scalar::Scalar;

/// Computes [`IsSorted`] of a [`Widen`] from its values.
///
/// Decoding preserves order and validity, so the values give the verdict of the `Widen`, sorted
/// or not. [`IsSorted::make_partial`] reads the first and last values through the `Widen`, so the
/// partial holds the decoded values.
#[derive(Debug)]
pub(crate) struct WidenIsSortedKernel;

impl DynAggregateKernel for WidenIsSortedKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(options) = aggregate_fn.as_opt::<IsSorted>() else {
            return Ok(None);
        };

        let Some(widen) = batch.as_opt::<Widen>() else {
            return Ok(None);
        };

        let strict = options.strict;
        let values = widen.values();

        // NB: The accumulator reports a batch without a valid row as sorted but not strictly
        // sorted. The free `is_strict_sorted` reports any array of at most one row as strictly
        // sorted, so it disagrees on a single null.
        let is_sorted = if values.all_invalid(ctx)? {
            !strict
        } else if strict {
            is_strict_sorted(values, ctx)?
        } else {
            is_sorted(values, ctx)?
        };

        Ok(Some(IsSorted::make_partial(batch, is_sorted, strict, ctx)?))
    }
}
