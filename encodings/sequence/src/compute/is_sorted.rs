// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::zero;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::Sequence;

/// Sequence-specific is_sorted kernel.
///
/// A sequence `A[i] = base + i * multiplier` is sorted iff multiplier >= 0,
/// and strict sorted iff multiplier > 0.
#[derive(Debug)]
pub(crate) struct SequenceIsSortedKernel;

impl DynAggregateKernel for SequenceIsSortedKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(options) = aggregate_fn.as_opt::<IsSorted>() else {
            return Ok(None);
        };

        let Some(array) = batch.as_opt::<Sequence>() else {
            return Ok(None);
        };

        let m = array.multiplier();
        let result = match_each_native_ptype!(m.ptype(), |P| {
            m.cast::<P>().map(|x| {
                if options.strict {
                    x > zero::<P>()
                } else {
                    x >= zero::<P>()
                }
            })
        })?;

        Ok(Some(IsSorted::make_partial(
            batch,
            result,
            options.strict,
            ctx,
        )?))
    }
}
