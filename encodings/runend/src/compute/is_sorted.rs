// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray as _;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
use vortex_array::aggregate_fn::fns::is_sorted::is_strict_sorted;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::RunEnd;

/// RunEnd-specific is_sorted kernel.
///
/// Non-strict: values array sorted implies the run-end array is sorted.
/// Strict: must canonicalize since runs repeat values.
#[derive(Debug)]
pub(crate) struct RunEndIsSortedKernel;

impl DynAggregateKernel for RunEndIsSortedKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(options) = aggregate_fn.as_opt::<IsSorted>() else {
            return Ok(None);
        };

        let Some(array) = batch.as_opt::<RunEnd>() else {
            return Ok(None);
        };

        let result = if options.strict {
            // Strict sort with run-end encoding means we need to canonicalize
            // since run-end encoding repeats values.
            is_strict_sorted(&array.to_canonical()?.into_array(), ctx)?
        } else {
            is_sorted(array.values(), ctx)?
        };

        Ok(Some(IsSorted::make_partial(batch, result, options.strict)?))
    }
}
