// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray as _;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::is_sorted;
use crate::aggregate_fn::fns::is_sorted::is_strict_sorted;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::array::RunEndArraySlotsExt;
use crate::scalar::Scalar;

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
            is_strict_sorted(
                &array
                    .array()
                    .clone()
                    .execute::<Canonical>(ctx)?
                    .into_array(),
                ctx,
            )?
        } else {
            is_sorted(array.values(), ctx)?
        };

        Ok(Some(IsSorted::make_partial(
            batch,
            result,
            options.strict,
            ctx,
        )?))
    }
}
