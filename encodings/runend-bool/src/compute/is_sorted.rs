// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray as _;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
use vortex_array::aggregate_fn::fns::is_sorted::is_strict_sorted;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::RunEndBool;

/// RunEndBool-specific is_sorted kernel.
///
/// Sortedness depends on the decoded values, so we canonicalize and delegate.
#[derive(Debug)]
pub(crate) struct RunEndBoolIsSortedKernel;

impl DynAggregateKernel for RunEndBoolIsSortedKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(options) = aggregate_fn.as_opt::<IsSorted>() else {
            return Ok(None);
        };

        if batch.as_opt::<RunEndBool>().is_none() {
            return Ok(None);
        }

        let canonical = batch.clone().execute::<Canonical>(ctx)?.into_array();
        let result = if options.strict {
            is_strict_sorted(&canonical, ctx)?
        } else {
            is_sorted(&canonical, ctx)?
        };

        Ok(Some(IsSorted::make_partial(
            batch,
            result,
            options.strict,
            ctx,
        )?))
    }
}
