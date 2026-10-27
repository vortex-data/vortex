// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::make_minmax_dtype;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::array::RunEndArraySlotsExt;
use crate::scalar::Scalar;

/// RunEnd-specific min/max kernel.
///
/// Run-end encoded arrays store each unique run value once, so min/max can be computed directly
/// on the values array without decoding.
#[derive(Debug)]
pub(crate) struct RunEndMinMaxKernel;

impl DynAggregateKernel for RunEndMinMaxKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(options) = aggregate_fn.as_opt::<MinMax>() else {
            return Ok(None);
        };

        let Some(run_end) = batch.as_opt::<RunEnd>() else {
            return Ok(None);
        };

        let struct_dtype = make_minmax_dtype(batch.dtype());
        match min_max(run_end.values(), ctx, *options)? {
            Some(result) => Ok(Some(Scalar::struct_(
                struct_dtype,
                vec![result.min, result.max],
            ))),
            None => Ok(Some(Scalar::null(struct_dtype))),
        }
    }
}
