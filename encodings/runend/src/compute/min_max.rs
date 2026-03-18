// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::fns::min_max::make_minmax_dtype;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::RunEnd;

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
        if !aggregate_fn.is::<MinMax>() {
            return Ok(None);
        }

        let Some(run_end) = batch.as_opt::<RunEnd>() else {
            return Ok(None);
        };

        let struct_dtype = make_minmax_dtype(batch.dtype());
        match min_max(run_end.values(), ctx)? {
            Some(result) => Ok(Some(Scalar::struct_(
                struct_dtype,
                vec![result.min, result.max],
            ))),
            None => Ok(Some(Scalar::null(struct_dtype))),
        }
    }
}
