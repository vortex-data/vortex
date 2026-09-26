// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::compute::windowed_values;

/// RunEnd-specific is_constant kernel.
///
/// If every run value the array's logical window covers is the same, the whole array is constant.
/// Runs outside that window hold no rows, so they must not take part — see [`windowed_values`].
#[derive(Debug)]
pub(crate) struct RunEndIsConstantKernel;

impl DynAggregateKernel for RunEndIsConstantKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<IsConstant>() {
            return Ok(None);
        }

        let Some(array) = batch.as_opt::<RunEnd>() else {
            return Ok(None);
        };

        let values = windowed_values(&array, batch.len(), ctx)?;
        let result = is_constant(&values, ctx)?;
        Ok(Some(IsConstant::make_partial(batch, result, ctx)?))
    }
}
