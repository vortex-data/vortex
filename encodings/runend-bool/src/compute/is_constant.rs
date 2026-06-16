// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::RunEndBool;
use crate::array::RunEndBoolArrayExt;

/// RunEndBool-specific is_constant kernel.
///
/// A non-nullable run-end bool array with a single run is constant. Other cases defer to the
/// default canonicalization path by returning `None`.
#[derive(Debug)]
pub(crate) struct RunEndBoolIsConstantKernel;

impl DynAggregateKernel for RunEndBoolIsConstantKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<IsConstant>() {
            return Ok(None);
        }

        let Some(array) = batch.as_opt::<RunEndBool>() else {
            return Ok(None);
        };

        // Single physical run with no validity child => every element is identical.
        if array.ends().len() == 1
            && matches!(
                array.bool_validity(),
                vortex_array::validity::Validity::NonNullable
                    | vortex_array::validity::Validity::AllValid
            )
        {
            return Ok(Some(IsConstant::make_partial(batch, true, ctx)?));
        }

        Ok(None)
    }
}
