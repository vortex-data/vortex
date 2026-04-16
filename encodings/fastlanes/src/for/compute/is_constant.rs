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

use crate::FoR;
use crate::r#for::array::FoRArrayExt;

/// FoR-specific is_constant kernel.
///
/// Delegates to checking if the encoded array is constant.
#[derive(Debug)]
pub(crate) struct FoRIsConstantKernel;

impl DynAggregateKernel for FoRIsConstantKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<IsConstant>() {
            return Ok(None);
        }

        let Some(array) = batch.as_opt::<FoR>() else {
            return Ok(None);
        };

        let result = is_constant(array.encoded(), ctx)?;
        Ok(Some(IsConstant::make_partial(batch, result, ctx)?))
    }
}
