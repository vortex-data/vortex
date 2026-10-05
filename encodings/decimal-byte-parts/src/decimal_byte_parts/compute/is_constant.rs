// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArraySlotsExt;

/// DecimalByteParts-specific is_constant kernel.
///
/// Delegates to checking that every part is constant: the MSP (most significant part) plus
/// each lower part. An all-null array is constant regardless of the bits its lower parts
/// hold in null slots.
#[derive(Debug)]
pub(crate) struct DecimalBytePartsIsConstantKernel;

impl DynAggregateKernel for DecimalBytePartsIsConstantKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<IsConstant>() {
            return Ok(None);
        }

        let Some(array) = batch.as_opt::<DecimalByteParts>() else {
            return Ok(None);
        };

        let result = is_constant_parts(array, ctx)?;
        Ok(Some(IsConstant::make_partial(batch, result, ctx)?))
    }
}

fn is_constant_parts(
    array: ArrayView<'_, DecimalByteParts>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    if !is_constant(array.msp(), ctx)? {
        return Ok(false);
    }
    // Null slots hold undefined bits in the lower parts, so they cannot make a constant
    // (all-null) array non-constant.
    if array.array().all_invalid(ctx)? {
        return Ok(true);
    }
    for part in array.lower_parts().iter() {
        if !is_constant(part, ctx)? {
            return Ok(false);
        }
    }
    Ok(true)
}
