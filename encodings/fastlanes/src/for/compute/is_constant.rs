// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::FL_CHUNK_SIZE;
use crate::FoR;
use crate::r#for::array::FoRArrayExt;
use crate::r#for::array::FoRArraySlotsExt;

/// FoR-specific is_constant kernel.
///
/// When every chunk shares one reference, delegates to checking if the encoded array is constant.
/// Otherwise, compares one value from each of two chunks with different references: if they
/// differ, the array is not constant. If they match, it declines.
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
        let result =
            if array.constant_reference().is_some() || is_constant(array.references(), ctx)? {
                is_constant(array.encoded(), ctx)?
            } else if differs_across_references(array, batch, ctx)? {
                false
            } else {
                return Ok(None);
            };
        Ok(Some(IsConstant::make_partial(batch, result, ctx)?))
    }
}

/// Whether the first values of the first chunk and of the first chunk with a different reference
/// are both valid and differ.
fn differs_across_references(
    array: ArrayView<'_, FoR>,
    batch: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let references = array.references().clone().execute::<PrimitiveArray>(ctx)?;
    let Some(chunk) = match_each_integer_ptype!(references.ptype(), |T| {
        let references = references.as_slice::<T>();
        references.iter().position(|r| *r != references[0])
    }) else {
        return Ok(false);
    };

    let first = batch.execute_scalar(0, ctx)?;
    let other = batch.execute_scalar(chunk * FL_CHUNK_SIZE - usize::from(array.offset()), ctx)?;
    Ok(!first.is_null() && !other.is_null() && first != other)
}
