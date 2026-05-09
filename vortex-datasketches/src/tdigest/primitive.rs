// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::ToPrimitive;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::PrimitiveScalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::TDigestPartial;

pub(super) fn update_primitive(
    partial: &mut TDigestPartial,
    scalar: PrimitiveScalar<'_>,
) -> VortexResult<()> {
    if let Some(value) = scalar.pvalue() {
        partial.update(value.cast::<f64>()?);
    }
    Ok(())
}

pub(super) fn accumulate_primitive(
    partial: &mut TDigestPartial,
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match_each_native_ptype!(array.ptype(), |T| {
        accumulate_typed::<T>(partial, array, ctx)
    })
}

fn accumulate_typed<T>(
    partial: &mut TDigestPartial,
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()>
where
    T: NativePType + ToPrimitive,
{
    let values = array.as_slice::<T>();
    match array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?
    {
        Mask::AllTrue(_) => {
            for &value in values {
                partial.update(value.to_f64().vortex_expect("primitive converts to f64"));
            }
        }
        Mask::AllFalse(_) => {}
        Mask::Values(validity) => {
            for (&value, valid) in values.iter().zip(validity.bit_buffer().iter()) {
                if valid {
                    partial.update(value.to_f64().vortex_expect("primitive converts to f64"));
                }
            }
        }
    }
    Ok(())
}
