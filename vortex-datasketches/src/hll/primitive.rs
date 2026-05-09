// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::PrimitiveScalar;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::HllPartial;

pub(super) fn update_primitive(
    partial: &mut HllPartial,
    scalar: PrimitiveScalar<'_>,
) -> VortexResult<()> {
    if let Some(value) = scalar.pvalue() {
        partial.update_value(value);
    }
    Ok(())
}

pub(super) fn accumulate_primitive(
    partial: &mut HllPartial,
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match_each_native_ptype!(array.ptype(), |T| {
        accumulate_typed::<T>(partial, array, ctx)
    })
}

fn accumulate_typed<T>(
    partial: &mut HllPartial,
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()>
where
    T: NativePType,
    PValue: From<T>,
{
    let values = array.as_slice::<T>();
    match array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?
    {
        Mask::AllTrue(_) => {
            for &value in values {
                partial.update_value(PValue::from(value));
            }
        }
        Mask::AllFalse(_) => {}
        Mask::Values(validity) => {
            for (&value, valid) in values.iter().zip(validity.bit_buffer().iter()) {
                if valid {
                    partial.update_value(PValue::from(value));
                }
            }
        }
    }
    Ok(())
}
