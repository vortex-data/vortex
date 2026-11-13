// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::IsSortedIteratorExt;
use super::IsSortedOptions;
use crate::ExecutionCtx;
use crate::arrays::DecimalArray;
use crate::dtype::NativeDecimalType;
use crate::match_each_decimal_value_type;

pub(super) fn check_decimal_sorted(
    array: &DecimalArray,
    options: &IsSortedOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    match_each_decimal_value_type!(array.values_type(), |S| {
        compute_is_sorted::<S>(array, options, ctx)
    })
}

fn compute_is_sorted<T: NativeDecimalType>(
    array: &DecimalArray,
    options: &IsSortedOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool>
where
    dyn Iterator<Item = T>: IsSortedIteratorExt,
{
    match array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?
    {
        Mask::AllFalse(_) => Ok(!options.strict),
        Mask::AllTrue(_) => {
            let buf = array.buffer::<T>();
            let iter = buf.iter().copied();

            Ok(options.is_sorted(iter))
        }
        Mask::Values(mask_values) => {
            let values = array.buffer::<T>();
            let iter = mask_values
                .bit_buffer()
                .iter()
                .zip_eq(values)
                .map(|(is_valid, v)| is_valid.then_some(v));

            Ok(options.is_nullable_sorted(iter))
        }
    }
}
