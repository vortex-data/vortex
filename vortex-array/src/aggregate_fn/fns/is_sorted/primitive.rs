// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::IsSortedOptions;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::NativeValue;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;

pub(super) fn check_primitive_sorted(
    array: &PrimitiveArray,
    options: &IsSortedOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    match_each_native_ptype!(array.ptype(), |P| {
        compute_is_sorted::<P>(array, options, ctx)
    })
}

fn compute_is_sorted<T: NativePType>(
    array: &PrimitiveArray,
    options: &IsSortedOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    match array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?
    {
        Mask::AllFalse(_) => Ok(!options.strict),
        Mask::AllTrue(_) => {
            let slice = array.as_slice::<T>();
            let iter = slice.iter().copied().map(NativeValue);

            Ok(options.is_sorted(iter))
        }
        Mask::Values(mask_values) => {
            let iter = mask_values
                .bit_buffer()
                .iter()
                .zip_eq(array.as_slice::<T>())
                .map(|(is_valid, value)| is_valid.then_some(NativeValue(*value)));

            Ok(options.is_nullable_sorted(iter))
        }
    }
}
