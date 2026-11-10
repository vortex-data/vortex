// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::IsSortedOptions;
use crate::ExecutionCtx;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;

pub(super) fn check_bool_sorted(
    array: &BoolArray,
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
            let values = array.bit_buffer_view();
            Ok(options.is_sorted(values.iter()))
        }
        Mask::Values(mask_values) => {
            // Nulls sort first, so map each element to `Option` where `None < Some(_)`.
            let values = array.bit_buffer_view();
            let iter = mask_values
                .bit_buffer()
                .iter()
                .zip(values.iter())
                .map(|(is_valid, value)| is_valid.then_some(value));
            Ok(options.is_nullable_sorted(iter))
        }
    }
}
