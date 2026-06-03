// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::IsSortedIteratorExt;
use crate::ExecutionCtx;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;

pub(super) fn check_bool_sorted(
    array: &BoolArray,
    strict: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    match array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?
    {
        Mask::AllFalse(_) => Ok(!strict),
        Mask::AllTrue(_) => {
            let values = array.bit_buffer_view();
            Ok(if strict {
                values.iter().is_strict_sorted()
            } else {
                values.iter().is_sorted()
            })
        }
        Mask::Values(mask_values) => {
            if strict {
                let validity_buffer = mask_values.bit_buffer();
                let values = array.bit_buffer_view();
                Ok(validity_buffer
                    .iter()
                    .zip(values.iter())
                    .map(|(is_valid, value)| is_valid.then_some(value))
                    .is_strict_sorted())
            } else {
                let set_indices = mask_values.bit_buffer().set_indices();
                let values = array.bit_buffer_view();
                let values_iter = set_indices.map(|idx|
                    // Safety:
                    // All idxs are in-bounds for the array.
                    unsafe {
                        values.value_unchecked(idx)
                    });
                Ok(values_iter.is_sorted())
            }
        }
    }
}
