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
            // TODO (joacoc): `primitive`, `decimal` and `bool` repeat the same match on the validity
            // mask, consolidate into a single `is_masked_sorted`.
            if options.strict {
                let validity_buffer = mask_values.bit_buffer();
                let values = array.bit_buffer_view();
                Ok(options.is_nullable_sorted(
                    validity_buffer
                        .iter()
                        .zip(values.iter())
                        .map(|(is_valid, value)| is_valid.then_some(value)),
                ))
            } else {
                // TODO (joacoc): Here nulls are skipped, but everywhere else they come first.
                // For example, `[false, null, true]` is sorted here because the null is skipped,
                // but the same pattern with integers, `[0, null, 1]`, is not sorted.
                // The checks between batches also put nulls first, so `[true, null]` is sorted as
                // one batch, but not when split into `[true]` and `[null]`.
                let set_indices = mask_values.bit_buffer().set_indices();
                let values = array.bit_buffer_view();
                let values_iter = set_indices.map(|idx|
                    // Safety:
                    // All idxs are in-bounds for the array.
                    unsafe {
                        values.value_unchecked(idx)
                    });
                Ok(options.is_sorted(values_iter))
            }
        }
    }
}
