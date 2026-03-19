// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::IsSortedIteratorExt;
use crate::arrays::BoolArray;

pub(super) fn check_bool_sorted(array: &BoolArray, strict: bool) -> VortexResult<bool> {
    match array.validity_mask()? {
        Mask::AllFalse(_) => Ok(!strict),
        Mask::AllTrue(_) => {
            let values = array.to_bit_buffer();
            Ok(if strict {
                values.iter().is_strict_sorted()
            } else {
                values.iter().is_sorted()
            })
        }
        Mask::Values(mask_values) => {
            if strict {
                let validity_buffer = mask_values.bit_buffer();
                let values = array.to_bit_buffer();
                Ok(validity_buffer
                    .iter()
                    .zip(values.iter())
                    .map(|(is_valid, value)| is_valid.then_some(value))
                    .is_strict_sorted())
            } else {
                let set_indices = mask_values.bit_buffer().set_indices();
                let values = array.to_bit_buffer();
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
