// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{BoolArray, BoolVTable};
use crate::compute::{IsSortedIteratorExt, IsSortedKernel, IsSortedKernelAdapter};
use crate::register_kernel;

impl IsSortedKernel for BoolVTable {
    fn is_sorted(&self, array: &BoolArray) -> VortexResult<Option<bool>> {
        match array.validity_mask() {
            Mask::AllFalse(_) => Ok(Some(true)),
            Mask::AllTrue(_) => Ok(Some(array.bit_buffer().iter().is_sorted())),
            Mask::Values(mask_values) => {
                let set_indices = mask_values.bit_buffer().set_indices();
                let values = array.bit_buffer();
                let values_iter = set_indices.map(|idx|
                    // Safety:
                    // All idxs are in-bounds for the array.
                    unsafe {
                        values.value_unchecked(idx)
                    });

                Ok(Some(values_iter.is_sorted()))
            }
        }
    }

    fn is_strict_sorted(&self, array: &BoolArray) -> VortexResult<Option<bool>> {
        match array.validity_mask() {
            Mask::AllFalse(_) => Ok(Some(false)),
            Mask::AllTrue(_) => Ok(Some(array.bit_buffer().iter().is_strict_sorted())),
            Mask::Values(mask_values) => {
                let validity_buffer = mask_values.bit_buffer();
                let values = array.bit_buffer();

                Ok(Some(
                    validity_buffer
                        .iter()
                        .zip(values.iter())
                        .map(|(is_valid, value)| is_valid.then_some(value))
                        .is_strict_sorted(),
                ))
            }
        }
    }
}

register_kernel!(IsSortedKernelAdapter(BoolVTable).lift());
