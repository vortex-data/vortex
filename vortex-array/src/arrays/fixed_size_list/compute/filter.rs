// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::{ConstantArray, FixedSizeListArray, FixedSizeListVTable};
use crate::compute::{self, FilterKernel, FilterKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

impl FilterKernel for FixedSizeListVTable {
    fn filter(&self, array: &FixedSizeListArray, selection_mask: &Mask) -> VortexResult<ArrayRef> {
        let new_len = selection_mask.true_count();
        let null_mask = array.validity_mask();

        // If the entire array is null, then we only need to adjust the length of the array.
        if let Mask::AllFalse(_) = null_mask {
            return Ok(ConstantArray::new(
                Scalar::null(array.dtype().clone()),
                selection_mask.true_count(),
            )
            .into_array());
        }

        let elements = array.elements();
        let list_size = array.list_size();

        let new_validity = array.validity().filter(selection_mask)?;
        debug_assert!(new_validity.maybe_len().is_none_or(|len| len == new_len));

        let new_elements = {
            // We want to create a new mask specialized to the underlying `elements` of the array.
            if list_size != 0 {
                let elements_mask = compute_fsl_elements_mask(selection_mask, list_size);

                // Allow the child array to filter themselves.
                let new_elements = compute::filter(elements, &elements_mask)?;
                debug_assert_eq!(new_elements.len(), new_len * list_size as usize);

                new_elements
            } else {
                // We make a special case for degenerate `FixedSizeList` arrays.
                debug_assert_eq!(
                    elements.len(),
                    0,
                    "degenerate FixedSizeListArray is invalid"
                );

                // NB: The safety comment for the `list_size == 0` case is here for clarity.

                // SAFETY: We have verified that when `list_size == 0`
                // - `elements` has length 0 (since it came from a valid `FixedSizeListArray`)
                // - `new_validity` has the correct length because we filter with the same
                //   `selection_mask` as the array itself
                elements.clone()
            }
        };

        Ok(
            // SAFETY: We have verified that
            // - The case when `list_size == 0` is safe (see above)
            // - The `new_elements` array is guaranteed to have a length that is a multiple of
            //   `list_size`
            // - `new_validity` has the correct length because we filter with the same
            //   `selection_mask` as the array itself
            unsafe {
                FixedSizeListArray::new_unchecked(new_elements, list_size, new_validity, new_len)
            }
            .into_array(),
        )
    }
}

register_kernel!(FilterKernelAdapter(FixedSizeListVTable).lift());

/// Given a mask for a fixed-size list array, creates a new mask for the underlying elements.
///
/// This function simply "expands" out the input `fsl_mask` by duplicating each bit `list_size`
/// times.
///
/// The output `Mask` is guaranteed to have a length equal to `fsl_mask.len() * list_size`.
fn compute_fsl_elements_mask(fsl_mask: &Mask, list_size: u32) -> Mask {
    let expanded_len = fsl_mask.len() * list_size as usize;

    match fsl_mask {
        Mask::AllTrue(_) => Mask::AllTrue(expanded_len),
        Mask::AllFalse(_) => Mask::AllFalse(expanded_len),
        Mask::Values(values) => {
            let mut builder = arrow_buffer::BooleanBufferBuilder::new(expanded_len);

            for value in values.boolean_buffer().iter() {
                builder.append_n(list_size as usize, value);
            }

            Mask::from_buffer(builder.finish())
        }
    }
}
