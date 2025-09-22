// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::{Mask, MaskIter};

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::compute::{self, FilterKernel, FilterKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

/// Density threshold for choosing between indices and slices representation when expanding masks.
///
/// When the mask density is below this threshold, we use indices. Otherwise, we use slices.
///
/// Note that this is somewhat arbitrarily chosen...
const MASK_EXPANSION_DENSITY_THRESHOLD: f64 = 0.05;

/// Filter implementation for [`FixedSizeListArray`].
///
/// Expands the selection mask to cover all elements within selected lists and pushes the expanded
/// mask down to the child elements array.
impl FilterKernel for FixedSizeListVTable {
    fn filter(&self, array: &FixedSizeListArray, selection_mask: &Mask) -> VortexResult<ArrayRef> {
        let elements = array.elements();
        let new_len = selection_mask.true_count();
        let list_size = array.list_size();

        let new_validity = array.validity().filter(selection_mask)?;
        debug_assert!(new_validity.maybe_len().is_none_or(|len| len == new_len));

        let new_elements = {
            // We want to create a new mask specialized to the underlying `elements` of the array.
            if list_size != 0 {
                let elements_mask = compute_fsl_elements_mask(selection_mask, list_size as usize);

                // Allow the child array to filter itself.
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
/// This function simply "expands" out the input `selection_mask` by duplicating each bit
/// `list_size` times.
///
/// The output `Mask` is guaranteed to have a length equal to `selection_mask.len() * list_size`.
fn compute_fsl_elements_mask(selection_mask: &Mask, list_size: usize) -> Mask {
    let expanded_len = selection_mask.len() * list_size;

    let values = match selection_mask {
        Mask::AllTrue(_) => return Mask::AllTrue(expanded_len),
        Mask::AllFalse(_) => return Mask::AllFalse(expanded_len),
        Mask::Values(values) => values,
    };

    // Use threshold_iter to choose the optimal representation based on density.
    let expanded_slices = match values.threshold_iter(MASK_EXPANSION_DENSITY_THRESHOLD) {
        MaskIter::Slices(slices) => {
            // Expand a dense mask (represented as slices) by scaling each slice by `list_size`.
            slices
                .iter()
                .map(|&(start, end)| (start * list_size, end * list_size))
                .collect()
        }
        MaskIter::Indices(indices) => {
            // Expand a sparse mask (represented as indices) by duplicating each index `list_size`
            // times.
            //
            // Note that in the worst case, it is possible that we create only a few slices with a
            // small range (for example, when list_size <= 2). This could be further optimized,
            // but we choose simplicity for now.
            indices
                .iter()
                .map(|&idx| {
                    let start = idx * list_size;
                    let end = (idx + 1) * list_size;
                    (start, end)
                })
                .collect()
        }
    };

    Mask::from_slices(expanded_len, expanded_slices)
}
