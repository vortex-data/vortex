// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::{Mask, MaskIter};
use vortex_scalar::Scalar;

use crate::arrays::{ConstantArray, FixedSizeListArray, FixedSizeListVTable};
use crate::compute::{self, FilterKernel, FilterKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

/// Density threshold for choosing between indices and slices representation when expanding masks.
///
/// When the mask density is below this threshold, we use indices. Otherwise, we use slices.
///
/// Note that this is somewhat arbitrarily chosen...
const FSL_MASK_EXPANSION_DENSITY_THRESHOLD: f64 = 0.1;

/// List size threshold for choosing between indices and slices in sparse mask expansion.
///
/// When expanding sparse masks, if the list size is at or above this threshold, we convert
/// indices to slices to avoid materializing too many individual indices. This prevents
/// memory bloat when each FSL element contains many items.
///
/// Note that this is somewhat arbitrarily chosen...
const FSL_SPARSE_MASK_LIST_SIZE_THRESHOLD: usize = 8;

impl FilterKernel for FixedSizeListVTable {
    fn filter(&self, array: &FixedSizeListArray, selection_mask: &Mask) -> VortexResult<ArrayRef> {
        let new_len = selection_mask.true_count();
        let null_mask = array.validity_mask();

        // If the entire array is null, then we only need to adjust the length of the array.
        if let Mask::AllFalse(_) = null_mask {
            return Ok(
                ConstantArray::new(Scalar::null(array.dtype().clone()), new_len).into_array(),
            );
        }

        let elements = array.elements();
        let list_size = array.list_size();

        let new_validity = array.validity().filter(selection_mask)?;
        debug_assert!(new_validity.maybe_len().is_none_or(|len| len == new_len));

        let new_elements = {
            // We want to create a new mask specialized to the underlying `elements` of the array.
            if list_size != 0 {
                let elements_mask = compute_fsl_elements_mask(selection_mask, list_size as usize);

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
/// This function simply "expands" out the input `selection_mask` by duplicating each bit
/// `list_size` times.
///
/// The output `Mask` is guaranteed to have a length equal to `selection_mask.len() * list_size`.
fn compute_fsl_elements_mask(selection_mask: &Mask, list_size: usize) -> Mask {
    let expanded_len = selection_mask.len() * list_size;

    match selection_mask {
        Mask::AllTrue(_) => Mask::AllTrue(expanded_len),
        Mask::AllFalse(_) => Mask::AllFalse(expanded_len),
        Mask::Values(values) => {
            // Use threshold_iter to choose the optimal representation based on density.
            match values.threshold_iter(FSL_MASK_EXPANSION_DENSITY_THRESHOLD) {
                MaskIter::Slices(slices) => expand_dense_mask(slices, list_size, expanded_len),
                MaskIter::Indices(indices) => expand_sparse_mask(indices, list_size, expanded_len),
            }
        }
    }
}

/// Expands a dense mask (represented as slices) by scaling each slice by `list_size`.
fn expand_dense_mask(slices: &[(usize, usize)], list_size: usize, expanded_len: usize) -> Mask {
    let expanded_slices: Vec<(usize, usize)> = slices
        .iter()
        .map(|&(start, end)| (start * list_size, end * list_size))
        .collect();

    Mask::from_slices(expanded_len, expanded_slices)
}

/// Expands a sparse mask (represented as indices) by duplicating each index `list_size` times.
fn expand_sparse_mask(indices: &[usize], list_size: usize, expanded_len: usize) -> Mask {
    if list_size < FSL_SPARSE_MASK_LIST_SIZE_THRESHOLD {
        // For small list sizes, expand each index into individual indices.
        let expanded_indices: Vec<usize> = indices
            .iter()
            .flat_map(|&idx| {
                let start = idx * list_size;
                start..start + list_size
            })
            .collect();

        Mask::from_indices(expanded_len, expanded_indices)
    } else {
        // For sparse masks with large list sizes, it's more efficient to create slices rather than
        // materializing all individual indices.
        let expanded_slices: Vec<(usize, usize)> = indices
            .iter()
            .map(|&idx| {
                let start = idx * list_size;
                let end = (idx + 1) * list_size;
                (start, end)
            })
            .collect();

        Mask::from_slices(expanded_len, expanded_slices)
    }
}
