// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_mask::MaskIter;
use vortex_mask::MaskValues;

use crate::arrays::FixedSizeListArray;
use crate::arrays::filter::execute::filter_validity;

/// Density threshold for choosing between indices and slices representation when expanding masks.
///
/// When the mask density is below this threshold, we use indices. Otherwise, we use slices.
const MASK_EXPANSION_DENSITY_THRESHOLD: f64 = 0.05;

/// Filter implementation for [`FixedSizeListArray`].
///
/// Expands the selection mask to cover all elements within selected lists and pushes the expanded
/// mask down to the child elements array.
pub fn filter_fixed_size_list(
    array: &FixedSizeListArray,
    selection_mask: &Arc<MaskValues>,
) -> FixedSizeListArray {
    let filtered_validity = filter_validity(array.validity(), selection_mask);

    let elements = array.elements();
    let new_len = selection_mask.true_count();
    let list_size = array.list_size();

    let new_elements = {
        // We want to create a new mask specialized to the underlying `elements` of the array.
        if list_size != 0 {
            // TODO(connor): If we can push down a "indices" or "slices" selection instead that
            // would be much more performant.
            let elements_mask = compute_mask_for_fsl_elements(selection_mask, list_size as usize);

            // Allow the child array to filter itself.
            let new_elements = elements
                .filter(elements_mask)
                .vortex_expect("FixedSizeListArray elements are guaranteed to support filter");
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
            // - `filtered_validity` has the correct length because we filter with the same
            //   `selection_mask` as the array itself
            elements.clone()
        }
    };

    // SAFETY: We have verified that
    // - The case when `list_size == 0` is safe (see above)
    // - The `new_elements` array is guaranteed to have a length that is a multiple of
    //   `list_size`
    // - `filtered_validity` has the correct length because we filter with the same
    //   `selection_mask` as the array itself
    unsafe {
        FixedSizeListArray::new_unchecked(new_elements, list_size, filtered_validity, new_len)
    }
}

/// Given a mask for a fixed-size list array, creates a new mask for the underlying elements.
///
/// This function simply "expands" out the input `selection_mask` by duplicating each bit
/// `list_size` times.
///
/// The output `Mask` is guaranteed to have a length equal to `selection_mask.len() * list_size`.
fn compute_mask_for_fsl_elements(selection_mask: &MaskValues, list_size: usize) -> Mask {
    let expanded_len = selection_mask.len() * list_size;

    // Use threshold_iter to choose the optimal representation based on density.
    let expanded_slices = match selection_mask.threshold_iter(MASK_EXPANSION_DENSITY_THRESHOLD) {
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

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::dtype::Nullability;
    use crate::validity::Validity;

    #[test]
    fn test_filter_fixed_size_list_conformance() {
        let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8, 9]);
        let array = FixedSizeListArray::new(elements.into_array(), 3, Validity::NonNullable, 3);
        test_filter_conformance(&array.into_array());
    }

    #[test]
    fn test_filter_fixed_size_list_with_nulls_conformance() {
        let elements =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), Some(5), None]);
        let validity = Validity::from_iter([true, false, true]);
        let array = FixedSizeListArray::new(elements.into_array(), 2, validity, 3);
        test_filter_conformance(&array.into_array());
    }

    #[test]
    fn filter_fixed_size_list_selects_correct_lists() {
        let elements = PrimitiveArray::from_iter([10i32, 20, 30, 40, 50, 60]);
        let array = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

        let mask = Mask::from_iter([true, false, true]);
        let filtered = array.filter(mask).unwrap();

        // Should select lists 0 and 2: [10, 20] and [50, 60].
        let expected_elements = PrimitiveArray::from_iter([10i32, 20, 50, 60]);
        let expected =
            FixedSizeListArray::new(expected_elements.into_array(), 2, Validity::NonNullable, 2);

        assert_arrays_eq!(filtered, expected);
    }

    #[test]
    fn filter_degenerate_list_size_zero() {
        let elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
        let array = FixedSizeListArray::new(elements.into_array(), 0, Validity::NonNullable, 5);

        let mask = Mask::from_iter([true, false, true, false, true]);
        let filtered = array.filter(mask).unwrap();

        let expected_elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
        let expected =
            FixedSizeListArray::new(expected_elements.into_array(), 0, Validity::NonNullable, 3);

        assert_arrays_eq!(filtered, expected);
    }

    #[test]
    fn filter_nested_fixed_size_lists() {
        // Inner lists of size 2, outer lists of size 2 (so 2 outer lists, each with 2 inner lists).
        let inner_elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8].into_array();
        let inner_fsl = FixedSizeListArray::new(inner_elements, 2, Validity::NonNullable, 4);
        let outer_fsl =
            FixedSizeListArray::new(inner_fsl.into_array(), 2, Validity::NonNullable, 2);

        // Keep only the second outer list.
        let mask = Mask::from_iter([false, true]);
        let filtered = outer_fsl.filter(mask).unwrap();

        let expected_inner_elements = buffer![5i32, 6, 7, 8].into_array();
        let expected_inner =
            FixedSizeListArray::new(expected_inner_elements, 2, Validity::NonNullable, 2);
        let expected_outer =
            FixedSizeListArray::new(expected_inner.into_array(), 2, Validity::NonNullable, 1);

        assert_arrays_eq!(filtered, expected_outer);
    }
}
