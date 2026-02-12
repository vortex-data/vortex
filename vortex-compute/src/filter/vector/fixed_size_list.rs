// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_mask::Mask;
use vortex_mask::MaskMut;
use vortex_vector::Vector;
use vortex_vector::VectorMut;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;
use vortex_vector::fixed_size_list::FixedSizeListVector;
use vortex_vector::fixed_size_list::FixedSizeListVectorMut;

use crate::filter::Filter;

// TODO(connor): Implement filtering for the other mask types (`BitView`).

/// Density threshold for choosing between indices and slices representation when expanding masks.
///
/// When the mask density is below this threshold, we use indices. Otherwise, we use slices.
///
/// Note that this is somewhat arbitrarily chosen...
const MASK_EXPANSION_DENSITY_THRESHOLD: f64 = 0.05;

impl<M> Filter<M> for &FixedSizeListVector
where
    for<'a> &'a Mask: Filter<M, Output = Mask>,
    for<'a> &'a Vector: Filter<[(usize, usize)], Output = Vector>,
    M: ?Sized,
{
    type Output = FixedSizeListVector;

    fn filter(self, selection: &M) -> Self::Output {
        let list_size = self.list_size();
        let filtered_validity = self.validity().filter(selection);

        let filtered_elements = if list_size != 0 {
            // Expand the mask to cover all elements within selected lists.
            let elements_mask = compute_fsl_elements_mask(&filtered_validity, list_size as usize);

            // Filter the child elements vector.
            self.elements().as_ref().filter(elements_mask.as_slice())
        } else {
            debug_assert!(
                self.elements().is_empty(),
                "degenerate FixedSizeListVector is invalid, it should have no elements"
            );

            self.elements().as_ref().clone()
        };

        // SAFETY: We have verified that:
        // - The case when `list_size == 0` is safe (elements is empty and stays empty).
        // - The `filtered_elements` is guaranteed to have length that is a multiple of `list_size`.
        // - `filtered_validity` has the correct length because we filter with the same
        //   `selection` mask.
        unsafe {
            FixedSizeListVector::new_unchecked(
                Arc::new(filtered_elements),
                list_size,
                filtered_validity,
            )
        }
    }
}

impl<M> Filter<M> for &mut FixedSizeListVectorMut
where
    for<'a> &'a mut MaskMut: Filter<M, Output = ()>,
    for<'a> &'a mut VectorMut: Filter<[(usize, usize)], Output = ()>,
    M: ?Sized,
{
    type Output = ();

    fn filter(self, selection: &M) -> Self::Output {
        let list_size = self.list_size();

        // Filter validity first to get the new length.
        // SAFETY: We will ensure the elements vector is filtered with an appropriately sized mask
        // to maintain the invariant `elements.len() == len * list_size`.
        unsafe {
            self.validity_mut().filter(selection);
            self.set_len(self.validity().len());
        }

        if list_size != 0 {
            // Expand the mask to cover all elements within selected lists.
            // We need to freeze a copy of the validity to get a Mask for the computation.
            let validity_frozen = self.validity().clone().freeze();
            let elements_mask = compute_fsl_elements_mask(&validity_frozen, list_size as usize);

            // Filter the elements vector with the expanded mask.
            // SAFETY: The expanded mask has the correct length (`validity.len() * list_size`),
            // which maintains the invariant after filtering.
            unsafe {
                self.elements_mut().filter(elements_mask.as_slice());
            }

            debug_assert_eq!(
                self.elements().len(),
                self.len() * list_size as usize,
                "elements length must equal len * list_size after filtering"
            );
        } else {
            debug_assert!(
                self.elements().is_empty(),
                "degenerate FixedSizeListVector is invalid, it should have no elements"
            );
        }
    }
}

impl<M> Filter<M> for FixedSizeListVector
where
    for<'a> &'a FixedSizeListVector: Filter<M, Output = FixedSizeListVector>,
    for<'a> &'a mut FixedSizeListVectorMut: Filter<M, Output = ()>,
{
    type Output = Self;

    fn filter(self, selection: &M) -> Self {
        match self.try_into_mut() {
            // If we have exclusive access, we can perform the filter in place.
            Ok(mut vector_mut) => {
                (&mut vector_mut).filter(selection);
                vector_mut.freeze()
            }
            // Otherwise, allocate a new vector and fill it in (delegate to the
            // `&FixedSizeListVector` impl).
            Err(vector) => (&vector).filter(selection),
        }
    }
}

/// Given a mask for a fixed-size list array, creates a new mask for the underlying elements.
///
/// This function simply "expands" out the input `selection_mask` by duplicating each bit
/// `list_size` times.
///
/// The output [`Mask`] is guaranteed to have a length equal to `selection_mask.len() * list_size`.
fn compute_fsl_elements_mask(selection_mask: &Mask, list_size: usize) -> Vec<(usize, usize)> {
    // let expanded_len = selection_mask.len() * list_size;

    let values = match selection_mask {
        Mask::AllTrue(_) => return vec![(0, selection_mask.len() * list_size)],
        Mask::AllFalse(_) => return vec![],
        Mask::Values(values) => values,
    };

    let expanded_slices = if values.density() >= MASK_EXPANSION_DENSITY_THRESHOLD {
        values
            .bit_buffer()
            .set_slices()
            .map(|(start, end)| (start * list_size, end * list_size))
            .collect()
    } else {
        values
            .bit_buffer()
            .set_indices()
            .map(|idx| {
                let start = idx * list_size;
                let end = (idx + 1) * list_size;
                (start, end)
            })
            .collect()
    };

    expanded_slices
}
