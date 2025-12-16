// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Either;
use vortex_dtype::UnsignedPType;
use vortex_vector::VectorOps;
use vortex_vector::fixed_size_list::FixedSizeListVector;
use vortex_vector::primitive::PVector;

use crate::take::Take;

impl<I: UnsignedPType> Take<PVector<I>> for &FixedSizeListVector {
    type Output = FixedSizeListVector;

    fn take(self, indices: &PVector<I>) -> FixedSizeListVector {
        if indices.validity().all_true() {
            self.take(indices.elements().as_slice())
        } else {
            take_nullable(self, indices)
        }
    }
}

impl<I: UnsignedPType> Take<[I]> for &FixedSizeListVector {
    type Output = FixedSizeListVector;

    fn take(self, indices: &[I]) -> FixedSizeListVector {
        let list_size = self.list_size() as usize;
        let taken_validity = self.validity().take(indices);

        // If we are in the degenerate case (`list_size == 0`), then we just need to `take` on the
        // validity (since there is no backing data).
        if list_size == 0 {
            debug_assert!(
                self.elements().is_empty(),
                "The elements of a degenerate `FixedSizeListVector` is somehow non-empty"
            );

            // SAFETY: `list_size` is 0 and elements array is empty, so the invariant is maintained.
            return unsafe {
                FixedSizeListVector::new_unchecked(
                    self.elements().clone(),
                    self.list_size(),
                    taken_validity,
                )
            };
        }

        let element_indices = expand_indices(indices, self.list_size());
        let taken_elements = self.elements().as_ref().take(element_indices.as_slice());

        debug_assert_eq!(taken_elements.len(), indices.len() * list_size);
        debug_assert_eq!(taken_validity.len(), indices.len());

        // SAFETY: We took elements at expanded indices and validity at list indices, maintaining
        // the invariant that `elements.len() == validity.len() * list_size`.
        unsafe {
            FixedSizeListVector::new_unchecked(
                Arc::new(taken_elements),
                self.list_size(),
                taken_validity,
            )
        }
    }
}

fn take_nullable<I: UnsignedPType>(
    fsl: &FixedSizeListVector,
    indices: &PVector<I>,
) -> FixedSizeListVector {
    let list_size = fsl.list_size() as usize;
    let taken_validity = fsl.validity().take(indices);

    // If we are in the degenerate case (`list_size == 0`), then we just need to `take` on the
    // validity (since there is no backing data).
    if list_size == 0 {
        debug_assert!(
            fsl.elements().is_empty(),
            "The elements of a degenerate `FixedSizeListVector` is somehow non-empty"
        );

        // SAFETY: `list_size` is 0 and elements array is empty, so the invariant is maintained.
        return unsafe {
            FixedSizeListVector::new_unchecked(
                fsl.elements().clone(),
                fsl.list_size(),
                taken_validity,
            )
        };
    }

    let expanded_nullable_indices = expand_nullable_indices(indices, list_size);
    let taken_elements = fsl
        .elements()
        .as_ref()
        .take(expanded_nullable_indices.as_slice());

    debug_assert_eq!(taken_elements.len(), indices.len() * list_size);
    debug_assert_eq!(taken_validity.len(), indices.len());

    // SAFETY: We took elements at expanded indices and validity at list indices, maintaining the
    // invariant that `elements.len() == validity.len() * list_size`.
    unsafe {
        FixedSizeListVector::new_unchecked(
            Arc::new(taken_elements),
            fsl.list_size(),
            taken_validity,
        )
    }
}

// TODO(connor): Ideally we match on the pointe width and return either a `Vec<u64>` or `Vec<u32>`
// (feature gated by `#[cfg(target_pointer_width = "64")]`), but that is probably overkill.
/// "Expands" the given indices by constructing new indices where for each list index `i`, we fill
/// add indices `[i*list_size..(i+1)*list_size]`.
///
/// The resulting indices vector will have length `indices.len() * list_size`.
fn expand_indices<I: UnsignedPType>(indices: &[I], list_size: u32) -> Vec<u64> {
    let list_size_u64 = list_size as u64;

    // Expand list indices to element indices.
    let expanded_indices: Vec<u64> = indices
        .iter()
        .flat_map(|idx| {
            let list_idx = (*idx).as_() as u64;
            let start = list_idx * list_size_u64;
            let end = (list_idx + 1) * list_size_u64;
            start..end
        })
        .collect();

    debug_assert_eq!(indices.len() * list_size as usize, expanded_indices.len());
    expanded_indices
}

// TODO(connor): Ideally we match on the pointe width and return either a `Vec<u64>` or `Vec<u32>`
// (feature gated by `#[cfg(target_pointer_width = "64")]`), but that is probably overkill.
/// "Expands" the given indices by constructing new indices where for each list index `i`, we fill
/// add indices `[i*list_size..(i+1)*list_size]`.
///
/// The resulting indices vector will have length `indices.len() * list_size`.
///
/// For null indices that we need to expand, we use index 0 as a placeholder for all of them since
/// the validity will mask them out anyway.
fn expand_nullable_indices<I: UnsignedPType>(indices: &PVector<I>, list_size: usize) -> Vec<u64> {
    let indices_validity = indices.validity();
    let list_size_u64 = list_size as u64;

    indices_validity.iter_bools(|validity_iter| {
        let expanded_indices: Vec<u64> = indices
            .elements()
            .iter()
            .zip(validity_iter)
            .flat_map(|(idx, is_valid)| {
                if is_valid {
                    let list_idx = (*idx).as_() as u64;
                    let start = list_idx * list_size_u64;
                    let end = (list_idx + 1) * list_size_u64;
                    Either::Left(start..end)
                } else {
                    Either::Right(std::iter::repeat_n(0u64, list_size))
                }
            })
            .collect();

        debug_assert_eq!(indices.len() * list_size, expanded_indices.len());
        expanded_indices
    })
}
