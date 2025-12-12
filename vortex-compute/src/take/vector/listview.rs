// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Implementations of `take` on [`ListViewVector`].
//!
//! take` on the list view simply performs a `take` on the views of the vector, which means the
//! resulting vector may have "garbage" data in its `elements` child vector.
//!
//! Note that it is on the outer array type to perform compaction / garbage collection.

use vortex_dtype::UnsignedPType;
use vortex_vector::VectorOps;
use vortex_vector::listview::ListViewVector;
use vortex_vector::primitive::PVector;

use crate::take::Take;

impl<I: UnsignedPType> Take<PVector<I>> for &ListViewVector {
    type Output = ListViewVector;

    fn take(self, indices: &PVector<I>) -> ListViewVector {
        if indices.validity().all_true() {
            self.take(indices.elements().as_slice())
        } else {
            take_nullable(self, indices)
        }
    }
}

impl<I: UnsignedPType> Take<[I]> for &ListViewVector {
    type Output = ListViewVector;

    fn take(self, indices: &[I]) -> ListViewVector {
        let taken_offsets = self.offsets().take(indices);
        let taken_sizes = self.sizes().take(indices);
        let taken_validity = self.validity().take(indices);

        debug_assert_eq!(taken_offsets.len(), taken_validity.len());
        debug_assert_eq!(taken_sizes.len(), taken_validity.len());

        // SAFETY: We called take on offsets, sizes, and validity with the same indices, so the new
        // components must have the same length. The offsets and sizes still point into the same
        // elements array which we clone via Arc, so all view references remain valid.
        unsafe {
            ListViewVector::new_unchecked(
                self.elements().clone(),
                taken_offsets,
                taken_sizes,
                taken_validity,
            )
        }
    }
}

fn take_nullable<I: UnsignedPType>(
    list_view: &ListViewVector,
    indices: &PVector<I>,
) -> ListViewVector {
    // We ignore nullability when taking the offsets and sizes since we can let the `Mask`
    // implementation determine which elements are null.
    let taken_offsets = list_view.offsets().take(indices.elements().as_slice());
    let taken_sizes = list_view.sizes().take(indices.elements().as_slice());

    // Note that this is **not** the same as the `indices: &[I]` `take` implementation above.
    let taken_validity = list_view.validity().take(indices);

    debug_assert_eq!(taken_offsets.len(), taken_validity.len());
    debug_assert_eq!(taken_sizes.len(), taken_validity.len());

    // SAFETY: We used the same indices to take from all components, so they should still have the
    // same length. The offsets and sizes still point into the same elements array which we clone
    // via Arc, so all view references remain valid.
    unsafe {
        ListViewVector::new_unchecked(
            list_view.elements().clone(),
            taken_offsets,
            taken_sizes,
            taken_validity,
        )
    }
}
