// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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

    fn take(self, _indices: &[I]) -> ListViewVector {
        todo!("TODO(connor): Implement `take` for `ListViewVector` and figure out rebuilding");

        /*

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

        */
    }
}

fn take_nullable<I: UnsignedPType>(
    _lvector: &ListViewVector,
    _indices: &PVector<I>,
) -> ListViewVector {
    todo!("TODO(connor): Implement `take` for `ListViewVector` and figure out rebuilding");

    /*

    // We ignore nullability when taking the offsets and sizes since we can let the `Mask`
    // implementation determine which elements are null.
    let taken_offsets = lvector.offsets().take(indices.elements().as_slice());
    let taken_sizes = lvector.sizes().take(indices.elements().as_slice());
    let taken_validity = lvector.validity().take(indices);

    debug_assert_eq!(taken_offsets.len(), taken_validity.len());
    debug_assert_eq!(taken_sizes.len(), taken_validity.len());

    // SAFETY: We used the same indices to take from all components, so they should still have the
    // same length. The offsets and sizes still point into the same elements array which we clone
    // via Arc, so all view references remain valid.
    unsafe {
        ListViewVector::new_unchecked(
            lvector.elements().clone(),
            taken_offsets,
            taken_sizes,
            taken_validity,
        )
    }

    */
}
