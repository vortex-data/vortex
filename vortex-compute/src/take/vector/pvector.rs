// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::NativePType;
use vortex_dtype::UnsignedPType;
use vortex_vector::VectorOps;
use vortex_vector::primitive::PVector;

use crate::take::Take;

impl<T: NativePType, I: UnsignedPType> Take<PVector<I>> for &PVector<T> {
    type Output = PVector<T>;

    fn take(self, indices: &PVector<I>) -> PVector<T> {
        if indices.validity().all_true() {
            self.take(indices.elements().as_slice())
        } else {
            take_nullable(self, indices)
        }
    }
}

impl<T: NativePType, I: UnsignedPType> Take<[I]> for &PVector<T> {
    type Output = PVector<T>;

    fn take(self, indices: &[I]) -> PVector<T> {
        let taken_elements = self.elements().take(indices);
        let taken_validity = self.validity().take(indices);

        debug_assert_eq!(taken_elements.len(), taken_validity.len());

        // SAFETY: we called take on both components of the vector with the same indices, so the new
        // components must have the same length.
        unsafe { PVector::new_unchecked(taken_elements, taken_validity) }
    }
}

fn take_nullable<T: NativePType, I: UnsignedPType>(
    pvector: &PVector<T>,
    indices: &PVector<I>,
) -> PVector<T> {
    // We ignore nullability when taking the elements since we can let the `Mask` implementation
    // determine which elements are null.
    let taken_elements = pvector.elements().take(indices.elements().as_slice());
    let taken_validity = pvector.validity().take(indices);

    debug_assert_eq!(taken_elements.len(), taken_validity.len());

    // SAFETY: We used the same indices to take from both components, so they should still have the
    // same length.
    unsafe { PVector::new_unchecked(taken_elements, taken_validity) }
}
