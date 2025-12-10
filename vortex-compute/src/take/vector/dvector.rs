// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::NativeDecimalType;
use vortex_dtype::UnsignedPType;
use vortex_vector::VectorOps;
use vortex_vector::decimal::DVector;
use vortex_vector::primitive::PVector;

use crate::take::Take;

impl<D: NativeDecimalType, I: UnsignedPType> Take<PVector<I>> for &DVector<D> {
    type Output = DVector<D>;

    fn take(self, indices: &PVector<I>) -> DVector<D> {
        if indices.validity().all_true() {
            self.take(indices.elements().as_slice())
        } else {
            take_nullable(self, indices)
        }
    }
}

impl<D: NativeDecimalType, I: UnsignedPType> Take<[I]> for &DVector<D> {
    type Output = DVector<D>;

    fn take(self, indices: &[I]) -> DVector<D> {
        let taken_elements = self.elements().take(indices);
        let taken_validity = self.validity().take(indices);

        debug_assert_eq!(taken_elements.len(), taken_validity.len());

        // SAFETY: We called take on both components of the vector with the same indices, so the new
        // components must have the same length. The elements are unchanged, so they must still be
        // within the precision/scale bounds.
        unsafe { DVector::new_unchecked(self.precision_scale(), taken_elements, taken_validity) }
    }
}

fn take_nullable<D: NativeDecimalType, I: UnsignedPType>(
    dvector: &DVector<D>,
    indices: &PVector<I>,
) -> DVector<D> {
    // We ignore nullability when taking the elements since we can let the `Mask` implementation
    // determine which elements are null.
    let taken_elements = dvector.elements().take(indices.elements().as_slice());
    let taken_validity = dvector.validity().take(indices);

    debug_assert_eq!(taken_elements.len(), taken_validity.len());

    // SAFETY: We used the same indices to take from both components, so they should still have the
    // same length. The elements are unchanged, so they must still be within the precision/scale
    // bounds.
    unsafe { DVector::new_unchecked(dvector.precision_scale(), taken_elements, taken_validity) }
}
