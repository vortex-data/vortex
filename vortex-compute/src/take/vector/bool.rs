// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::UnsignedPType;
use vortex_vector::VectorOps;
use vortex_vector::bool::BoolVector;
use vortex_vector::primitive::PVector;

use crate::take::Take;

impl<I: UnsignedPType> Take<PVector<I>> for &BoolVector {
    type Output = BoolVector;

    fn take(self, indices: &PVector<I>) -> BoolVector {
        if indices.validity().all_true() {
            self.take(indices.elements().as_slice())
        } else {
            take_nullable(self, indices)
        }
    }
}

impl<I: UnsignedPType> Take<[I]> for &BoolVector {
    type Output = BoolVector;

    fn take(self, indices: &[I]) -> BoolVector {
        let taken_bits = self.bits().take(indices);
        let taken_validity = self.validity().take(indices);

        debug_assert_eq!(taken_bits.len(), taken_validity.len());

        // SAFETY: We called take on both components of the vector with the same indices, so the new
        // components must have the same length.
        unsafe { BoolVector::new_unchecked(taken_bits, taken_validity) }
    }
}

fn take_nullable<I: UnsignedPType>(bvector: &BoolVector, indices: &PVector<I>) -> BoolVector {
    // We ignore nullability when taking the bits since we can let the `Mask` implementation
    // determine which elements are null.
    let taken_bits = bvector.bits().take(indices.elements().as_slice());
    let taken_validity = bvector.validity().take(indices);

    debug_assert_eq!(taken_bits.len(), taken_validity.len());

    // SAFETY: We used the same indices to take from both components, so they should still have the
    // same length.
    unsafe { BoolVector::new_unchecked(taken_bits, taken_validity) }
}
