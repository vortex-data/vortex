// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_dtype::UnsignedPType;
use vortex_vector::Vector;
use vortex_vector::VectorOps;
use vortex_vector::primitive::PVector;
use vortex_vector::struct_::StructVector;

use crate::take::Take;

impl<I: UnsignedPType> Take<PVector<I>> for &StructVector {
    type Output = StructVector;

    fn take(self, indices: &PVector<I>) -> StructVector {
        if indices.validity().all_true() {
            self.take(indices.elements().as_slice())
        } else {
            take_nullable(self, indices)
        }
    }
}

impl<I: UnsignedPType> Take<[I]> for &StructVector {
    type Output = StructVector;

    fn take(self, indices: &[I]) -> StructVector {
        let taken_fields: Box<[Vector]> = self
            .fields()
            .iter()
            .map(|field| field.take(indices))
            .collect();
        let taken_validity = self.validity().take(indices);

        // SAFETY: We called take on all fields and validity with the same indices, so all fields
        // must have the same length as each other and as the validity.
        unsafe { StructVector::new_unchecked(Arc::new(taken_fields), taken_validity) }
    }
}

fn take_nullable<I: UnsignedPType>(svector: &StructVector, indices: &PVector<I>) -> StructVector {
    // We ignore nullability when taking the fields since we can let the `Mask` implementation
    // determine which elements are null.
    let taken_fields: Box<[Vector]> = svector
        .fields()
        .iter()
        .map(|field| field.take(indices.elements().as_slice()))
        .collect();

    // NB: This is the nullable version of `take`, so this is not the same as the `take`
    // implementation `indices: &[I]` above.
    let taken_validity = svector.validity().take(indices);

    // SAFETY: We called take on all fields and validity with the same indices, so all fields must
    // have the same length as each other and as the validity.
    unsafe { StructVector::new_unchecked(Arc::new(taken_fields), taken_validity) }
}
