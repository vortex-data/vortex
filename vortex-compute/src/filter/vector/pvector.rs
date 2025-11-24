// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::NativePType;
use vortex_mask::{Mask, MaskMut};
use vortex_vector::primitive::{PVector, PVectorMut};
use vortex_vector::{VectorMutOps, VectorOps};

use crate::filter::Filter;

impl<M, T: NativePType> Filter<M> for &PVector<T>
where
    for<'a> &'a Buffer<T>: Filter<M, Output = Buffer<T>>,
    for<'a> &'a Mask: Filter<M, Output = Mask>,
{
    type Output = PVector<T>;

    fn filter(self, selection_mask: &M) -> PVector<T> {
        let filtered_elements = self.elements().filter(selection_mask);
        let filtered_validity = self.validity().filter(selection_mask);

        // SAFETY: We filtered both components by the same mask, so the length invariants are
        // upheld.
        unsafe { PVector::new_unchecked(filtered_elements, filtered_validity) }
    }
}

impl<M, T: NativePType> Filter<M> for &mut PVectorMut<T>
where
    for<'a> &'a mut BufferMut<T>: Filter<M, Output = ()>,
    for<'a> &'a mut MaskMut: Filter<M, Output = ()>,
{
    type Output = ();

    fn filter(self, selection_mask: &M) {
        // SAFETY: We filter the two components of the vector at the same time, so the length
        // invariants remain true.
        unsafe {
            self.elements_mut().filter(selection_mask);
            self.validity_mut().filter(selection_mask);
        }
    }
}

impl<M, T: NativePType> Filter<M> for PVector<T>
where
    for<'a> &'a PVector<T>: Filter<M, Output = PVector<T>>,
    for<'a> &'a mut PVectorMut<T>: Filter<M, Output = ()>,
{
    type Output = Self;

    fn filter(self, selection_mask: &M) -> Self {
        match self.try_into_mut() {
            // If we have exclusive access, we can perform the filter in place.
            Ok(mut vector_mut) => {
                (&mut vector_mut).filter(selection_mask);
                vector_mut.freeze()
            }
            // Otherwise, allocate a new buffer and fill it in (delegate to the `&PVector` impl).
            Err(vector) => (&vector).filter(selection_mask),
        }
    }
}
