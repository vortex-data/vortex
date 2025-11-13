// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::NativePType;
use vortex_mask::Mask;
use vortex_vector::primitive::{PVector, PVectorMut};
use vortex_vector::{VectorMutOps, VectorOps};

use crate::filter::Filter;

impl<T: NativePType> Filter<Mask> for &PVector<T> {
    type Output = PVector<T>;

    fn filter(self, selection_mask: &Mask) -> PVector<T> {
        assert_eq!(
            selection_mask.len(),
            self.len(),
            "Selection mask length must equal the vector length"
        );

        let filtered_elements = self.elements().filter(selection_mask);
        let filtered_validity = self.validity().filter(selection_mask);

        // SAFETY: We filtered both components by the same mask, so the length invariants are
        // upheld.
        unsafe { PVector::new_unchecked(filtered_elements, filtered_validity) }
    }
}

impl<T: NativePType> Filter<Mask> for &mut PVectorMut<T> {
    type Output = ();

    fn filter(self, selection_mask: &Mask) {
        assert_eq!(
            selection_mask.len(),
            self.len(),
            "Selection mask length must equal the vector length"
        );

        // SAFETY: We filter the two components of the vector at the same time, so the length
        // invariants remain true.
        unsafe {
            self.elements_mut().filter(selection_mask);
            self.validity_mut().filter(selection_mask);
        }
    }
}

impl<T: NativePType> Filter<Mask> for PVector<T> {
    type Output = Self;

    fn filter(self, selection_mask: &Mask) -> Self {
        assert_eq!(
            selection_mask.len(),
            self.len(),
            "Selection mask length must equal the buffer length"
        );

        // If we have exclusive access, we can perform the filter in place.
        match self.try_into_mut() {
            Ok(mut vector_mut) => {
                (&mut vector_mut).filter(selection_mask);
                vector_mut.freeze()
            }
            // Otherwise, allocate a new buffer and fill it in (delegate to the `&PVector` impl).
            Err(vector) => (&vector).filter(selection_mask),
        }
    }
}
