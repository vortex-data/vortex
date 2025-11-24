// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_mask::{Mask, MaskMut};
use vortex_vector::listview::{ListViewVector, ListViewVectorMut};
use vortex_vector::primitive::{PrimitiveVector, PrimitiveVectorMut};
use vortex_vector::{VectorMutOps, VectorOps};

use crate::filter::Filter;

impl<M> Filter<M> for &ListViewVector
where
    for<'a> &'a PrimitiveVector: Filter<M, Output = PrimitiveVector>,
    for<'a> &'a Mask: Filter<M, Output = Mask>,
{
    type Output = ListViewVector;

    fn filter(self, selection: &M) -> Self::Output {
        let offsets = self.offsets().filter(selection);
        let sizes = self.sizes().filter(selection);
        let validity = self.validity().filter(selection);

        // SAFETY: all components filtered with same mask
        unsafe {
            ListViewVector::new_unchecked(Arc::clone(self.elements()), offsets, sizes, validity)
        }
    }
}

impl<M> Filter<M> for &mut ListViewVectorMut
where
    for<'a> &'a mut PrimitiveVectorMut: Filter<M, Output = ()>,
    for<'a> &'a mut MaskMut: Filter<M, Output = ()>,
{
    type Output = ();

    fn filter(self, selection: &M) -> Self::Output {
        // SAFETY: offsets, sizes, validity all being filtered with same mask
        unsafe {
            self.offsets_mut().filter(selection);
            self.sizes_mut().filter(selection);
            self.validity_mut().filter(selection);
        }
    }
}

impl<M> Filter<M> for ListViewVector
where
    for<'a> &'a ListViewVector: Filter<M, Output = ListViewVector>,
    for<'a> &'a mut ListViewVectorMut: Filter<M, Output = ()>,
{
    type Output = Self;

    fn filter(self, selection: &M) -> Self {
        match self.try_into_mut() {
            // If we have exclusive access, we can perform the filter in place.
            Ok(mut vector_mut) => {
                (&mut vector_mut).filter(selection);
                vector_mut.freeze()
            }
            // Otherwise, allocate a new buffer and fill it in (delegate to the `&ListViewVector`
            // impl).
            Err(vector) => (&vector).filter(selection),
        }
    }
}
