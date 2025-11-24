// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_mask::{Mask, MaskMut};
use vortex_vector::struct_::{StructVector, StructVectorMut};
use vortex_vector::{Vector, VectorMut, VectorMutOps, VectorOps};

use crate::filter::Filter;

impl<M> Filter<M> for &StructVector
where
    for<'a> &'a Mask: Filter<M, Output = Mask>,
    for<'a> &'a Vector: Filter<M, Output = Vector>,
{
    type Output = StructVector;

    fn filter(self, selection: &M) -> Self::Output {
        let fields: Vec<Vector> = self
            .fields()
            .iter()
            .map(|field| Filter::filter(field, selection))
            .collect();

        let fields = Arc::new(fields.into_boxed_slice());
        let validity = self.validity().filter(selection);

        // SAFETY: all field vectors and validity are filtered with same mask
        unsafe { StructVector::new_unchecked(fields, validity) }
    }
}

impl<M> Filter<M> for &mut StructVectorMut
where
    for<'a> &'a mut MaskMut: Filter<M, Output = ()>,
    for<'a> &'a mut VectorMut: Filter<M, Output = ()>,
{
    type Output = ();

    fn filter(self, selection: &M) -> Self::Output {
        // SAFETY: all field vectors and selection vector are filtered with same mask
        unsafe {
            for field in self.fields_mut() {
                field.filter(selection);
            }

            self.validity_mut().filter(selection);
        }
    }
}

impl<M> Filter<M> for StructVector
where
    for<'a> &'a StructVector: Filter<M, Output = StructVector>,
    for<'a> &'a mut StructVectorMut: Filter<M, Output = ()>,
{
    type Output = Self;

    fn filter(self, selection: &M) -> Self {
        match self.try_into_mut() {
            // If we have exclusive access, we can perform the filter in place.
            Ok(mut vector_mut) => {
                (&mut vector_mut).filter(selection);
                vector_mut.freeze()
            }
            // Otherwise, allocate a new buffer and fill it in (delegate to the `&StructVector` impl).
            Err(vector) => (&vector).filter(selection),
        }
    }
}
