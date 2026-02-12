// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_mask::Mask;
use vortex_mask::MaskMut;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;
use vortex_vector::binaryview::BinaryView;
use vortex_vector::binaryview::BinaryViewType;
use vortex_vector::binaryview::BinaryViewVector;
use vortex_vector::binaryview::BinaryViewVectorMut;

use crate::filter::Filter;

impl<M, T: BinaryViewType> Filter<M> for &BinaryViewVector<T>
where
    for<'a> &'a Mask: Filter<M, Output = Mask>,
    for<'a> &'a Buffer<BinaryView>: Filter<M, Output = Buffer<BinaryView>>,
    M: ?Sized,
{
    type Output = BinaryViewVector<T>;

    fn filter(self, selection: &M) -> Self::Output {
        let views = self.views().filter(selection);
        let validity = self.validity().filter(selection);

        // SAFETY: we filter the views and validity using the same mask
        unsafe { BinaryViewVector::<T>::new_unchecked(views, self.buffers().clone(), validity) }
    }
}

impl<M, T: BinaryViewType> Filter<M> for &mut BinaryViewVectorMut<T>
where
    for<'a> &'a mut MaskMut: Filter<M, Output = ()>,
    for<'a> &'a mut BufferMut<BinaryView>: Filter<M, Output = ()>,
    M: ?Sized,
{
    type Output = ();

    fn filter(self, selection: &M) -> Self::Output {
        // SAFETY: views and validity filtered by the same mask will have
        //  same resultant length.
        unsafe {
            self.views_mut().filter(selection);
            self.validity_mut().filter(selection);
        }
    }
}

impl<M, T: BinaryViewType> Filter<M> for BinaryViewVector<T>
where
    for<'a> &'a BinaryViewVector<T>: Filter<M, Output = BinaryViewVector<T>>,
    for<'a> &'a mut BinaryViewVectorMut<T>: Filter<M, Output = ()>,
{
    type Output = Self;

    fn filter(self, selection: &M) -> Self {
        match self.try_into_mut() {
            // If we have exclusive access, we can perform the filter in place.
            Ok(mut vector_mut) => {
                (&mut vector_mut).filter(selection);
                vector_mut.freeze()
            }
            // Otherwise, allocate a new buffer and fill it in (delegate to the `&BinaryViewVector`
            // impl).
            Err(vector) => (&vector).filter(selection),
        }
    }
}
