// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{Buffer, BufferMut};
use vortex_mask::{Mask, MaskMut};
use vortex_vector::VectorOps;
use vortex_vector::binaryview::{
    BinaryView, BinaryViewType, BinaryViewVector, BinaryViewVectorMut,
};

use crate::filter::Filter;

impl<M, T: BinaryViewType> Filter<M> for &BinaryViewVector<T>
where
    for<'a> &'a Mask: Filter<M, Output = Mask>,
    for<'a> &'a Buffer<BinaryView>: Filter<M, Output = Buffer<BinaryView>>,
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
