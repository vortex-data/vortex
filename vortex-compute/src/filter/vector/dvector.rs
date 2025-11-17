// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::NativeDecimalType;
use vortex_mask::{Mask, MaskMut};
use vortex_vector::decimal::{DVector, DVectorMut};
use vortex_vector::{VectorMutOps, VectorOps};

use crate::filter::Filter;

impl<M, D: NativeDecimalType> Filter<M> for &DVector<D>
where
    for<'a> &'a Buffer<D>: Filter<M, Output = Buffer<D>>,
    for<'a> &'a Mask: Filter<M, Output = Mask>,
{
    type Output = DVector<D>;

    fn filter(self, selection: &M) -> Self::Output {
        let elements = self.elements().filter(selection);
        let validity = self.validity().filter(selection);
        // SAFETY: we're filtering the elements and validity with the same mask
        unsafe { DVector::<D>::new_unchecked(self.precision_scale(), elements, validity) }
    }
}

impl<M, D: NativeDecimalType> Filter<M> for &mut DVectorMut<D>
where
    for<'a> &'a mut BufferMut<D>: Filter<M, Output = ()>,
    for<'a> &'a mut MaskMut: Filter<M, Output = ()>,
{
    type Output = ();

    fn filter(self, selection: &M) -> Self::Output {
        // SAFETY: we filter elements and validity using the same mask
        unsafe {
            self.elements_mut().filter(selection);
            self.validity_mut().filter(selection);
        }
    }
}
