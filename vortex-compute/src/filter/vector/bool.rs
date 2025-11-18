// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{BitBuffer, BitBufferMut};
use vortex_mask::{Mask, MaskMut};
use vortex_vector::VectorOps;
use vortex_vector::bool::{BoolVector, BoolVectorMut};

use crate::filter::Filter;

impl<M> Filter<M> for &BoolVector
where
    for<'a> &'a BitBuffer: Filter<M, Output = BitBuffer>,
    for<'a> &'a Mask: Filter<M, Output = Mask>,
{
    type Output = BoolVector;

    fn filter(self, selection: &M) -> Self::Output {
        let filtered_bits = self.bits().filter(selection);
        let filtered_validity = self.validity().filter(selection);

        // SAFETY: We filter the bits and validity with the same mask, and since they came from an
        // existing and valid `BoolVector`, we know that the filtered output must have the same
        // length.
        unsafe { BoolVector::new_unchecked(filtered_bits, filtered_validity) }
    }
}

impl<M> Filter<M> for &mut BoolVectorMut
where
    for<'a> &'a mut BitBufferMut: Filter<M, Output = ()>,
    for<'a> &'a mut MaskMut: Filter<M, Output = ()>,
{
    type Output = ();

    fn filter(self, selection: &M) -> Self::Output {
        unsafe { self.bits_mut().filter(selection) };
        unsafe { self.validity_mut().filter(selection) };
    }
}
