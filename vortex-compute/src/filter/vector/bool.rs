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

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_mask::Mask;
    use vortex_vector::bool::BoolVectorMut;
    use vortex_vector::{VectorMutOps, VectorOps};

    use super::*;
    use crate::filter::MaskIndices;

    #[test]
    fn test_filter_bool_vector_with_mask() {
        let vec = BoolVectorMut::from_iter([true, false, true, false, true]).freeze();
        let mask = Mask::from_iter([true, false, true, false, true]);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.validity().true_count(), 3);
        assert_eq!(filtered.bits(), &BitBuffer::from_iter([true, true, true]));
    }

    #[test]
    fn test_filter_bool_vector_with_mask_indices() {
        let vec = BoolVectorMut::from_iter([true, false, true, false, true]).freeze();
        let indices = unsafe { MaskIndices::new_unchecked(&[0, 2, 4]) };

        let filtered = vec.filter(&indices);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.validity().true_count(), 3);
        assert_eq!(filtered.bits(), &BitBuffer::from_iter([true, true, true]));
    }

    #[test]
    fn test_filter_bool_vector_with_nulls() {
        let vec =
            BoolVectorMut::from_iter([Some(true), None, Some(false), Some(true), None]).freeze();
        let mask = Mask::from_iter([true, true, false, true, false]);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.validity().true_count(), 2);

        assert_eq!(
            filtered.validity().to_bit_buffer(),
            BitBuffer::from_iter([true, false, true])
        );
        assert_eq!(filtered.bits(), &BitBuffer::from_iter([true, false, true]));
    }

    #[test]
    fn test_filter_bool_vector_all_true() {
        let vec = BoolVectorMut::from_iter([true, false, true]).freeze();
        let mask = Mask::new_true(3);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.bits(), &BitBuffer::from_iter([true, false, true]));
    }

    #[test]
    fn test_filter_bool_vector_all_false() {
        let vec = BoolVectorMut::from_iter([true, false, true]).freeze();
        let mask = Mask::new_false(3);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_filter_bool_vector_mut_with_mask() {
        let mut vec = BoolVectorMut::from_iter([true, false, true, false, true]);
        let mask = Mask::from_iter([true, false, true, false, true]);

        vec.filter(&mask);

        assert_eq!(vec.len(), 3);
        let frozen = vec.freeze();
        assert_eq!(frozen.len(), 3);
        assert_eq!(frozen.validity().true_count(), 3);
    }

    #[test]
    fn test_filter_bool_vector_mut_with_mask_indices() {
        let mut vec = BoolVectorMut::from_iter([true, false, true, false, true]);
        let indices = unsafe { MaskIndices::new_unchecked(&[0, 2, 4]) };

        vec.filter(&indices);

        assert_eq!(vec.len(), 3);
        let frozen = vec.freeze();
        assert_eq!(frozen.len(), 3);
        assert_eq!(frozen.validity().true_count(), 3);
    }

    #[test]
    fn test_filter_bool_vector_mut_with_nulls() {
        let mut vec = BoolVectorMut::from_iter([Some(true), None, Some(false), Some(true), None]);
        let mask = Mask::from_iter([true, true, false, true, false]);

        vec.filter(&mask);

        assert_eq!(vec.len(), 3);
        let frozen = vec.freeze();
        assert_eq!(frozen.len(), 3);
        assert_eq!(frozen.validity().true_count(), 2);
    }
}
