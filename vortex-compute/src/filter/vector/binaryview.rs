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

#[cfg(test)]
mod tests {
    use vortex_mask::Mask;
    use vortex_vector::binaryview::StringVectorMut;
    use vortex_vector::{VectorMutOps, VectorOps};

    use super::*;
    use crate::filter::MaskIndices;

    #[test]
    fn test_filter_binary_view_vector_with_mask() {
        let mut vec = StringVectorMut::with_capacity(5);
        vec.append_values("hello", 1);
        vec.append_values("world", 1);
        vec.append_values("foo", 1);
        vec.append_values("bar", 1);
        vec.append_values("baz", 1);
        let vec = vec.freeze();

        let mask = Mask::from_iter([true, false, true, false, true]);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.validity().true_count(), 3);
        assert_eq!(filtered.get_ref(0), Some("hello"));
        assert_eq!(filtered.get_ref(1), Some("foo"));
        assert_eq!(filtered.get_ref(2), Some("baz"));
    }

    #[test]
    fn test_filter_binary_view_vector_with_mask_indices() {
        let mut vec = StringVectorMut::with_capacity(5);
        vec.append_values("hello", 1);
        vec.append_values("world", 1);
        vec.append_values("foo", 1);
        vec.append_values("bar", 1);
        vec.append_values("baz", 1);
        let vec = vec.freeze();

        let indices = unsafe { MaskIndices::new_unchecked(&[0, 2, 4]) };

        let filtered = vec.filter(&indices);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.validity().true_count(), 3);
        assert_eq!(filtered.get_ref(0), Some("hello"));
        assert_eq!(filtered.get_ref(1), Some("foo"));
        assert_eq!(filtered.get_ref(2), Some("baz"));
    }

    #[test]
    fn test_filter_binary_view_vector_with_nulls() {
        let mut vec = StringVectorMut::with_capacity(5);
        vec.append_values("hello", 1);
        vec.append_nulls(1);
        vec.append_values("foo", 1);
        vec.append_values("bar", 1);
        vec.append_nulls(1);
        let vec = vec.freeze();

        let mask = Mask::from_iter([true, true, false, true, false]);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.validity().true_count(), 2);
        assert_eq!(filtered.get_ref(0), Some("hello"));
        assert_eq!(filtered.get_ref(1), None);
        assert_eq!(filtered.get_ref(2), Some("bar"));
    }

    #[test]
    fn test_filter_binary_view_vector_all_true() {
        let mut vec = StringVectorMut::with_capacity(3);
        vec.append_values("hello", 1);
        vec.append_values("world", 1);
        vec.append_values("foo", 1);
        let vec = vec.freeze();

        let mask = Mask::new_true(3);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.get_ref(0), Some("hello"));
        assert_eq!(filtered.get_ref(1), Some("world"));
        assert_eq!(filtered.get_ref(2), Some("foo"));
    }

    #[test]
    fn test_filter_binary_view_vector_all_false() {
        let mut vec = StringVectorMut::with_capacity(3);
        vec.append_values("hello", 1);
        vec.append_values("world", 1);
        vec.append_values("foo", 1);
        let vec = vec.freeze();

        let mask = Mask::new_false(3);

        let filtered = vec.filter(&mask);

        assert_eq!(filtered.len(), 0);
    }

    #[test]
    fn test_filter_binary_view_vector_mut_with_mask() {
        let mut vec = StringVectorMut::with_capacity(5);
        vec.append_values("hello", 1);
        vec.append_values("world", 1);
        vec.append_values("foo", 1);
        vec.append_values("bar", 1);
        vec.append_values("baz", 1);

        let mask = Mask::from_iter([true, false, true, false, true]);

        vec.filter(&mask);

        assert_eq!(vec.len(), 3);
        let frozen = vec.freeze();
        assert_eq!(frozen.len(), 3);
        assert_eq!(frozen.validity().true_count(), 3);
        assert_eq!(frozen.get_ref(0), Some("hello"));
        assert_eq!(frozen.get_ref(1), Some("foo"));
        assert_eq!(frozen.get_ref(2), Some("baz"));
    }

    #[test]
    fn test_filter_binary_view_vector_mut_with_mask_indices() {
        let mut vec = StringVectorMut::with_capacity(5);
        vec.append_values("hello", 1);
        vec.append_values("world", 1);
        vec.append_values("foo", 1);
        vec.append_values("bar", 1);
        vec.append_values("baz", 1);

        let indices = unsafe { MaskIndices::new_unchecked(&[0, 2, 4]) };

        vec.filter(&indices);

        assert_eq!(vec.len(), 3);
        let frozen = vec.freeze();
        assert_eq!(frozen.len(), 3);
        assert_eq!(frozen.validity().true_count(), 3);
        assert_eq!(frozen.get_ref(0), Some("hello"));
        assert_eq!(frozen.get_ref(1), Some("foo"));
        assert_eq!(frozen.get_ref(2), Some("baz"));
    }

    #[test]
    fn test_filter_binary_view_vector_mut_with_nulls() {
        let mut vec = StringVectorMut::with_capacity(5);
        vec.append_values("hello", 1);
        vec.append_nulls(1);
        vec.append_values("foo", 1);
        vec.append_values("bar", 1);
        vec.append_nulls(1);

        let mask = Mask::from_iter([true, true, false, true, false]);

        vec.filter(&mask);

        assert_eq!(vec.len(), 3);
        let frozen = vec.freeze();
        assert_eq!(frozen.len(), 3);
        assert_eq!(frozen.validity().true_count(), 2);
        assert_eq!(frozen.get_ref(0), Some("hello"));
        assert_eq!(frozen.get_ref(1), None);
        assert_eq!(frozen.get_ref(2), Some("bar"));
    }
}
