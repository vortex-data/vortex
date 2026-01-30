// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitView;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::filter::Filter;

impl<M, T: Copy> Filter<M> for Buffer<T>
where
    for<'a> &'a Buffer<T>: Filter<M, Output = Buffer<T>>,
    for<'a> &'a mut BufferMut<T>: Filter<M, Output = ()>,
{
    type Output = Self;

    /// Filters a `Buffer` according to some selection mask.
    ///
    /// This will attempt to filter in-place if possible.
    fn filter(self, selection_mask: &M) -> Self {
        // If we have exclusive access, we can perform the filter in place.
        match self.try_into_mut() {
            Ok(mut buffer_mut) => {
                (&mut buffer_mut).filter(selection_mask);
                buffer_mut.freeze()
            }
            // Otherwise, allocate a new buffer and fill it in (delegate to the `&Buffer` impl).
            Err(buffer) => (&buffer).filter(selection_mask),
        }
    }
}

impl<T: Copy> Filter<Mask> for &Buffer<T> {
    type Output = Buffer<T>;

    fn filter(self, selection_mask: &Mask) -> Buffer<T> {
        // We delegate checking that the mask length is equal to self to the `MaskValues`
        // filter implementation below.

        match selection_mask {
            Mask::AllTrue(_) => self.clone(),
            Mask::AllFalse(_) => Buffer::empty(),
            Mask::Values(v) => self.filter(v.as_ref()),
        }
    }
}

impl<T: Copy> Filter<MaskValues> for &Buffer<T> {
    type Output = Buffer<T>;

    fn filter(self, mask_values: &MaskValues) -> Buffer<T> {
        // Delegates to the filter implementation over slices.
        self.as_slice().filter(mask_values)
    }
}

impl<T: Copy> Filter<[usize]> for &Buffer<T> {
    type Output = Buffer<T>;

    /// Filters by indices.
    ///
    /// The caller should ensure that the indices are strictly increasing, otherwise the resulting
    /// buffer might have strange values.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of bounds. With the additional constraint that the indices are
    /// strictly increasing, the length of the indices must be less than or equal to the length of
    /// `self`.
    fn filter(self, indices: &[usize]) -> Buffer<T> {
        // Delegates to the filter implementation over slices.
        self.as_slice().filter(indices)
    }
}

impl<T: Copy> Filter<[(usize, usize)]> for &Buffer<T> {
    type Output = Buffer<T>;

    /// Filters by ranges of indices.
    ///
    /// The caller should ensure that the ranges are strictly increasing, otherwise the resulting
    /// buffer might have strange values.
    ///
    /// # Panics
    ///
    /// Panics if any range is out of bounds. With the additional constraint that the ranges are
    /// strictly increasing, the length of the `slices` array must be less than or equal to the
    /// length of `self`.
    fn filter(self, slices: &[(usize, usize)]) -> Buffer<T> {
        // Delegates to the filter implementation over slices.
        self.as_slice().filter(slices)
    }
}

impl<const NB: usize, T: Copy> Filter<BitView<'_, NB>> for &Buffer<T> {
    type Output = Buffer<T>;

    fn filter(self, selection: &BitView<'_, NB>) -> Self::Output {
        // Delegates to the filter implementation over slices.
        self.as_slice().filter(selection)
    }
}

impl<M, T> Filter<M> for &mut BufferMut<T>
where
    for<'a> &'a mut [T]: Filter<M, Output = &'a mut [T]>,
{
    type Output = ();

    fn filter(self, selection_mask: &M) -> Self::Output {
        // Delegates to the filter implementation over slices, as that is also an in-place
        // operation.
        let true_count = self.as_mut_slice().filter(selection_mask).len();
        self.truncate(true_count);
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BufferMut;
    use vortex_buffer::buffer;
    use vortex_buffer::buffer_mut;
    use vortex_mask::Mask;

    use super::*;

    #[test]
    fn test_filter_buffer_by_indices() {
        let buf = buffer![10u32, 20, 30, 40, 50];
        let mask = Mask::from_iter([true, false, true, false, true]);

        let result = buf.filter(&mask);
        assert_eq!(result, buffer![10u32, 30, 50]);
    }

    #[test]
    fn test_filter_buffer_all_true() {
        let buf = buffer![1u64, 2, 3];
        let mask = Mask::new_true(3);

        let result = buf.filter(&mask);
        assert_eq!(result, buffer![1u64, 2, 3]);
    }

    #[test]
    fn test_filter_buffer_all_false() {
        let buf = buffer![1i32, 2, 3, 4];
        let mask = Mask::new_false(4);

        let result = buf.filter(&mask);
        assert!(result.is_empty());
    }

    #[test]
    fn test_filter_indices_direct() {
        let buf = buffer![100u32, 200, 300, 400];
        let result = (&buf).filter([0usize, 2, 3].as_slice());
        assert_eq!(result, buffer![100u32, 300, 400]);
    }

    #[test]
    fn test_filter_slices_direct() {
        let buf = buffer![1u32, 2, 3, 4, 5];
        let result = (&buf).filter([(0usize, 2), (4, 5)].as_slice());
        assert_eq!(result, buffer![1u32, 2, 5]);
    }

    #[test]
    fn test_filter_all_true() {
        let mut buf = buffer_mut![1u32, 2, 3, 4, 5];
        let mask = Mask::new_true(5);

        buf.filter(&mask);
        assert_eq!(buf.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_filter_all_false() {
        let mut buf = buffer_mut![1u32, 2, 3, 4, 5];
        let mask = Mask::new_false(5);

        buf.filter(&mask);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_filter_sparse() {
        let mut buf = buffer_mut![10u32, 20, 30, 40, 50];
        // Select indices 0, 2, 4 (sparse selection).
        let mask = Mask::from_iter([true, false, true, false, true]);

        buf.filter(&mask);
        assert_eq!(buf.as_slice(), &[10, 30, 50]);
    }

    #[test]
    fn test_filter_dense() {
        let mut buf = buffer_mut![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        // Dense selection (80% selected).
        let mask = Mask::from_iter([true, true, true, true, false, true, true, true, false, true]);

        buf.filter(&mask);
        assert_eq!(buf.as_slice(), &[1, 2, 3, 4, 6, 7, 8, 10]);
    }

    #[test]
    fn test_filter_single_element_kept() {
        let mut buf = buffer_mut![1u32, 2, 3, 4, 5];
        let mask = Mask::from_iter([false, false, true, false, false]);

        buf.filter(&mask);
        assert_eq!(buf.as_slice(), &[3]);
    }

    #[test]
    fn test_filter_first_last() {
        let mut buf = buffer_mut![1u32, 2, 3, 4, 5];
        let mask = Mask::from_iter([true, false, false, false, true]);

        buf.filter(&mask);
        assert_eq!(buf.as_slice(), &[1, 5]);
    }

    #[test]
    fn test_filter_alternating() {
        let mut buf = buffer_mut![1u32, 2, 3, 4, 5, 6];
        let mask = Mask::from_iter([true, false, true, false, true, false]);

        buf.filter(&mask);
        assert_eq!(buf.as_slice(), &[1, 3, 5]);
    }

    #[test]
    fn test_filter_empty_buffer() {
        let mut buf: BufferMut<u32> = BufferMut::with_capacity(0);
        let mask = Mask::new_false(0);

        buf.filter(&mask);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_filter_contiguous_regions() {
        let mut buf = buffer_mut![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        // Two contiguous regions: [0..3] and [7..10].
        let mask = Mask::from_iter([
            true, true, true, false, false, false, false, true, true, true,
        ]);

        buf.filter(&mask);
        assert_eq!(buf.as_slice(), &[1, 2, 3, 8, 9, 10]);
    }

    #[test]
    fn test_filter_large_buffer() {
        let mut buf: BufferMut<u32> = BufferMut::from_iter(0..1000);
        // Keep every third element.
        let mask = Mask::from_iter((0..1000).map(|i| i % 3 == 0));

        buf.filter(&mask);
        let expected: Vec<u32> = (0..1000).filter(|i| i % 3 == 0).collect();
        assert_eq!(buf.as_slice(), &expected[..]);
    }
}
