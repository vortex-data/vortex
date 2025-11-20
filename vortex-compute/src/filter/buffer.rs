// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{BitView, Buffer, BufferMut};
use vortex_mask::{Mask, MaskIter};

use crate::filter::Filter;

// This is modeled after the constant with the equivalent name in arrow-rs.
const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

impl<M, T: Copy> Filter<M> for Buffer<T>
where
    for<'a> &'a Buffer<T>: Filter<M, Output = Buffer<T>>,
    for<'a> &'a mut BufferMut<T>: Filter<M, Output = ()>,
{
    type Output = Self;

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
        assert_eq!(
            selection_mask.len(),
            self.len(),
            "Selection mask length must equal the buffer length"
        );

        match selection_mask {
            Mask::AllTrue(_) => self.clone(),
            Mask::AllFalse(_) => Buffer::empty(),
            Mask::Values(v) => match v.threshold_iter(FILTER_SLICES_SELECTIVITY_THRESHOLD) {
                MaskIter::Indices(indices) => filter_indices(self.as_slice(), indices),
                MaskIter::Slices(slices) => {
                    filter_slices(self.as_slice(), selection_mask.true_count(), slices)
                }
            },
        }
    }
}

impl<const NB: usize, T: Copy> Filter<BitView<'_, NB>> for &Buffer<T> {
    type Output = Buffer<T>;

    fn filter(self, selection: &BitView<'_, NB>) -> Self::Output {
        self.as_slice().filter(selection)
    }
}

impl<M, T> Filter<M> for &mut BufferMut<T>
where
    for<'a> &'a mut [T]: Filter<M, Output = &'a mut [T]>,
{
    type Output = ();

    fn filter(self, selection_mask: &M) -> Self::Output {
        let true_count = self.as_mut_slice().filter(selection_mask).len();
        self.truncate(true_count);
    }
}

fn filter_indices<T: Copy>(values: &[T], indices: &[usize]) -> Buffer<T> {
    Buffer::<T>::from_trusted_len_iter(indices.iter().map(|&idx| values[idx]))
}

fn filter_slices<T>(values: &[T], output_len: usize, slices: &[(usize, usize)]) -> Buffer<T> {
    let mut out = BufferMut::<T>::with_capacity(output_len);
    for (start, end) in slices {
        out.extend_from_slice(&values[*start..*end]);
    }
    out.freeze()
}

#[cfg(test)]
mod tests {
    use vortex_buffer::{BufferMut, buffer, buffer_mut};
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
        let result = filter_indices(buf.as_slice(), &[0, 2, 3]);
        assert_eq!(result, buffer![100u32, 300, 400]);
    }

    #[test]
    fn test_filter_slices_direct() {
        let buf = buffer![1u32, 2, 3, 4, 5];
        let result = filter_slices(buf.as_slice(), 3, &[(0, 2), (4, 5)]);
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

    #[test]
    #[should_panic(expected = "Mask length must equal the slice length")]
    fn test_filter_length_mismatch() {
        let mut buf = buffer_mut![1u32, 2, 3];
        let mask = Mask::new_true(5); // Wrong length.

        buf.filter(&mask);
    }
}
