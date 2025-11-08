// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::ptr;

use vortex_buffer::BufferMut;
use vortex_mask::Mask;

use crate::filter::Filter;

impl<T: Copy> Filter for &mut BufferMut<T> {
    type Output = ();

    fn filter(self, selection_mask: &Mask) {
        assert_eq!(
            selection_mask.len(),
            self.len(),
            "Selection mask length must equal the buffer length"
        );

        match selection_mask {
            Mask::AllTrue(_) => {}
            Mask::AllFalse(_) => self.clear(),
            Mask::Values(values) => {
                // We choose to _always_ use slices here because iterating over indices will have
                // strictly more loop iterations than slices, and the overhead over batched
                // `ptr::copy(len)` is not worth it.
                let slices = values.slices();

                // SAFETY: We checked above that the selection mask has the same length as the
                // buffer.
                let new_len = unsafe { filter_slices_in_place(self.as_mut_slice(), slices) };

                debug_assert!(
                    new_len <= self.len(),
                    "The new length was somehow larger after filter"
                );

                // Truncate the buffer to the new length.
                // SAFETY: The new length cannot be larger than the old length, so all values must
                // be initialized.
                unsafe { self.set_len(new_len) };
            }
        }
    }
}

/// Filters a buffer in-place using slice ranges to determine which values to keep.
///
/// Returns the new length of the buffer.
///
/// # Safety
///
/// The slice ranges must be in the range of the `buffer`.
#[must_use = "The caller should set the new length of the buffer"]
unsafe fn filter_slices_in_place<T: Copy>(buffer: &mut [T], slices: &[(usize, usize)]) -> usize {
    let mut write_pos = 0;

    // For each range in the selection, copy all of the elements to the current write position.
    for &(start, end) in slices {
        // Note that we could add an if statement here that checks `if read_idx != write_idx`, but
        // it's probably better to just avoid the branch misprediction.

        let len = end - start;

        // SAFETY: The safety contract enforces that all ranges are within bounds.
        unsafe {
            ptr::copy(
                buffer.as_ptr().add(start),
                buffer.as_mut_ptr().add(write_pos),
                len,
            )
        };

        write_pos += len;
    }

    write_pos
}

#[cfg(test)]
mod tests {
    use vortex_buffer::{BufferMut, buffer_mut};
    use vortex_mask::Mask;

    use super::*;

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
    #[should_panic(expected = "Selection mask length must equal the buffer length")]
    fn test_filter_length_mismatch() {
        let mut buf = buffer_mut![1u32, 2, 3];
        let mask = Mask::new_true(5); // Wrong length.

        buf.filter(&mask);
    }
}
