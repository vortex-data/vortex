// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{Buffer, BufferMut};
use vortex_mask::{Mask, MaskIter};

use crate::filter::Filter;

// This is modeled after the constant with the equivalent name in arrow-rs.
const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

impl<T: Copy> Filter for &Buffer<T> {
    type Output = Buffer<T>;

    fn filter(self, mask: &Mask) -> Buffer<T> {
        assert_eq!(mask.len(), self.len());
        match mask {
            Mask::AllTrue(_) => self.clone(),
            Mask::AllFalse(_) => Buffer::empty(),
            Mask::Values(v) => match v.threshold_iter(FILTER_SLICES_SELECTIVITY_THRESHOLD) {
                MaskIter::Indices(indices) => filter_indices(self.as_slice(), indices),
                MaskIter::Slices(slices) => {
                    filter_slices(self.as_slice(), mask.true_count(), slices)
                }
            },
        }
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
    use vortex_buffer::buffer;
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
}
