// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{Buffer, BufferMut};
use vortex_mask::{Mask, MaskValues};

use crate::expand::Expand;

impl<T: Copy + Default> Expand for &Buffer<T> {
    type Output = Buffer<T>;

    fn expand(self, mask: &Mask) -> Self::Output {
        assert_eq!(
            mask.true_count(),
            self.len(),
            "Expand mask true count must equal the buffer length"
        );

        match mask {
            Mask::AllTrue(_) => self.clone(),
            Mask::AllFalse(_) => self.clone(),
            Mask::Values(mask_values) => expand(self.as_slice(), mask_values),
        }
    }
}

impl<T: Copy + Default> Expand for BufferMut<T> {
    type Output = BufferMut<T>;

    fn expand(self, mask: &Mask) -> Self::Output {
        assert_eq!(
            mask.true_count(),
            self.len(),
            "Expand mask true count must equal the buffer length"
        );

        match mask {
            Mask::AllTrue(_) => self,
            Mask::AllFalse(_) => self,
            Mask::Values(mask_values) => expand(self.as_slice(), mask_values).into_mut(),
        }
    }
}

/// Expands a slice into a new buffer at the target size, scattering elements to
/// true positions in the mask.
///
/// # Arguments
///
/// * `src` - The source slice containing elements to scatter
/// * `mask_values` - The mask indicating where elements should be placed
///
/// # Returns
///
/// A new `Buffer<T>` with length equal to `mask.len`, with elements from `src` scattered
/// to positions marked true in the mask. Positions marked false are set to `T::default`.
fn expand<T: Copy + Default>(src: &[T], mask_values: &MaskValues) -> Buffer<T> {
    let mask_len = mask_values.len();

    let mut target_buf = BufferMut::<T>::with_capacity(mask_len);
    let target_slice = target_buf.spare_capacity_mut();

    let mut element_idx = 0;

    let bit_buffer = mask_values.bit_buffer();

    bit_buffer.iter_bits(|mask_idx, is_valid| {
        if is_valid {
            unsafe {
                target_slice
                    .get_unchecked_mut(mask_idx)
                    .write(src[element_idx])
            };
            element_idx += 1;
        } else {
            unsafe {
                target_slice.get_unchecked_mut(mask_idx).write(T::default());
            };
        }
    });

    // SAFETY: Buffer has sufficient capacity and all elements have been initialized.
    unsafe { target_buf.set_len(mask_len) };

    target_buf.freeze()
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use super::*;

    // Tests for Buffer<T>
    #[test]
    fn test_expand_scattered() {
        let buf = buffer![100u32, 200, 300];
        // Mask with scattered true values: [T, F, T, F, T]
        let mask = Mask::from_iter([true, false, true, false, true]);

        let result = buf.expand(&mask);
        assert_eq!(result.len(), 5);
        assert_eq!(result.as_slice()[0], 100);
        assert_eq!(result.as_slice()[2], 200);
        assert_eq!(result.as_slice()[4], 300);
    }

    #[test]
    fn test_expand_all_true() {
        let buf = buffer![10u32, 20, 30];
        let mask = Mask::new_true(3);

        let result = buf.expand(&mask);
        assert_eq!(result, buffer![10u32, 20, 30]);
    }

    #[test]
    fn test_expand_all_false() {
        let buf: Buffer<u32> = Buffer::empty();
        let mask = Mask::new_false(0);

        let result = buf.expand(&mask);
        assert!(result.is_empty());
    }

    #[test]
    fn test_expand_contiguous_start() {
        let buf = buffer![10u32, 20, 30, 40];
        // Mask with true values at start: [T, T, T, T, F, F, F]
        let mask = Mask::from_iter([true, true, true, true, false, false, false]);

        let result = buf.expand(&mask);
        assert_eq!(result.len(), 7);
        assert_eq!(result.as_slice()[0..4], [10u32, 20, 30, 40]);
    }

    #[test]
    fn test_expand_contiguous_end() {
        let buf = buffer![100u32, 200, 300];
        // Mask with true values at end: [F, F, F, F, T, T, T]
        let mask = Mask::from_iter([false, false, false, false, true, true, true]);

        let result = buf.expand(&mask);
        assert_eq!(result.len(), 7);
        assert_eq!(result.as_slice()[4..7], [100u32, 200, 300]);
    }

    #[test]
    #[should_panic(expected = "Expand mask true count must equal the buffer length")]
    fn test_expand_mismatch_true_count() {
        let buf = buffer![10u32, 20];
        // Mask has 3 true values but buffer has only 2 elements
        let mask = Mask::from_iter([true, true, true, false]);
        buf.expand(&mask);
    }

    #[test]
    fn test_expand_ref_scattered() {
        let buf = buffer![100u32, 200, 300];
        let mask = Mask::from_iter([true, false, true, false, true]);

        let result = (&buf).expand(&mask);
        assert_eq!(result.len(), 5);
        assert_eq!(result.as_slice()[0], 100);
        assert_eq!(result.as_slice()[2], 200);
        assert_eq!(result.as_slice()[4], 300);
    }

    #[test]
    fn test_expand_ref_all_true() {
        let buf = buffer![10u32, 20, 30];
        let mask = Mask::new_true(3);

        let result = (&buf).expand(&mask);
        assert_eq!(result, buffer![10u32, 20, 30]);
    }

    #[test]
    fn test_expand_copy() {
        let src = [10u32, 20, 30, 40, 50];
        // Alternating pattern with gaps: [T, F, T, F, T, F, T, F, T, F]
        let mask = Mask::from_iter([
            true, false, true, false, true, false, true, false, true, false,
        ]);

        let Mask::Values(mask_values) = mask else {
            panic!("Expected Mask::Values");
        };

        let result = expand(&src, &mask_values);
        assert_eq!(result.len(), 10);
        assert_eq!(result.as_slice()[0], 10);
        assert_eq!(result.as_slice()[2], 20);
        assert_eq!(result.as_slice()[4], 30);
        assert_eq!(result.as_slice()[6], 40);
        assert_eq!(result.as_slice()[8], 50);
    }
}
