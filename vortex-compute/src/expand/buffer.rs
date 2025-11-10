// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_mask::{Mask, MaskValues};

use crate::expand::Expand;

impl<T: Copy> Expand for Buffer<T> {
    type Output = Buffer<T>;

    fn expand(self, mask: &Mask) -> Self::Output {
        assert_eq!(
            mask.true_count(),
            self.len(),
            "Expand mask true count must equal the buffer length"
        );

        match mask {
            Mask::AllTrue(_) => self,
            Mask::AllFalse(_) => Buffer::empty(),
            Mask::Values(mask_values) => expand_indices(self, mask_values),
        }
    }
}

impl<T: Copy> Expand for &Buffer<T> {
    type Output = Buffer<T>;

    fn expand(self, mask: &Mask) -> Self::Output {
        assert_eq!(
            mask.true_count(),
            self.len(),
            "Expand mask true count must equal the buffer length"
        );

        match mask {
            Mask::AllTrue(_) => self.clone(),
            Mask::AllFalse(_) => Buffer::empty(),
            Mask::Values(mask_values) => expand_indices(self.clone(), mask_values),
        }
    }
}

/// Expands a buffer by placing its elements at positions marked as `true` in the mask.
///
/// # Arguments
///
/// * `buf` - The buffer containing elements to scatter
/// * `mask_values` - The mask indicating where elements should be placed
///
/// # Panics
///
/// Panics if the number of `true` values in the mask does not equal the buffer length.
fn expand_indices<T: Copy>(buf: Buffer<T>, mask_values: &MaskValues) -> Buffer<T> {
    let buf_len = buf.len();

    assert_eq!(
        mask_values.true_count(),
        buf_len,
        "Mask true count must equal buffer length"
    );

    if buf.is_empty() {
        return Buffer::empty();
    }

    let mut buf_mut = buf.into_mut();
    let mask_len = mask_values.len();
    buf_mut.reserve(mask_len - buf_len);

    // Expand to the new buffer size which is equals the length of the mask.
    unsafe {
        buf_mut.set_len(mask_len);
    }

    let buf_slice = buf_mut.as_mut_slice();
    let mut element_idx = buf_len;

    // Pick the first value as a default value. The buffer is not empty, and we
    // know that the first value is guaranteed to be initialized. By doing this
    // T does does not require to implement `Default`.
    let pseudo_default_value = buf_slice[0];

    // Iterate backwards through the mask to avoid overwriting unprocessed elements.
    for mask_idx in (buf_len..mask_len).rev() {
        if mask_values.value(mask_idx) {
            element_idx -= 1;
            buf_slice[mask_idx] = buf_slice[element_idx];
        } else {
            // Initialize with a pseudo-default value.
            buf_slice[mask_idx] = pseudo_default_value;
        }
    }

    for mask_idx in (0..buf_len).rev() {
        if mask_values.value(mask_idx) {
            element_idx -= 1;
            buf_slice[mask_idx] = buf_slice[element_idx];
        }
        // For the range up to buffer length, all positions are already initialized.
    }

    buf_mut.freeze()
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use super::*;

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
}
