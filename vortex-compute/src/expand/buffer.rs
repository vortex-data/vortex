// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{Buffer, BufferMut};
use vortex_mask::Mask;

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
            Mask::Values(_) => {
                // Try to get exclusive access to expand in-place.
                match self.try_into_mut() {
                    Ok(mut buf_mut) => {
                        (&mut buf_mut).expand(mask);
                        buf_mut.freeze()
                    }
                    // Otherwise, expand into a new buffer.
                    Err(buffer) => expand_into_new_buffer(buffer.as_slice(), mask),
                }
            }
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
            // Expand into new buffer unconditionally as `try_into_mut` can never succeed on `&Buffer`.
            Mask::Values(_) => expand_into_new_buffer(self.as_slice(), mask),
        }
    }
}

impl<T: Copy> Expand for &mut BufferMut<T> {
    type Output = ();

    fn expand(self, mask: &Mask) {
        assert_eq!(
            mask.true_count(),
            self.len(),
            "Expand mask true count must equal the buffer length"
        );

        match mask {
            Mask::AllTrue(_) => {}
            Mask::AllFalse(_) => self.clear(),
            Mask::Values(mask_values) => {
                let buf_len = self.len();
                let mask_len = mask_values.len();

                if buf_len == 0 {
                    return;
                }

                // Expand to the new buffer size which equals the length of the mask.
                self.reserve(mask_len - buf_len);

                // SAFETY: We just reserved enough space above.
                unsafe {
                    self.set_len(mask_len);
                }

                let buf_slice = self.as_mut_slice();
                expand_into_slice_inplace(buf_slice, buf_len, mask_values);
            }
        }
    }
}

/// Scatters elements from a mutable slice into itself at positions marked true in the mask.
/// Used for in-place expansion where source and destination are the same buffer.
///
/// # Arguments
///
/// * `buf_slice` - The buffer slice to scatter into (already expanded to mask length)
/// * `src_len` - The original length of the buffer before expansion
/// * `mask_values` - The mask indicating where elements should be placed
fn expand_into_slice_inplace<T: Copy>(
    buf_slice: &mut [T],
    src_len: usize,
    mask_values: &vortex_mask::MaskValues,
) {
    let mask_len = buf_slice.len();

    // Pick the first value as a default value. The buffer is not empty, and we
    // know that the first value is guaranteed to be initialized. By doing this
    // T does not require to implement `Default`.
    let pseudo_default_value = buf_slice[0];

    let mut element_idx = src_len;

    // Iterate backwards through the mask to avoid overwriting unprocessed elements.
    for mask_idx in (src_len..mask_len).rev() {
        if mask_values.value(mask_idx) {
            element_idx -= 1;
            buf_slice[mask_idx] = buf_slice[element_idx];
        } else {
            // Initialize with a pseudo-default value.
            buf_slice[mask_idx] = pseudo_default_value;
        }
    }

    for mask_idx in (0..src_len).rev() {
        if mask_values.value(mask_idx) {
            element_idx -= 1;
            buf_slice[mask_idx] = buf_slice[element_idx];
        }
        // For the range up to buffer length, all positions are already initialized.
    }
}

/// Scatters elements from a source buffer into a destination slice at positions marked true
/// in the mask.
///
/// # Arguments
///
/// * `src` - The source elements to scatter
/// * `dest` - The destination buffer slice (already expanded to mask length)
/// * `mask_values` - The mask indicating where elements should be placed
fn scatter_into_slice_from<T: Copy>(
    src: &[T],
    dest: &mut [T],
    mask_values: &vortex_mask::MaskValues,
) {
    let mask_len = dest.len();

    // Pick the first value as a default value. The source buffer is not empty.
    let pseudo_default_value = src[0];

    let src_len = src.len();
    let mut element_idx = src_len;

    // Iterate backwards through the mask to avoid any issues.
    for mask_idx in (src_len..mask_len).rev() {
        if mask_values.value(mask_idx) {
            element_idx -= 1;
            dest[mask_idx] = src[element_idx];
        } else {
            // Initialize with a pseudo-default value.
            dest[mask_idx] = pseudo_default_value;
        }
    }

    for mask_idx in (0..src_len).rev() {
        if mask_values.value(mask_idx) {
            element_idx -= 1;
            dest[mask_idx] = src[element_idx];
        }
    }
}

/// Expands a slice into a new buffer at the target size, scattering elements to
/// true positions in the mask.
///
/// # Arguments
///
/// * `src` - The source slice containing elements to scatter
/// * `mask` - The mask indicating where elements should be placed
///
/// # Returns
///
/// A new `Buffer<T>` with length equal to `mask.len()`, with elements from `src` scattered
/// to positions marked true in the mask. Positions marked false can have arbitrary values.
fn expand_into_new_buffer<T: Copy>(src: &[T], mask: &Mask) -> Buffer<T> {
    let src_len = src.len();
    let mask_len = mask.len();

    match mask {
        Mask::AllTrue(_) => Buffer::from_trusted_len_iter(src.iter().copied()),
        Mask::AllFalse(_) => Buffer::empty(),
        Mask::Values(mask_values) => {
            if src_len == 0 {
                return Buffer::empty();
            }

            let mut buf_mut = BufferMut::<T>::with_capacity(mask_len);

            // SAFETY: We're preallocating the full target capacity.
            unsafe {
                buf_mut.set_len(mask_len);
            }

            let buf_slice = buf_mut.as_mut_slice();
            scatter_into_slice_from(src, buf_slice, mask_values);
            buf_mut.freeze()
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::{buffer, buffer_mut};
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

    // Tests for &Buffer<T> impl
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

    // Tests for &mut BufferMut<T> impl
    #[test]
    fn test_expand_mut_scattered() {
        let mut buf = buffer_mut![100u32, 200, 300];
        let mask = Mask::from_iter([true, false, true, false, true]);

        (&mut buf).expand(&mask);
        assert_eq!(buf.len(), 5);
        assert_eq!(buf.as_slice()[0], 100);
        assert_eq!(buf.as_slice()[2], 200);
        assert_eq!(buf.as_slice()[4], 300);
    }

    #[test]
    fn test_expand_mut_all_true() {
        let mut buf = buffer_mut![10u32, 20, 30];
        let mask = Mask::new_true(3);

        (&mut buf).expand(&mask);
        assert_eq!(buf.as_slice(), &[10, 20, 30]);
    }

    #[test]
    fn test_expand_mut_all_false() {
        let mut buf: BufferMut<u32> = BufferMut::with_capacity(0);
        let mask = Mask::new_false(0);

        (&mut buf).expand(&mask);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_expand_mut_contiguous_start() {
        let mut buf = buffer_mut![10u32, 20, 30, 40];
        let mask = Mask::from_iter([true, true, true, true, false, false, false]);

        (&mut buf).expand(&mask);
        assert_eq!(buf.len(), 7);
        assert_eq!(buf.as_slice()[0..4], [10u32, 20, 30, 40]);
    }

    #[test]
    fn test_expand_mut_contiguous_end() {
        let mut buf = buffer_mut![100u32, 200, 300];
        let mask = Mask::from_iter([false, false, false, false, true, true, true]);

        (&mut buf).expand(&mask);
        assert_eq!(buf.len(), 7);
        assert_eq!(buf.as_slice()[4..7], [100u32, 200, 300]);
    }

    #[test]
    fn test_expand_mut_dense() {
        let mut buf = buffer_mut![1u32, 2, 3, 4, 5];
        let mask = Mask::from_iter([
            true, false, true, true, false, true, true, false, false, false,
        ]);

        (&mut buf).expand(&mask);
        assert_eq!(buf.len(), 10);
        assert_eq!(buf.as_slice()[0], 1);
        assert_eq!(buf.as_slice()[2], 2);
        assert_eq!(buf.as_slice()[3], 3);
        assert_eq!(buf.as_slice()[5], 4);
        assert_eq!(buf.as_slice()[6], 5);
    }

    #[test]
    #[should_panic(expected = "Expand mask true count must equal the buffer length")]
    fn test_expand_mut_mismatch_true_count() {
        let mut buf = buffer_mut![10u32, 20];
        let mask = Mask::from_iter([true, true, true, false]);
        (&mut buf).expand(&mask);
    }
}
