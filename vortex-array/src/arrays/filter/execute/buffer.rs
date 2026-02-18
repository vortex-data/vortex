// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Buffer-level filter dispatch.
//!
//! Provides [`filter_buffer`] which filters a [`Buffer<T>`] by [`MaskValues`], attempting an
//! in-place filter when the buffer has exclusive ownership.

use vortex_buffer::Buffer;
use vortex_mask::MaskValues;

use crate::arrays::filter::execute::slice;

/// Filter a [`Buffer<T>`] by [`MaskValues`], returning a new buffer.
///
/// This will attempt to filter in-place (via [`Buffer::try_into_mut`]) when the buffer has
/// exclusive ownership, avoiding an extra allocation.
pub(super) fn filter_buffer<T: Copy>(buffer: Buffer<T>, mask: &MaskValues) -> Buffer<T> {
    match buffer.try_into_mut() {
        Ok(mut buffer_mut) => {
            let new_len = slice::filter_slice_mut_by_mask_values(buffer_mut.as_mut_slice(), mask);
            buffer_mut.truncate(new_len);
            buffer_mut.freeze()
        }
        // Otherwise, allocate a new buffer and fill it in.
        Err(buffer) => slice::filter_slice_by_mask_values(buffer.as_slice(), mask),
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BufferMut;
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use super::*;

    // Helper to get `MaskValues` from a `Mask`.
    fn mask_values(mask: &Mask) -> &MaskValues {
        match mask {
            Mask::Values(v) => v.as_ref(),
            _ => panic!("expected Mask::Values"),
        }
    }

    #[test]
    fn test_filter_buffer_by_indices() {
        let buf = buffer![10u32, 20, 30, 40, 50];
        let mask = Mask::from_iter([true, false, true, false, true]);

        let result = filter_buffer(buf, mask_values(&mask));
        assert_eq!(result, buffer![10u32, 30, 50]);
    }

    #[test]
    fn test_filter_indices_direct() {
        let buf = buffer![100u32, 200, 300, 400];
        let mask = Mask::from_iter([true, false, true, true]);
        let result = filter_buffer(buf, mask_values(&mask));
        assert_eq!(result, buffer![100u32, 300, 400]);
    }

    #[test]
    fn test_filter_sparse() {
        let buf = Buffer::from(BufferMut::from_iter(0u32..1000));
        // Keep every third element.
        let mask = Mask::from_iter((0..1000).map(|i| i % 3 == 0));

        let result = filter_buffer(buf, mask_values(&mask));
        let expected: Vec<u32> = (0..1000).filter(|i| i % 3 == 0).collect();
        assert_eq!(result.as_slice(), &expected[..]);
    }

    #[test]
    fn test_filter_dense() {
        let buf = buffer![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        // Dense selection (80% selected).
        let mask = Mask::from_iter([true, true, true, true, false, true, true, true, false, true]);

        let result = filter_buffer(buf, mask_values(&mask));
        assert_eq!(result, buffer![1u32, 2, 3, 4, 6, 7, 8, 10]);
    }
}
