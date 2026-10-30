// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Buffer-level filtering by a [`MaskValues`].
//!
//! The strategies live in [`vortex_buffer::filter`], and [`MaskValues::selection`] chooses the
//! mask representation that they use. All-true, all-false, and contiguous masks are handled
//! before buffer filtering.

use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::filter;
use vortex_mask::MaskValues;

/// Filter a [`Buffer<T>`] by [`MaskValues`], returning a new buffer.
///
/// Dense uniquely owned buffers are compacted in place; other buffers allocate a new output.
pub(crate) fn filter_buffer<T: Copy>(
    buffer: Buffer<T>,
    mask: &MaskValues,
    allocator: &BufferAllocatorRef,
) -> Buffer<T> {
    filter::filter_buffer(buffer, mask.selection(), allocator)
}

/// Materialize sparse indices when enough sibling arrays will reuse the same mask.
pub(crate) fn prepare_mask_for_reuse(mask: &MaskValues, consumers: usize) {
    if consumers <= 1 || mask.cached_indices().is_some() || mask.cached_slices().is_some() {
        return;
    }

    let density_threshold = if consumers >= 3 { 0.1 } else { 0.05 };
    if mask.density() > density_threshold {
        return;
    }

    if super::contiguous_values_range(mask).is_some() {
        return;
    }

    let _ = mask.indices();
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_buffer::BufferMut;
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use super::*;

    fn filter_buffer<T: Copy>(buffer: Buffer<T>, mask: &MaskValues) -> Buffer<T> {
        super::filter_buffer(buffer, mask, BufferAllocatorRef::static_ref())
    }

    fn mask_values(mask: &Mask) -> &MaskValues {
        match mask {
            Mask::Values(v) => v,
            _ => panic!("expected Mask::Values"),
        }
    }

    #[test]
    fn test_filter_buffer() {
        let buf = buffer![10u32, 20, 30, 40, 50];
        let mask = Mask::from_iter([true, false, true, false, true]);

        let result = filter_buffer(buf, mask_values(&mask));
        assert_eq!(result, buffer![10u32, 30, 50]);
    }

    #[test]
    fn test_filter_sparse_bitmap() {
        let buf = Buffer::from(BufferMut::from_iter(0u32..1000));
        let mask = Mask::from_iter((0..1000).map(|i| i % 3 == 0));

        let result = filter_buffer(buf, mask_values(&mask));
        let expected: Vec<u32> = (0..1000).filter(|i| i % 3 == 0).collect();
        assert_eq!(result.as_slice(), &expected[..]);
    }

    #[test]
    fn test_filter_dense_in_place() {
        let buf = buffer![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let original_ptr = buf.as_ptr();
        let mask = Mask::from_iter([true, true, true, true, false, true, true, true, false, true]);

        let result = filter_buffer(buf, mask_values(&mask));
        assert_eq!(result, buffer![1u32, 2, 3, 4, 6, 7, 8, 10]);
        assert_eq!(result.as_ptr(), original_ptr);
    }

    #[test]
    fn test_filter_shared_buffer_by_cached_indices() {
        let buf = Buffer::from(BufferMut::from_iter(0u64..16));
        let shared = buf.clone();
        let mask = Mask::from_indices(16, [1, 5, 9, 15]);

        let result = filter_buffer(buf, mask_values(&mask));
        assert_eq!(result, buffer![1u64, 5, 9, 15]);
        assert_eq!(shared.as_slice(), &(0u64..16).collect::<Vec<_>>());
    }

    #[test]
    fn test_filter_shared_buffer_by_cached_slices() {
        let buf = Buffer::from(BufferMut::from_iter(0u32..32));
        let shared = buf.clone();
        let mask = Mask::from_slices(32, vec![(3, 15), (20, 30)]);

        let result = filter_buffer(buf, mask_values(&mask));
        let expected = (3u32..15).chain(20..30).collect::<Vec<_>>();
        assert_eq!(result.as_slice(), expected.as_slice());
        assert_eq!(shared.as_slice(), &(0u32..32).collect::<Vec<_>>());
    }

    #[test]
    fn test_filter_unaligned_bitmap_words() {
        const LEN: usize = 151;
        const OFFSET: usize = 5;

        let backing = BitBuffer::from_iter(
            std::iter::repeat_n(false, OFFSET).chain((0..LEN).map(|index| index % 7 == 2)),
        );
        let mask = Mask::from_buffer(BitBuffer::new_with_offset(
            backing.inner().clone(),
            LEN,
            OFFSET,
        ));
        let buf = Buffer::from(BufferMut::from_iter(0u64..LEN as u64));

        let result = filter_buffer(buf, mask_values(&mask));
        let expected = (0u64..LEN as u64)
            .filter(|value| value % 7 == 2)
            .collect::<Vec<_>>();
        assert_eq!(result.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_prepare_sparse_mask_for_sibling_reuse() {
        let two_consumers = Mask::from_iter((0..100).map(|index| index % 20 == 0));
        let values = mask_values(&two_consumers);
        assert!(values.cached_indices().is_none());
        prepare_mask_for_reuse(values, 2);
        assert!(values.cached_indices().is_some());

        let three_consumers = Mask::from_iter((0..100).map(|index| index % 10 == 0));
        let values = mask_values(&three_consumers);
        assert!(values.cached_indices().is_none());
        prepare_mask_for_reuse(values, 2);
        assert!(values.cached_indices().is_none());
        prepare_mask_for_reuse(values, 3);
        assert!(values.cached_indices().is_some());

        let contiguous = Mask::from_iter((0..100).map(|index| (20..25).contains(&index)));
        let values = mask_values(&contiguous);
        prepare_mask_for_reuse(values, 3);
        assert!(values.cached_indices().is_none());
    }
}
