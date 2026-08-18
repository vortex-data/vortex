// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Buffer-level filter dispatch across cached, SIMD, and scalar strategies.
//!
//! Selection is an ordered ladder, not a set of independent choices. The first matching row wins:
//!
//! | Priority | Condition | Out-of-place | In-place |
//! | --- | --- | --- | --- |
//! | 1 | cached slices: one run, or average run length >= 8 | copy runs | move runs |
//! | 2 | cached indices and density <= 0.50 | gather indices | move indices |
//! | 3 | an eligible architecture kernel | SIMD compress | SIMD compress |
//! | 4 | density reaches [`byte_compress_density_threshold`] | byte compress | bitmap walk |
//! | 5 | otherwise | bitmap walk | bitmap walk |
//!
//! In-place dispatch is attempted only for a uniquely owned buffer with density >= 0.50. SIMD
//! eligibility is summarized in [`simd_compress`]. All-true, all-false, and contiguous masks are
//! handled before buffer dispatch.

use std::mem::size_of;

use vortex_buffer::Buffer;
use vortex_mask::MaskValues;

use crate::arrays::filter::execute::byte_compress;
use crate::arrays::filter::execute::simd_compress;
use crate::arrays::filter::execute::slice;

const CACHED_INDICES_MAX_DENSITY: f64 = 0.5;
const IN_PLACE_MIN_DENSITY: f64 = 0.5;
const MIN_SLICES_AVERAGE_RUN_LENGTH: usize = 8;

/// Filter a [`Buffer<T>`] by [`MaskValues`], returning a new buffer.
///
/// Dense uniquely owned buffers are compacted in place; other buffers allocate a new output.
pub(crate) fn filter_buffer<T: Copy>(buffer: Buffer<T>, mask: &MaskValues) -> Buffer<T> {
    assert_eq!(buffer.len(), mask.len());

    let buffer = if mask.density() >= IN_PLACE_MIN_DENSITY {
        match buffer.try_into_mut() {
            Ok(mut buffer_mut) => {
                let new_len = filter_slice_in_place(buffer_mut.as_mut_slice(), mask);
                buffer_mut.truncate(new_len);
                return buffer_mut.freeze();
            }
            Err(buffer) => buffer,
        }
    } else {
        buffer
    };

    filter_slice(buffer.as_slice(), mask)
}

fn filter_slice<T: Copy>(values: &[T], mask: &MaskValues) -> Buffer<T> {
    if let Some(slices) = useful_cached_slices(mask) {
        return slice::filter_slice_by_slices(values, slices, mask.true_count());
    }

    if mask.density() <= CACHED_INDICES_MAX_DENSITY
        && let Some(indices) = mask.cached_indices()
    {
        return slice::filter_slice_by_indices(values, indices);
    }

    if let Some(filtered) = simd_compress::filter_slice_by_bitmap(values, mask) {
        return filtered;
    }

    if mask.density() >= byte_compress_density_threshold::<T>() {
        byte_compress::filter_buffer(values, mask)
    } else {
        slice::filter_slice_by_bitmap(values, mask)
    }
}

fn filter_slice_in_place<T: Copy>(values: &mut [T], mask: &MaskValues) -> usize {
    if let Some(slices) = useful_cached_slices(mask) {
        return slice::filter_slice_mut_by_slices(values, slices);
    }

    if mask.density() <= CACHED_INDICES_MAX_DENSITY
        && let Some(indices) = mask.cached_indices()
    {
        return slice::filter_slice_mut_by_indices(values, indices);
    }

    if let Some(new_len) = simd_compress::filter_slice_mut_by_bitmap(values, mask) {
        return new_len;
    }

    slice::filter_slice_mut_by_bitmap(values, mask)
}

fn useful_cached_slices(mask: &MaskValues) -> Option<&[(usize, usize)]> {
    mask.cached_slices().filter(|slices| {
        slices.len() == 1 || mask.true_count() / slices.len() >= MIN_SLICES_AVERAGE_RUN_LENGTH
    })
}

fn byte_compress_density_threshold<T>() -> f64 {
    let width = size_of::<T>();

    // A density at or above the table entry selects byte compress after the higher-priority
    // strategies have declined the mask. These crossovers are benchmarked in
    // `benches/filter_fixed_width.rs`.
    //
    // | Target | 1 byte | 2 bytes | 4 bytes | 8 bytes | other |
    // | --- | ---: | ---: | ---: | ---: | ---: |
    // | aarch64 | 0.90 | 0.90 | 0.90 | 0.75 | 0.90 |
    // | other | 0.00 | 0.50 | 0.50 | 0.75 | 0.875 |
    if cfg!(target_arch = "aarch64") {
        return match width {
            8 => 0.75,
            _ => 0.9,
        };
    }

    match width {
        1 => 0.0,
        2 | 4 => 0.5,
        8 => 0.75,
        _ => 0.875,
    }
}

/// Materialize sparse indices when enough sibling arrays will reuse the same mask.
pub(super) fn prepare_mask_for_reuse(mask: &MaskValues, consumers: usize) {
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

    fn mask_values(mask: &Mask) -> &MaskValues {
        match mask {
            Mask::Values(v) => v.as_ref(),
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
        let mask = Mask::from_iter([true, true, true, true, false, true, true, true, false, true]);

        let result = filter_buffer(buf, mask_values(&mask));
        assert_eq!(result, buffer![1u32, 2, 3, 4, 6, 7, 8, 10]);
    }

    #[test]
    fn test_filter_shared_buffer_by_cached_indices() {
        let buf = Buffer::from(BufferMut::from_iter(0u64..16));
        let shared = buf.clone();
        let mask = Mask::from_indices(16, [1, 5, 9, 15]);

        let result = filter_buffer(buf, mask_values(&mask));
        assert_eq!(result, buffer![1u64, 5, 9, 15]);
        assert_eq!(shared.len(), 16);
    }

    #[test]
    fn test_filter_shared_buffer_by_cached_slices() {
        let buf = Buffer::from(BufferMut::from_iter(0u32..32));
        let shared = buf.clone();
        let mask = Mask::from_slices(32, vec![(3, 15), (20, 30)]);

        let result = filter_buffer(buf, mask_values(&mask));
        let expected = (3u32..15).chain(20..30).collect::<Vec<_>>();
        assert_eq!(result.as_slice(), expected.as_slice());
        assert_eq!(shared.len(), 32);
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
