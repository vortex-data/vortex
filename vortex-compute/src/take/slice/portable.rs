// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take function implementations on slices using `portable_simd`.

#![cfg(vortex_nightly)]

use std::mem::MaybeUninit;
use std::mem::size_of;
use std::simd;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdUint;

use multiversion::multiversion;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::UnsignedPType;
use vortex_dtype::match_each_unsigned_integer_ptype;

/// SIMD lane count. Benchmarking shows 16 is optimal on AVX-512 systems:
/// - 8 lanes: ~33 µs for 100k elements
/// - 16 lanes: ~31 µs (best)
/// - 32 lanes: ~34 µs
/// - 64 lanes: ~41 µs (causes stack spills)
pub const SIMD_WIDTH: usize = 16;

/// Takes the specified indices into a new [`Buffer`] using portable SIMD.
///
/// This function handles the type matching required to satisfy `SimdElement` bounds by casting
/// to unsigned integers of the same size. Falls back to scalar implementation for unsupported
/// type sizes.
#[inline]
pub fn take_portable<T: Copy, I: UnsignedPType>(buffer: &[T], indices: &[I]) -> Buffer<T> {
    // SIMD gather operations only care about bit patterns, not semantic type. We cast to unsigned
    // integers which implement `SimdElement` and then cast back.
    //
    // SAFETY: The pointer casts below are safe because:
    // - `T` and the target type have the same size (matched by `size_of::<T>()`).
    // - The alignment of unsigned integers is always <= their size, and `buffer` came from a valid
    //   `&[T]` which guarantees proper alignment for types of the same size.
    match size_of::<T>() {
        1 => {
            let buffer: &[u8] =
                unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const u8, buffer.len()) };
            take_with_indices(buffer, indices).cast_into::<T>()
        }
        2 => {
            let buffer: &[u16] =
                unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const u16, buffer.len()) };
            take_with_indices(buffer, indices).cast_into::<T>()
        }
        4 => {
            let buffer: &[u32] =
                unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const u32, buffer.len()) };
            take_with_indices(buffer, indices).cast_into::<T>()
        }
        8 => {
            let buffer: &[u64] =
                unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const u64, buffer.len()) };
            take_with_indices(buffer, indices).cast_into::<T>()
        }
        // Fall back to scalar implementation for unsupported type sizes.
        _ => super::take_scalar(buffer, indices),
    }
}

/// Helper that matches on index type and calls `take_portable_simd`.
///
/// We separate this code out from above to add the [`simd::SimdElement`] constraint.
#[inline]
fn take_with_indices<T: Copy + Default + simd::SimdElement, I: UnsignedPType>(
    buffer: &[T],
    indices: &[I],
) -> Buffer<T> {
    match_each_unsigned_integer_ptype!(I::PTYPE, |IC| {
        let indices: &[IC] =
            unsafe { std::slice::from_raw_parts(indices.as_ptr() as *const IC, indices.len()) };
        take_portable_simd::<T, IC, SIMD_WIDTH>(buffer, indices)
    })
}

/// Takes elements from an array using SIMD indexing.
///
/// Performs a gather operation that takes values at specified indices and returns them in a new
/// buffer. Uses SIMD instructions to process `LANE_COUNT` indices in parallel.
///
/// Returns a `Buffer<T>` where each element corresponds to `values[indices[i]]`.
///
/// # Panics
///
/// Panics if any index is out of bounds for `values`.
#[multiversion(targets("x86_64+avx512f+avx512vl", "x86_64+avx2", "x86_64+avx", "aarch64+neon"))]
pub fn take_portable_simd<T, I, const LANE_COUNT: usize>(values: &[T], indices: &[I]) -> Buffer<T>
where
    T: Copy + Default + simd::SimdElement,
    I: UnsignedPType + simd::SimdElement,
    simd::LaneCount<LANE_COUNT>: simd::SupportedLaneCount,
    simd::Simd<I, LANE_COUNT>: SimdUint<Cast<usize> = simd::Simd<usize, LANE_COUNT>>,
{
    let indices_len = indices.len();

    let mut buffer = BufferMut::<T>::with_capacity_aligned(
        indices_len,
        Alignment::of::<simd::Simd<T, LANE_COUNT>>(),
    );

    let buf_slice = buffer.spare_capacity_mut();

    // Set up a vector that we can SIMD compare against for out-of-bounds indices.
    let len_vec = simd::Simd::<usize, LANE_COUNT>::splat(values.len());
    let mut all_valid = simd::Mask::<isize, LANE_COUNT>::splat(true);

    for chunk_idx in 0..(indices_len / LANE_COUNT) {
        let offset = chunk_idx * LANE_COUNT;
        let codes_chunk = simd::Simd::<I, LANE_COUNT>::from_slice(&indices[offset..]);
        let codes_usize = codes_chunk.cast::<usize>();

        // Accumulate validity and use as gather mask. An out-of-bounds index will turn a bit off.
        all_valid &= codes_usize.simd_lt(len_vec);

        // SAFETY: We use `all_valid` to mask the gather, preventing OOB memory access. If any
        // index is OOB, `all_valid` will have those bits turned off, masking out the invalid
        // indices.
        // Note that this may also mask out valid indices in subsequent iterations. This is fine
        // because we will panic after the loop if **any** index was OOB, so we do not care if the
        // resulting gathered data is correct or not.
        let selection = unsafe {
            simd::Simd::gather_select_unchecked(
                values,
                all_valid,
                codes_usize,
                simd::Simd::<T, LANE_COUNT>::default(),
            )
        };

        // SAFETY: `MaybeUninit<T>` has the same layout as `T`, and we are about to initialize these
        // elements with the store.
        let uninit = unsafe {
            std::mem::transmute::<&mut [MaybeUninit<T>], &mut [T]>(
                &mut buf_slice[offset..][..LANE_COUNT],
            )
        };

        // SAFETY: The slice `buf_slice[offset..][..LANE_COUNT]` is guaranteed to have exactly
        // `LANE_COUNT` elements since `offset` is a multiple of `LANE_COUNT` and we only iterate
        // while `offset + LANE_COUNT <= indices_len`.
        unsafe {
            selection.store_select_unchecked(uninit, simd::Mask::splat(true));
        }
    }

    // Check accumulated validity after hot loop. If there are any 0's, then there was an
    // out-of-bounds index.
    assert!(all_valid.all(), "index out of bounds in SIMD take");

    // Fall back to scalar iteration for the remainder.
    for idx in ((indices_len / LANE_COUNT) * LANE_COUNT)..indices_len {
        // SAFETY: `idx` is in bounds for `buf_slice` since `idx < indices_len == buf_slice.len()`.
        // Note that the `values[...]` access is already bounds-checked and will panic if OOB.
        unsafe {
            buf_slice
                .get_unchecked_mut(idx)
                .write(values[indices[idx].as_()]);
        }
    }

    // SAFETY: All elements have been initialized: the SIMD loop handles `0..chunks * LANE_COUNT`
    // and the scalar loop handles the remainder up to `indices_len`.
    unsafe { buffer.set_len(indices_len) };

    buffer.freeze()
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::take_portable_simd;

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_take_out_of_bounds() {
        let indices = vec![2_000_000u32; 8];
        let values = vec![1i32];

        drop(take_portable_simd::<i32, u32, 8>(&values, &indices));
    }

    /// Tests SIMD gather with a mix of sequential, strided, and repeated indices. This exercises
    /// irregular access patterns that stress the gather operation.
    #[test]
    fn test_take_mixed_access_patterns() {
        // Create a values array with distinct elements.
        let values: Vec<i64> = (0..256).map(|i| i * 100).collect();

        // Build indices with mixed patterns:
        // - Sequential access (0, 1, 2, ...)
        // - Strided access (0, 4, 8, ...)
        // - Repeated indices (same index multiple times)
        // - Reverse order
        let mut indices: Vec<u32> = Vec::with_capacity(200);

        // Sequential: indices 0..64.
        indices.extend(0u32..64);
        // Strided by 4: 0, 4, 8, ..., 252.
        indices.extend((0u32..64).map(|i| i * 4));
        // Repeated: index 42 repeated 32 times.
        indices.extend(std::iter::repeat_n(42u32, 32));
        // Reverse: 255, 254, ..., 216.
        indices.extend((216u32..256).rev());

        // Use 4 lanes for i64 (256-bit / 64-bit = 4).
        let result = take_portable_simd::<i64, u32, 4>(&values, &indices);
        let result_slice = result.as_slice();

        // Verify sequential portion.
        for i in 0..64 {
            assert_eq!(result_slice[i], (i as i64) * 100, "sequential at index {i}");
        }

        // Verify strided portion.
        for i in 0..64 {
            assert_eq!(
                result_slice[64 + i],
                (i as i64) * 4 * 100,
                "strided at index {i}"
            );
        }

        // Verify repeated portion.
        for i in 0..32 {
            assert_eq!(result_slice[128 + i], 42 * 100, "repeated at index {i}");
        }

        // Verify reverse portion.
        for i in 0..40 {
            assert_eq!(
                result_slice[160 + i],
                (255 - i as i64) * 100,
                "reverse at index {i}"
            );
        }
    }

    /// Tests that the scalar remainder path works correctly when the number of indices is not
    /// evenly divisible by the SIMD lane count.
    #[test]
    fn test_take_with_remainder() {
        let values: Vec<u16> = (0..1000).collect();

        // Use 8 + 5 = 13 indices to test both the SIMD loop (8 elements) and the scalar
        // remainder (5 elements). Using 8 lanes for u16 values.
        let indices: Vec<u8> = (0u8..13).collect();

        let result = take_portable_simd::<u16, u8, 8>(&values, &indices);
        let result_slice = result.as_slice();

        assert_eq!(result_slice.len(), 13);

        // Verify all elements.
        for i in 0..13 {
            assert_eq!(result_slice[i], i as u16, "mismatch at index {i}");
        }

        // Also test with exactly 1 remainder element.
        let indices_one_remainder: Vec<u8> = (0u8..9).collect();
        let result_one = take_portable_simd::<u16, u8, 8>(&values, &indices_one_remainder);
        assert_eq!(result_one.as_slice().len(), 9);
        assert_eq!(result_one.as_slice()[8], 8);
    }

    /// Tests gather with large 64-bit values and various index types to ensure no truncation
    /// occurs during the operation.
    #[test]
    fn test_take_large_values_no_truncation() {
        // Create values near the edges of i64 range.
        let values: Vec<i64> = vec![
            i64::MIN,
            i64::MIN + 1,
            -1_000_000_000_000i64,
            -1,
            0,
            1,
            1_000_000_000_000i64,
            i64::MAX - 1,
            i64::MAX,
        ];

        // Indices that access each value multiple times in different orders.
        // Pad to 8 to ensure we hit the SIMD path (4 lanes for i64).
        let indices: Vec<u16> = vec![
            0, 8, 1, 7, 2, 6, 3, 5, // 8 indices - exercises SIMD path
        ];

        // Use 4 lanes for i64 (256-bit / 64-bit = 4).
        let result = take_portable_simd::<i64, u16, 4>(&values, &indices);
        let result_slice = result.as_slice();

        // Verify each result matches the expected value.
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(
                result_slice[i], values[idx as usize],
                "mismatch at position {i} for index {idx}"
            );
        }
    }
}
