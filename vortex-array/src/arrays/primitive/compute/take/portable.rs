// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An implementation of the Take kernel for primitive Arrays that uses
//! the nightly-only `portable_simd` feature.
//!
//! This is only enabled on non-x86_64 platforms and when using the nightly compiler for builds.

#![allow(unused)]

use std::mem::MaybeUninit;
use std::mem::size_of;
use std::mem::transmute;
use std::simd;
use std::simd::num::SimdUint;

use multiversion::multiversion;
use num_traits::AsPrimitive;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::compute::take::TakeImpl;
use crate::arrays::primitive::vtable::Primitive;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;
use crate::dtype::half::f16;
use crate::match_each_native_simd_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::validity::Validity;

pub(super) struct TakeKernelPortableSimd;

impl TakeImpl for TakeKernelPortableSimd {
    fn take(
        &self,
        array: ArrayView<'_, Primitive>,
        unsigned_indices: ArrayView<'_, Primitive>,
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        if array.ptype() == PType::F16 {
            // Special handling for f16 to treat as opaque u16.
            let decoded = match_each_unsigned_integer_ptype!(unsigned_indices.ptype(), |C| {
                take_portable_simd::<u16, C, SIMD_WIDTH>(
                    array.reinterpret_cast(PType::U16).as_slice(),
                    unsigned_indices.as_slice(),
                )
            });
            Ok(PrimitiveArray::new(decoded, validity)
                .reinterpret_cast(PType::F16)
                .into_array())
        } else {
            match_each_unsigned_integer_ptype!(unsigned_indices.ptype(), |C| {
                match_each_native_simd_ptype!(array.ptype(), |V| {
                    let decoded = take_portable_simd::<V, C, SIMD_WIDTH>(
                        array.as_slice(),
                        unsigned_indices.as_slice(),
                    );
                    Ok(PrimitiveArray::new(decoded, validity).into_array())
                })
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Portable SIMD take algorithm
// ---------------------------------------------------------------------------

/// SIMD types larger than the SIMD register size are beneficial for performance as this leads to
/// better instruction level parallelism.
const SIMD_WIDTH: usize = 64;

/// Takes the specified indices into a new [`Buffer`] using portable SIMD.
///
/// This function handles the type matching required to satisfy [`simd::SimdElement`] bounds. For
/// `f16` values, it reinterprets them as `u16` since `f16` doesn't implement `SimdElement`.
fn take_portable<T: NativePType, I: UnsignedPType>(buffer: &[T], indices: &[I]) -> Buffer<T> {
    if T::PTYPE == PType::F16 {
        assert_eq!(size_of::<f16>(), size_of::<T>());

        // Since Rust does not actually support 16-bit floats, we first reinterpret the data as
        // `u16` integers.
        // SAFETY: We know that f16 has the same bit pattern as u16, so this transmute is fine.
        let u16_slice: &[u16] =
            unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast(), buffer.len()) };
        return unsafe { take_with_indices(u16_slice, indices).transmute::<T>() };
    }

    match_each_native_simd_ptype!(T::PTYPE, |TC| {
        assert_eq!(size_of::<TC>(), size_of::<T>());

        // SAFETY: This is essentially a no-op that tricks the compiler into adding the
        // `simd::SimdElement` bound we need to call `take_with_indices`.
        let buffer: &[TC] =
            unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast::<TC>(), buffer.len()) };
        unsafe { take_with_indices(buffer, indices).transmute::<T>() }
    })
}

/// Helper that matches on index type and calls [`take_portable_simd`].
///
/// We separate this code out from above to add the [`simd::SimdElement`] constraint.
fn take_with_indices<T: NativePType + simd::SimdElement, I: UnsignedPType>(
    buffer: &[T],
    indices: &[I],
) -> Buffer<T> {
    match_each_unsigned_integer_ptype!(I::PTYPE, |IC| {
        let indices: &[IC] =
            unsafe { std::slice::from_raw_parts(indices.as_ptr().cast::<IC>(), indices.len()) };
        take_portable_simd::<T, IC, SIMD_WIDTH>(buffer, indices)
    })
}

/// Takes elements from an array using SIMD indexing.
///
/// Performs a gather operation that takes values at specified indices and returns them in a new
/// buffer. Uses SIMD instructions to process `LANE_COUNT` indices in parallel.
///
/// Returns a [`Buffer<T>`] where each element corresponds to `values[indices[i]]`.
#[multiversion(targets("x86_64+avx2", "x86_64+avx", "aarch64+neon"))]
fn take_portable_simd<T, I, const LANE_COUNT: usize>(values: &[T], indices: &[I]) -> Buffer<T>
where
    T: NativePType + simd::SimdElement,
    I: UnsignedPType + simd::SimdElement,
    simd::Simd<I, LANE_COUNT>: SimdUint<Cast<usize> = simd::Simd<usize, LANE_COUNT>>,
{
    let indices_len = indices.len();

    let mut buffer = BufferMut::<T>::with_capacity_aligned(
        indices_len,
        Alignment::of::<simd::Simd<T, LANE_COUNT>>(),
    );

    let buf_slice = buffer.spare_capacity_mut();

    for chunk_idx in 0..(indices_len / LANE_COUNT) {
        let offset = chunk_idx * LANE_COUNT;
        let mask = simd::Mask::from_bitmask(u64::MAX);
        let codes_chunk = simd::Simd::<I, LANE_COUNT>::from_slice(&indices[offset..]);

        let selection = simd::Simd::gather_select(
            values,
            mask,
            codes_chunk.cast::<usize>(),
            simd::Simd::<T, LANE_COUNT>::default(),
        );

        unsafe {
            selection.store_select_unchecked(
                transmute::<&mut [MaybeUninit<T>], &mut [T]>(&mut buf_slice[offset..][..64]),
                mask.cast(),
            );
        }
    }

    for idx in ((indices_len / LANE_COUNT) * LANE_COUNT)..indices_len {
        unsafe {
            buf_slice
                .get_unchecked_mut(idx)
                .write(values[indices[idx].as_()]);
        }
    }

    unsafe {
        buffer.set_len(indices_len);
    }

    // NOTE: if we don't do this, we pass back a Buffer which is over-aligned to the SIMD
    // register width. The caller expects that this memory should be aligned to the value type
    // so that we can slice it at value boundaries.
    buffer = buffer.aligned(Alignment::of::<T>());

    buffer.freeze()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_take_out_of_bounds() {
        let indices = vec![2_000_000u32; 64];
        let values = vec![1i32];

        let result = take_portable_simd::<i32, u32, 64>(&values, &indices);
        assert_eq!(result.as_slice(), [0i32; 64]);
    }

    /// Tests SIMD gather with a mix of sequential, strided, and repeated indices. This exercises
    /// irregular access patterns that stress the gather operation.
    #[test]
    fn test_take_mixed_access_patterns() {
        // Create a values array with distinct elements.
        let values: Vec<i64> = (0..256).map(|i| i * 100).collect();

        // Build indices with mixed patterns:
        // - Sequential access (0, 1, 2, ...).
        // - Strided access (0, 4, 8, ...).
        // - Repeated indices (same index multiple times).
        // - Reverse order.
        let mut indices: Vec<u32> = Vec::with_capacity(200);

        // Sequential: indices 0..64.
        indices.extend(0u32..64);
        // Strided by 4: 0, 4, 8, ..., 252.
        indices.extend((0u32..64).map(|i| i * 4));
        // Repeated: index 42 repeated 32 times.
        indices.extend(std::iter::repeat(42u32).take(32));
        // Reverse: 255, 254, ..., 216.
        indices.extend((216u32..256).rev());

        let result = take_portable_simd::<i64, u32, 64>(&values, &indices);
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

        // Use 64 + 37 = 101 indices to test both the SIMD loop (64 elements) and the scalar
        // remainder (37 elements).
        let indices: Vec<u8> = (0u8..101).collect();

        let result = take_portable_simd::<u16, u8, 64>(&values, &indices);
        let result_slice = result.as_slice();

        assert_eq!(result_slice.len(), 101);

        // Verify all elements.
        for i in 0..101 {
            assert_eq!(result_slice[i], i as u16, "mismatch at index {i}");
        }

        // Also test with exactly 1 remainder element.
        let indices_one_remainder: Vec<u8> = (0u8..65).collect();
        let result_one = take_portable_simd::<u16, u8, 64>(&values, &indices_one_remainder);
        assert_eq!(result_one.as_slice().len(), 65);
        assert_eq!(result_one.as_slice()[64], 64);
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
        let indices: Vec<u16> = vec![
            0, 8, 1, 7, 2, 6, 3, 5, 4, // Forward-backward interleaved.
            8, 8, 8, 0, 0, 0, // Repeated extremes.
            4, 4, 4, 4, 4, 4, 4, 4, // Repeated zero.
            0, 1, 2, 3, 4, 5, 6, 7, 8, // Sequential.
            8, 7, 6, 5, 4, 3, 2, 1, 0, // Reverse.
            // Pad to 64 to ensure we hit the SIMD path.
            0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3,
        ];

        let result = take_portable_simd::<i64, u16, 64>(&values, &indices);
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
