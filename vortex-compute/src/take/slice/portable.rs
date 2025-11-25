// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Take function implementations on slices using `portable_simd`.

#![cfg(vortex_nightly)]

use std::mem::MaybeUninit;
use std::mem::transmute;
use std::simd;
use std::simd::num::SimdUint;

use multiversion::multiversion;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_dtype::PType;
use vortex_dtype::UnsignedPType;

/// Takes the specified indices into a new [`Buffer`] using portable SIMD.
#[inline]
pub fn take_portable<T, I>(buffer: &[T], indices: &[I]) -> Buffer<T>
where
    T: NativePType + simd::SimdElement,
    I: UnsignedPType + simd::SimdElement,
{
    if T::PTYPE == PType::F16 {
        // Since Rust does not actually support 16-bit floats, we first reinterpret the data as
        // `u16` integers.
        let u16_slice: &[u16] =
            unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const u16, buffer.len()) };

        let taken_u16 = take_portable_simd::<u16, I, SIMD_WIDTH>(u16_slice, indices);
        let taken_f16 = taken_u16.cast_into::<T>();

        taken_f16
    } else {
        take_portable_simd::<T, I, SIMD_WIDTH>(buffer, indices)
    }
}

/// Takes elements from an array using SIMD indexing.
///
/// Performs a gather operation that takes values at specified indices and returns them in a new
/// buffer. Uses SIMD instructions to process `LANE_COUNT` indices in parallel.
///
/// Returns a `Buffer<T>` where each element corresponds to `values[indices[i]]`.
#[multiversion(targets("x86_64+avx2", "x86_64+avx", "aarch64+neon"))]
pub fn take_portable_simd<T, I, const LANE_COUNT: usize>(values: &[T], indices: &[I]) -> Buffer<T>
where
    T: NativePType + simd::SimdElement,
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

    buffer.freeze()
}

#[cfg(test)]
mod tests {
    use super::take_portable_simd;

    #[test]
    fn test_take_out_of_bounds() {
        let indices = vec![2_000_000u32; 64];
        let values = vec![1i32];

        let result = take_portable_simd::<i32, u32, 64>(&values, &indices);
        assert_eq!(result.as_slice(), [0i32; 64]);
    }
}
