// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Implementations of a specialized in-place filter for mutable buffers using AVX512.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::filter::slice::SimdCompress;
use crate::filter::slice::in_place::filter_in_place_scalar;

/// Filter a mutable slice of elements in-place depending on the given mask.
///
/// The mask is represented as a slice of bytes (LSB is the first element).
///
/// Returns the true count of the mask (number of elements remaining).
///
/// This function automatically dispatches to the most efficient implementation based on the
/// available CPU features at compile time.
///
/// # Panics
///
/// Panics if `mask.len() != data.len().div_ceil(8)`.
#[inline]
pub fn filter_in_place<T: SimdCompress>(data: &mut [T], mask: &[u8]) -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let use_simd = if T::WIDTH >= 32 {
            // 32-bit and 64-bit types only need AVX-512F.
            is_x86_feature_detected!("avx512f")
        } else {
            // 8-bit and 16-bit types need both AVX-512F and AVX-512VBMI2.
            is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vbmi2")
        };

        if use_simd {
            return unsafe { filter_in_place_avx512(data, mask) };
        }
    }

    // Fall back to scalar implementation for non-x86 or when SIMD not available.
    filter_in_place_scalar(data, mask)
}

/// Filter a mutable slice of elements in-place depending on the given mask.
///
/// The mask is represented as a slice of bytes (LSB is the first element).
///
/// Returns the true count of the mask.
///
/// This function uses AVX-512 SIMD instructions for high-performance filtering.
///
/// # Panics
///
/// Panics if `mask.len() != data.len().div_ceil(8)`.
///
/// # Safety
///
/// This function requires the appropriate SIMD instruction set to be available.
/// For AVX-512F types, the CPU must support AVX-512F.
/// For AVX-512VBMI2 types, the CPU must support AVX-512VBMI2.
#[inline]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512vbmi2,popcnt")]
pub unsafe fn filter_in_place_avx512<T: SimdCompress>(data: &mut [T], mask: &[u8]) -> usize {
    assert_eq!(
        mask.len(),
        data.len().div_ceil(8),
        "Mask length must be data.len().div_ceil(8)"
    );

    let data_len = data.len();
    let mut write_pos = 0;

    // Pre-calculate loop bounds to eliminate branch misprediction in the hot loop.
    let full_chunks = data_len / T::ELEMENTS_PER_VECTOR;
    let remainder = data_len % T::ELEMENTS_PER_VECTOR;

    // Process full chunks with no branches in the loop.
    for chunk_idx in 0..full_chunks {
        let read_pos = chunk_idx * T::ELEMENTS_PER_VECTOR;
        let mask_byte_offset = chunk_idx * T::MASK_BYTES;

        // Read the mask for this chunk.
        // SAFETY: `mask_byte_offset + T::MASK_BYTES <= mask.len()` for all full chunks.
        let mask_value = unsafe { T::read_mask(mask.as_ptr(), mask_byte_offset) };

        // Load elements into the SIMD register.
        // SAFETY: `read_pos + T::ELEMENTS_PER_VECTOR <= data.len()` for all full chunks.
        let vector = unsafe { _mm512_loadu_si512(data.as_ptr().add(read_pos) as *const __m512i) };

        // Moves all elements that have their bit set to 1 in the mask value to the left.
        let filtered = unsafe { T::compress_vector(mask_value, vector) };

        // Write the filtered result vector back to memory.
        // SAFETY: `write_pos + count_ones(mask_value) <= data.len()` since we're compacting.
        unsafe { _mm512_storeu_si512(data.as_mut_ptr().add(write_pos) as *mut __m512i, filtered) };

        // Uses the hardware `popcnt` instruction if available.
        let count = T::count_ones(mask_value);
        write_pos += count;
    }

    // Handle the final partial chunk with simple scalar processing.
    let read_pos = full_chunks * T::ELEMENTS_PER_VECTOR;
    for i in 0..remainder {
        let read_idx = read_pos + i;
        let bit_idx = read_idx % 8;
        let byte_idx = read_idx / 8;

        if (mask[byte_idx] >> bit_idx) & 1 == 1 {
            data[write_pos] = data[read_idx];
            write_pos += 1;
        }
    }

    write_pos
}
