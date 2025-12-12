// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An AVX2 implementation of take operation using gather instructions.
//!
//! Only enabled for x86_64 hosts and it is gated at runtime behind feature detection to ensure AVX2
//! instructions are available.

#![allow(
    unused,
    reason = "Compiler may see things in this module as unused based on enabled features"
)]
#![cfg(any(target_arch = "x86_64", target_arch = "x86"))]

use std::arch::x86_64::__m256i;
use std::arch::x86_64::_mm_loadu_si128;
use std::arch::x86_64::_mm_movemask_epi8;
use std::arch::x86_64::_mm_setzero_si128;
use std::arch::x86_64::_mm_shuffle_epi32;
use std::arch::x86_64::_mm_storeu_si128;
use std::arch::x86_64::_mm_unpacklo_epi64;
use std::arch::x86_64::_mm256_cmpgt_epi32;
use std::arch::x86_64::_mm256_cmpgt_epi64;
use std::arch::x86_64::_mm256_cvtepu8_epi32;
use std::arch::x86_64::_mm256_cvtepu8_epi64;
use std::arch::x86_64::_mm256_cvtepu16_epi32;
use std::arch::x86_64::_mm256_cvtepu16_epi64;
use std::arch::x86_64::_mm256_cvtepu32_epi64;
use std::arch::x86_64::_mm256_extracti128_si256;
use std::arch::x86_64::_mm256_loadu_si256;
use std::arch::x86_64::_mm256_mask_i32gather_epi32;
use std::arch::x86_64::_mm256_mask_i64gather_epi32;
use std::arch::x86_64::_mm256_mask_i64gather_epi64;
use std::arch::x86_64::_mm256_movemask_epi8;
use std::arch::x86_64::_mm256_set1_epi32;
use std::arch::x86_64::_mm256_set1_epi64x;
use std::arch::x86_64::_mm256_setzero_si256;
use std::arch::x86_64::_mm256_storeu_si256;
use std::convert::identity;
use std::mem::size_of;

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::UnsignedPType;
use vortex_dtype::match_each_unsigned_integer_ptype;

use crate::take::slice::take_scalar;

/// Takes the specified indices into a new [`Buffer`] using AVX2 SIMD.
///
/// This function handles the type matching required to satisfy AVX2 gather instruction requirements
/// by casting to unsigned integers of the same size. Falls back to scalar implementation for
/// unsupported type sizes.
///
/// # Panics
///
/// This function panics if any of the provided `indices` are out of bounds for `values`.
///
/// # Safety
///
/// The caller must ensure the `avx2` feature is enabled.
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn take_avx2<V: Copy, I: UnsignedPType>(buffer: &[V], indices: &[I]) -> Buffer<V> {
    // AVX2 gather operations only care about bit patterns, not semantic type. We cast to unsigned
    // integers which have the required gather implementations and then cast back.
    //
    // SAFETY: The pointer casts below are safe because:
    // - `V` and the target type have the same size (matched by `size_of::<V>()`)
    // - The alignment of unsigned integers is always <= their size, and `buffer` came from a valid
    //   `&[V]` which guarantees proper alignment for types of the same size.
    match size_of::<V>() {
        4 => {
            let values: &[u32] =
                unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast::<u32>(), buffer.len()) };
            match_each_unsigned_integer_ptype!(I::PTYPE, |IC| {
                let indices: &[IC] = unsafe {
                    std::slice::from_raw_parts(indices.as_ptr().cast::<IC>(), indices.len())
                };
                exec_take::<u32, IC, AVX2Gather>(values, indices).cast_into::<V>()
            })
        }
        8 => {
            let values: &[u64] =
                unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast::<u64>(), buffer.len()) };
            match_each_unsigned_integer_ptype!(I::PTYPE, |IC| {
                let indices: &[IC] = unsafe {
                    std::slice::from_raw_parts(indices.as_ptr().cast::<IC>(), indices.len())
                };
                exec_take::<u64, IC, AVX2Gather>(values, indices).cast_into::<V>()
            })
        }
        // Fall back to scalar implementation for unsupported type sizes (1, 2 byte types).
        _ => take_scalar(buffer, indices),
    }
}

/// The main gather function that is used by the inner loop kernel for AVX2 gather.
pub(crate) trait GatherFn<Idx, Values> {
    /// The number of data elements that are written to the `dst` on each loop iteration.
    const WIDTH: usize;
    /// The number of indices read from `indices` on each loop iteration.
    /// Depending on the available instructions and bit-width we may stride by a larger amount
    /// than we actually end up reading from `src` (governed by the `WIDTH` parameter).
    const STRIDE: usize = Self::WIDTH;

    /// Gather values from `src` into the `dst` using the `indices`, optionally using
    /// SIMD instructions.
    ///
    /// Returns `true` if all indices in this batch were valid (less than `max_idx`), `false`
    /// otherwise. Invalid indices are masked out during the gather (substituting zeros).
    ///
    /// # Safety
    ///
    /// This function can read up to `STRIDE` elements through `indices`, and read/write up to
    /// `WIDTH` elements through `src` and `dst` respectively.
    unsafe fn gather(
        indices: *const Idx,
        max_idx: Idx,
        src: *const Values,
        dst: *mut Values,
    ) -> bool;
}

/// AVX2 version of GatherFn defined for 32- and 64-bit value types.
enum AVX2Gather {}

macro_rules! impl_gather {
    ($idx:ty, $({$value:ty => load: $load:ident, extend: $extend:ident, splat: $splat:ident, zero_vec: $zero_vec:ident, mask_indices: $mask_indices:ident, mask_cvt: |$mask_var:ident| $mask_cvt:block, movemask: $movemask:ident, all_valid_mask: $all_valid_mask:expr, gather: $masked_gather:ident, store: $store:ident, WIDTH = $WIDTH:literal, STRIDE = $STRIDE:literal }),+) => {
        $(
            impl_gather!(single; $idx, $value, load: $load, extend: $extend, splat: $splat, zero_vec: $zero_vec, mask_indices: $mask_indices, mask_cvt: |$mask_var| $mask_cvt, movemask: $movemask, all_valid_mask: $all_valid_mask, gather: $masked_gather, store: $store, WIDTH = $WIDTH, STRIDE = $STRIDE);
        )*
    };
    (single; $idx:ty, $value:ty, load: $load:ident, extend: $extend:ident, splat: $splat:ident, zero_vec: $zero_vec:ident, mask_indices: $mask_indices:ident, mask_cvt: |$mask_var:ident| $mask_cvt:block, movemask: $movemask:ident, all_valid_mask: $all_valid_mask:expr, gather: $masked_gather:ident, store: $store:ident, WIDTH = $WIDTH:literal, STRIDE = $STRIDE:literal) => {
            impl GatherFn<$idx, $value> for AVX2Gather {
                const WIDTH: usize = $WIDTH;
                const STRIDE: usize = $STRIDE;

                #[allow(unused_unsafe, clippy::cast_possible_truncation)]
                #[inline(always)]
                unsafe fn gather(
                    indices: *const $idx,
                    max_idx: $idx,
                    src: *const $value,
                    dst: *mut $value
                ) -> bool {
                    const {
                        assert!($WIDTH <= $STRIDE, "dst cannot advance by more than the stride");
                    }

                    const SCALE: i32 = std::mem::size_of::<$value>() as i32;

                    let indices_vec = unsafe { $load(indices.cast()) };
                    // Extend indices to fill vector register.
                    let indices_vec = unsafe { $extend(indices_vec) };

                    // Create a vec of the max idx.
                    let max_idx_vec = unsafe { $splat(max_idx as _) };
                    // Create a mask for valid indices (where the max_idx > provided index).
                    let valid_mask = unsafe { $mask_indices(max_idx_vec, indices_vec) };
                    let valid_mask = {
                        let $mask_var = valid_mask;
                        $mask_cvt
                    };
                    let zero_vec = unsafe { $zero_vec() };

                    // Gather the values into new vector register, for masked positions
                    // it substitutes zero instead of accessing the src.
                    let values_vec = unsafe {
                        $masked_gather::<SCALE>(zero_vec, src.cast(), indices_vec, valid_mask)
                    };

                    // Write the vec out to dst.
                    unsafe { $store(dst.cast(), values_vec) };

                    // Return true if all indices were valid (all mask bits set).
                    let mask_bits = unsafe { $movemask(valid_mask) };
                    mask_bits == $all_valid_mask
                }
            }
    };
}

// Kernels for u8 indices.
impl_gather!(u8,
    // 32-bit values, loaded 8 at a time.
    { u32 =>
        load: _mm_loadu_si128,
        extend: _mm256_cvtepu8_epi32,
        splat: _mm256_set1_epi32,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi32,
        mask_cvt: |x| { x },
        movemask: _mm256_movemask_epi8,
        all_valid_mask: -1_i32,
        gather: _mm256_mask_i32gather_epi32,
        store: _mm256_storeu_si256,
        WIDTH = 8, STRIDE = 16
    },
    // 64-bit values, loaded 4 at a time.
    { u64 =>
        load: _mm_loadu_si128,
        extend: _mm256_cvtepu8_epi64,
        splat: _mm256_set1_epi64x,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi64,
        mask_cvt: |x| { x },
        movemask: _mm256_movemask_epi8,
        all_valid_mask: -1_i32,
        gather: _mm256_mask_i64gather_epi64,
        store: _mm256_storeu_si256,
        WIDTH = 4, STRIDE = 16
    }
);

// Kernels for u16 indices.
impl_gather!(u16,
    // 32-bit values. 8x indices loaded at a time and 8x values written at a time.
    { u32 =>
        load: _mm_loadu_si128,
        extend: _mm256_cvtepu16_epi32,
        splat: _mm256_set1_epi32,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi32,
        mask_cvt: |x| { x },
        movemask: _mm256_movemask_epi8,
        all_valid_mask: -1_i32,
        gather: _mm256_mask_i32gather_epi32,
        store: _mm256_storeu_si256,
        WIDTH = 8, STRIDE = 8
    },
    // 64-bit values. 8x indices loaded at a time and 4x values loaded at a time.
    { u64 =>
        load: _mm_loadu_si128,
        extend: _mm256_cvtepu16_epi64,
        splat: _mm256_set1_epi64x,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi64,
        mask_cvt: |x| { x },
        movemask: _mm256_movemask_epi8,
        all_valid_mask: -1_i32,
        gather: _mm256_mask_i64gather_epi64,
        store: _mm256_storeu_si256,
        WIDTH = 4, STRIDE = 8
    }
);

// Kernels for u32 indices.
impl_gather!(u32,
    // 32-bit values. 8x indices loaded at a time and 8x values written.
    { u32 =>
        load: _mm256_loadu_si256,
        extend: identity,
        splat: _mm256_set1_epi32,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi32,
        mask_cvt: |x| { x },
        movemask: _mm256_movemask_epi8,
        all_valid_mask: -1_i32,
        gather: _mm256_mask_i32gather_epi32,
        store: _mm256_storeu_si256,
        WIDTH = 8, STRIDE = 8
    },
    // 64-bit values.
    { u64 =>
        load: _mm_loadu_si128,
        extend: _mm256_cvtepu32_epi64,
        splat: _mm256_set1_epi64x,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi64,
        mask_cvt: |x| { x },
        movemask: _mm256_movemask_epi8,
        all_valid_mask: -1_i32,
        gather: _mm256_mask_i64gather_epi64,
        store: _mm256_storeu_si256,
        WIDTH = 4, STRIDE = 4
    }
);

// Kernels for u64 indices.
impl_gather!(u64,
    // 32-bit values.
    { u32 =>
        load: _mm256_loadu_si256,
        extend: identity,
        splat: _mm256_set1_epi64x,
        zero_vec: _mm_setzero_si128,
        mask_indices: _mm256_cmpgt_epi64,
        mask_cvt: |m| {
            unsafe {
                let lo_bits = _mm256_extracti128_si256::<0>(m);    // lower half
                let hi_bits = _mm256_extracti128_si256::<1>(m);    // upper half
                let lo_packed = _mm_shuffle_epi32::<0b01_01_01_01>(lo_bits);
                let hi_packed = _mm_shuffle_epi32::<0b01_01_01_01>(hi_bits);
                _mm_unpacklo_epi64(lo_packed, hi_packed)
            }
        },
        movemask: _mm_movemask_epi8,
        all_valid_mask: 0xFFFF_i32,
        gather: _mm256_mask_i64gather_epi32,
        store: _mm_storeu_si128,
        WIDTH = 4, STRIDE = 4
    },
    // 64-bit values.
    { u64 =>
        load: _mm256_loadu_si256,
        extend: identity,
        splat: _mm256_set1_epi64x,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi64,
        mask_cvt: |x| { x },
        movemask: _mm256_movemask_epi8,
        all_valid_mask: -1_i32,
        gather: _mm256_mask_i64gather_epi64,
        store: _mm256_storeu_si256,
        WIDTH = 4, STRIDE = 4
    }
);

/// AVX2 core inner loop for certain `Idx` and `Value` type.
#[inline(always)]
fn exec_take<Value, Idx, Gather>(values: &[Value], indices: &[Idx]) -> Buffer<Value>
where
    Value: Copy,
    Idx: UnsignedPType,
    Gather: GatherFn<Idx, Value>,
{
    let indices_len = indices.len();
    let max_index = Idx::from(values.len()).unwrap_or_else(|| Idx::max_value());
    let mut buffer =
        BufferMut::<Value>::with_capacity_aligned(indices_len, Alignment::of::<__m256i>());
    let buf_uninit = buffer.spare_capacity_mut();

    let mut offset = 0;
    let mut all_valid = true;

    // Loop terminates STRIDE elements before end of the indices array because the GatherFn
    // might read up to STRIDE src elements at a time, even though it only advances WIDTH elements
    // in the dst.
    while offset + Gather::STRIDE < indices_len {
        // SAFETY: gather_simd preconditions satisfied:
        //  1. `(indices + offset)..(indices + offset + STRIDE)` is in-bounds for indices allocation
        //  2. `buffer` has same len as indices so `buffer + offset + STRIDE` is always valid.
        let batch_valid = unsafe {
            Gather::gather(
                indices.as_ptr().add(offset),
                max_index,
                values.as_ptr(),
                buf_uninit.as_mut_ptr().add(offset).cast(),
            )
        };
        all_valid &= batch_valid;
        offset += Gather::WIDTH;
    }

    // Check accumulated validity after hot loop. If there are any 0's, then there was an
    // out-of-bounds index.
    assert!(all_valid, "index out of bounds in AVX2 take");

    // Fall back to scalar iteration for the remainder.
    while offset < indices_len {
        buf_uninit[offset].write(values[indices[offset].as_()]);
        offset += 1;
    }

    assert_eq!(offset, indices_len);

    // SAFETY: all elements have been initialized.
    unsafe { buffer.set_len(indices_len) };

    buffer.freeze()
}

#[cfg(test)]
#[cfg_attr(miri, ignore)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;

    macro_rules! test_cases {
        (index_type => $IDX:ty, value_types => $($VAL:ty),+) => {
            paste::paste! {
                $(
                    // test "happy path" take, valid indices on valid array
                    #[test]
                    #[allow(clippy::cast_possible_truncation)]
                    fn [<test_avx2_take_simple_ $IDX _ $VAL>]() {
                        let values: Vec<$VAL> = (1..=127).map(|x| x as $VAL).collect();
                        let indices: Vec<$IDX> = (0..127).collect();

                        let result = unsafe { take_avx2(&values, &indices) };
                        assert_eq!(&values, result.as_slice());
                    }

                    // test take on empty array
                    #[test]
                    #[should_panic]
                    #[allow(clippy::cast_possible_truncation)]
                    fn [<test_avx2_take_empty_ $IDX _ $VAL>]() {
                        let values: Vec<$VAL> = vec![];
                        let indices: Vec<$IDX> = (0..127).collect();
                        let result = unsafe { take_avx2(&values, &indices) };
                        assert!(result.is_empty());
                    }

                    // test all invalid take indices mapping to zeros
                    #[test]
                    #[should_panic]
                    #[allow(clippy::cast_possible_truncation)]
                    fn [<test_avx2_take_invalid_ $IDX _ $VAL>]() {
                        let values: Vec<$VAL> = (1..=127).map(|x| x as $VAL).collect();
                        // all out of bounds indices
                        let indices: Vec<$IDX> = (127..=254).collect();

                        let result = unsafe { take_avx2(&values, &indices) };
                        assert_eq!(&[0 as $VAL; 127], result.as_slice());
                    }
                )+
            }
        };
    }

    test_cases!(
        index_type => u8,
        value_types => u32, i32, u64, i64, f32, f64
    );
    test_cases!(
        index_type => u16,
        value_types => u32, i32, u64, i64, f32, f64
    );
    test_cases!(
        index_type => u32,
        value_types => u32, i32, u64, i64, f32, f64
    );
    test_cases!(
        index_type => u64,
        value_types => u32, i32, u64, i64, f32, f64
    );
}
