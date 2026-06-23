// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An AVX2 implementation of take operation using gather instructions.
//!
//! Only enabled for x86_64 hosts and it is gated at runtime behind feature detection to
//! ensure AVX2 instructions are available.

#![cfg(any(target_arch = "x86_64", target_arch = "x86"))]

use std::arch::x86_64::__m256i;
use std::arch::x86_64::_mm_loadu_si128;
use std::arch::x86_64::_mm_setzero_si128;
use std::arch::x86_64::_mm_shuffle_epi32;
use std::arch::x86_64::_mm_storeu_si128;
use std::arch::x86_64::_mm_unpacklo_epi64;
use std::arch::x86_64::_mm256_andnot_si256;
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
use std::arch::x86_64::_mm256_set1_epi32;
use std::arch::x86_64::_mm256_set1_epi64x;
use std::arch::x86_64::_mm256_setzero_si256;
use std::arch::x86_64::_mm256_storeu_si256;
use std::convert::identity;

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::compute::take::TakeImpl;
use crate::arrays::primitive::compute::take::take_primitive_scalar;
use crate::arrays::primitive::vtable::Primitive;
use crate::dtype::NativePType;
use crate::dtype::UnsignedPType;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::validity::Validity;

#[allow(unused)]
pub(super) struct TakeKernelAVX2;

impl TakeImpl for TakeKernelAVX2 {
    #[inline(always)]
    fn take(
        &self,
        values: ArrayView<'_, Primitive>,
        indices: ArrayView<'_, Primitive>,
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        assert!(indices.ptype().is_unsigned_int());

        Ok(match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
            match_each_native_ptype!(values.ptype(), |V| {
                // SAFETY: This kernel is only selected when avx2 cpu-feature is detected.
                unsafe {
                    take_primitive_avx2(values.as_slice::<V>(), indices.as_slice::<I>(), validity)
                }
            })
        })
        .into_array())
    }
}

/// # Safety
///
/// The caller must ensure that if the validity has a length, it is the same length as the indices,
/// and that the `avx2` feature is enabled.
#[target_feature(enable = "avx2")]
#[allow(unused)]
unsafe fn take_primitive_avx2<V, I>(
    values: &[V],
    indices: &[I],
    validity: Validity,
) -> PrimitiveArray
where
    V: NativePType,
    I: UnsignedPType,
{
    // SAFETY: The caller guarantees that the `avx2` feature is enabled.
    let buffer = unsafe { take_avx2(values, indices) };

    debug_assert!(
        validity
            .maybe_len()
            .is_none_or(|validity_len| validity_len == buffer.len())
    );

    // SAFETY: The caller ensures that the validity and indices have the same length, so the taken
    // buffer and the validity must have the same length.
    unsafe { PrimitiveArray::new_unchecked(buffer, validity) }
}

// ---------------------------------------------------------------------------
// AVX2 SIMD take algorithm
// ---------------------------------------------------------------------------

/// Takes the specified indices into a new [`Buffer`] using AVX2 SIMD.
///
/// An AVX2 gather only moves raw bytes, so signedness and float-ness are irrelevant — only the
/// byte width of `V` matters. Any 4-byte value rides the gather through the `u32` lane and any
/// 8-byte value through the `u64` lane, regardless of its actual type. Values 1 or 2 bytes wide
/// (AVX2 has no sub-32-bit gather) and wider than 8 bytes (`i128`, decimals) fall back to the
/// scalar kernel.
///
/// This treats `V` as plain-old-data: reinterpreting the gathered bytes as `V` is only sound
/// because every bit pattern is a valid `V`. All primitive and decimal-backing types satisfy
/// this, as does any `Copy` POD type the caller supplies.
///
/// # Panics
///
/// This function panics if any of the provided `indices` are out of bounds for `values`.
///
/// # Safety
///
/// The caller must ensure the `avx2` feature is enabled.
#[target_feature(enable = "avx2")]
#[doc(hidden)]
unsafe fn take_avx2<V: Copy, I: UnsignedPType>(buffer: &[V], indices: &[I]) -> Buffer<V> {
    if buffer.is_empty() {
        return Buffer::zeroed(indices.len());
    }

    // Dispatch on the gather lane width. The index type must still be concretized to select the
    // right `GatherFn` impl, so re-dispatch it with `match_each_unsigned_integer_ptype!`.
    macro_rules! dispatch {
        ($lane:ty) => {{
            match_each_unsigned_integer_ptype!(I::PTYPE, |Idx| {
                // SAFETY: `Idx` has the same `PTYPE` as `I`, so this is a no-op reinterpret of the
                // index slice into the concrete type the gather impl is keyed on.
                let indices = unsafe { std::mem::transmute::<&[I], &[Idx]>(indices) };
                exec_take::<V, $lane, Idx, AVX2Gather>(buffer, indices)
            })
        }};
    }

    match size_of::<V>() {
        4 => dispatch!(u32),
        8 => dispatch!(u64),
        // 1/2-byte and >8-byte values have no AVX2 gather lane, so fall back to scalar.
        _ => take_primitive_scalar(buffer, indices),
    }
}

/// The main gather function that is used by the inner loop kernel for AVX2 gather.
trait GatherFn<Idx, Values> {
    /// The number of data elements that are written to the `dst` on each loop iteration.
    const WIDTH: usize;
    /// The number of indices read from `indices` on each loop iteration. Depending on the
    /// available instructions and bit-width we may stride by a larger amount than we actually
    /// end up reading from `src` (governed by the `WIDTH` parameter).
    const STRIDE: usize = Self::WIDTH;

    /// Gather values from `src` into the `dst` using the `indices`, optionally using SIMD
    /// instructions.
    ///
    /// # Safety
    ///
    /// This function can read up to `STRIDE` elements through `indices`, and read/write up to
    /// `WIDTH` elements through `src` and `dst` respectively.
    unsafe fn gather(indices: *const Idx, max_idx: Idx, src: *const Values, dst: *mut Values);
}

/// AVX2 version of [`GatherFn`] defined for 32- and 64-bit value types.
enum AVX2Gather {}

macro_rules! impl_gather {
    ($idx:ty, $({$value:ty => load: $load:ident, extend: $extend:ident, splat: $splat:ident, zero_vec: $zero_vec:ident, mask_indices: $mask_indices:ident, mask_cvt: |$mask_var:ident| $mask_cvt:block, gather: $masked_gather:ident, store: $store:ident, WIDTH = $WIDTH:literal, STRIDE = $STRIDE:literal }),+) => {
        $(
            impl_gather!(single; $idx, $value, load: $load, extend: $extend, splat: $splat, zero_vec: $zero_vec, mask_indices: $mask_indices, mask_cvt: |$mask_var| $mask_cvt, gather: $masked_gather, store: $store, WIDTH = $WIDTH, STRIDE = $STRIDE);
        )*
    };
    (single; $idx:ty, $value:ty, load: $load:ident, extend: $extend:ident, splat: $splat:ident, zero_vec: $zero_vec:ident, mask_indices: $mask_indices:ident, mask_cvt: |$mask_var:ident| $mask_cvt:block, gather: $masked_gather:ident, store: $store:ident, WIDTH = $WIDTH:literal, STRIDE = $STRIDE:literal) => {
            impl GatherFn<$idx, $value> for AVX2Gather {
                const WIDTH: usize = $WIDTH;
                const STRIDE: usize = $STRIDE;

                #[allow(unused_unsafe, clippy::cast_possible_truncation)]
                #[inline(always)]
                unsafe fn gather(indices: *const $idx, max_idx: $idx, src: *const $value, dst: *mut $value) {
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
                    let invalid_mask = unsafe { _mm256_andnot_si256($mask_indices(indices_vec, max_idx_vec), $splat(-1)) };
                    let invalid_mask = {
                        let $mask_var = invalid_mask;
                        $mask_cvt
                    };
                    let zero_vec = unsafe { $zero_vec() };

                    // Gather the values into new vector register, for masked positions
                    // it substitutes zero instead of accessing the src.
                    let values_vec = unsafe { $masked_gather::<SCALE>(zero_vec, src.cast(), indices_vec, invalid_mask) };

                    // Write the vec out to dst.
                    unsafe { $store(dst.cast(), values_vec) };
                }
            }
    };
}

// kernels for u8 indices
impl_gather!(u8,
    // 32-bit values, loaded 8 at a time
    { u32 =>
        load: _mm_loadu_si128,
        extend: _mm256_cvtepu8_epi32,
        splat: _mm256_set1_epi32,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi32,
        mask_cvt: |x| { x },
        gather: _mm256_mask_i32gather_epi32,
        store: _mm256_storeu_si256,
        WIDTH = 8, STRIDE = 16
    },

    // 64-bit values, loaded 4 at a time
    { u64 =>
        load: _mm_loadu_si128,
        extend: _mm256_cvtepu8_epi64,
        splat: _mm256_set1_epi64x,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi64,
        mask_cvt: |x| { x },
        gather: _mm256_mask_i64gather_epi64,
        store: _mm256_storeu_si256,
        WIDTH = 4, STRIDE = 16
    }
);

// kernels for u16 indices
impl_gather!(u16,
    // 32-bit values. 8x indices loaded at a time and 8x values written at a time.
    { u32 =>
        load: _mm_loadu_si128,
        extend: _mm256_cvtepu16_epi32,
        splat: _mm256_set1_epi32,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi32,
        mask_cvt: |x| { x },
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
        gather: _mm256_mask_i64gather_epi64,
        store: _mm256_storeu_si256,
        WIDTH = 4, STRIDE = 8
    }
);

// kernels for u32 indices
impl_gather!(u32,
    // 32-bit values. 8x indices loaded at a time and 8x values written.
    { u32 =>
        load: _mm256_loadu_si256,
        extend: identity,
        splat: _mm256_set1_epi32,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi32,
        mask_cvt: |x| { x },
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
        gather: _mm256_mask_i64gather_epi64,
        store: _mm256_storeu_si256,
        WIDTH = 4, STRIDE = 4
    }
);

// kernels for u64 indices
impl_gather!(u64,
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
        gather: _mm256_mask_i64gather_epi64,
        store: _mm256_storeu_si256,
        WIDTH = 4, STRIDE = 4
    }
);

/// AVX2 core inner loop for a given index type `Idx`, output element type `Out`, and gather
/// `Lane` type.
///
/// `Out` is the element type written to the output buffer; `Lane` (`u32` or `u64`) is the
/// integer type the gather intrinsics operate on. The caller must pair them so that
/// `size_of::<Out>() == size_of::<Lane>()` (the only caller, [`take_avx2`], picks `Lane` from
/// `size_of::<Out>()`). The gather moves `size_of::<Lane>()` raw bytes per element, which only
/// yields a valid `Out` because `Out` is plain-old-data (every bit pattern is a valid `Out`).
/// Pointers into the `Out`-typed slices are cast to `*const Lane`/`*mut Lane`; gather tolerates
/// the (possibly weaker) `Out` alignment.
#[inline(always)]
fn exec_take<Out, Lane, Idx, Gather>(values: &[Out], indices: &[Idx]) -> Buffer<Out>
where
    Out: Copy,
    Idx: UnsignedPType,
    Gather: GatherFn<Idx, Lane>,
{
    debug_assert_eq!(
        size_of::<Out>(),
        size_of::<Lane>(),
        "gather lane and output element must have the same size"
    );

    let indices_len = indices.len();
    let max_index = Idx::from(values.len()).unwrap_or_else(|| Idx::max_value());
    let mut buffer =
        BufferMut::<Out>::with_capacity_aligned(indices_len, Alignment::of::<__m256i>());
    let buf_uninit = buffer.spare_capacity_mut();

    let mut offset = 0;
    // Loop terminates STRIDE elements before end of the indices array because the `GatherFn`
    // might read up to STRIDE src elements at a time, even though it only advances WIDTH elements
    // in the dst.
    while offset + Gather::STRIDE < indices_len {
        // SAFETY: `gather` preconditions satisfied:
        //  1. `(indices + offset)..(indices + offset + STRIDE)` is in-bounds for indices
        //     allocation.
        //  2. `buffer` has same len as indices so `buffer + offset + WIDTH` is always valid.
        //  3. `size_of::<Out>() == size_of::<Lane>()` (asserted above), so the `Lane`-typed
        //     pointers address the same bytes as the `Out`-typed `values`/`buffer` allocations.
        unsafe {
            Gather::gather(
                indices.as_ptr().add(offset),
                max_index,
                values.as_ptr().cast::<Lane>(),
                buf_uninit.as_mut_ptr().add(offset).cast::<Lane>(),
            )
        };
        offset += Gather::WIDTH;
    }

    // Remainder.
    while offset < indices_len {
        buf_uninit[offset].write(values[indices[offset].as_()]);
        offset += 1;
    }

    assert_eq!(offset, indices_len);

    // SAFETY: All elements have been initialized.
    unsafe { buffer.set_len(indices_len) };

    // Reset the buffer alignment to the output type.
    // NOTE: if we don't do this, we pass back a Buffer which is over-aligned to the SIMD
    // register width. The caller expects that this memory should be aligned to the value type
    // so that we can slice it at value boundaries.
    buffer = buffer.aligned(Alignment::of::<Out>());

    buffer.freeze()
}

#[cfg(test)]
#[cfg_attr(miri, ignore)]
#[cfg(target_arch = "x86_64")]
mod avx2_tests {
    use super::*;

    macro_rules! test_cases {
        (index_type => $IDX:ty, value_types => $($VAL:ty),+) => {
            paste::paste! {
                $(
                    // Test "happy path" take, valid indices on valid array.
                    #[test]
                    #[allow(clippy::cast_possible_truncation)]
                    fn [<test_avx2_take_simple_ $IDX _ $VAL>]() {
                        let values: Vec<$VAL> = (1..=127).map(|x| x as $VAL).collect();
                        let indices: Vec<$IDX> = (0..127).collect();

                        let result = unsafe { take_avx2(&values, &indices) };
                        assert_eq!(&values, result.as_slice());
                    }

                    // Test take on empty array.
                    #[test]
                    #[should_panic]
                    #[allow(clippy::cast_possible_truncation)]
                    fn [<test_avx2_take_empty_ $IDX _ $VAL>]() {
                        let values: Vec<$VAL> = vec![];
                        let indices: Vec<$IDX> = (0..127).collect();
                        let result = unsafe { take_avx2(&values, &indices) };
                        assert!(result.is_empty());
                    }

                    // Test all invalid take indices mapping to zeros.
                    #[test]
                    #[should_panic]
                    #[allow(clippy::cast_possible_truncation)]
                    fn [<test_avx2_take_invalid_ $IDX _ $VAL>]() {
                        let values: Vec<$VAL> = (1..=127).map(|x| x as $VAL).collect();
                        // All out of bounds indices.
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

    #[test]
    fn test_avx2_take_last_valid_index_u8() {
        let values: Vec<i64> = (0..(255 + 1)).collect();
        let indices: Vec<u8> = vec![255; 20];

        let result = unsafe { take_avx2(&values, &indices) };
        assert_eq!(&vec![255; indices.len()], result.as_slice());
    }

    #[test]
    fn test_avx2_take_last_valid_index_u16() {
        let values: Vec<i64> = (0..(65535 + 1)).collect();
        let indices: Vec<u16> = vec![65535; 20];

        let result = unsafe { take_avx2(&values, &indices) };
        assert_eq!(&vec![65535; indices.len()], result.as_slice());
    }

    /// A `[u8; 4]` is a 4-byte `Copy` POD that is not a `NativePType`. This proves the kernel
    /// gathers an arbitrary 4-byte value type through the `u32` SIMD lane.
    #[test]
    fn test_avx2_take_simd_array_u8x4() {
        let values: Vec<[u8; 4]> = (1u32..=200).map(u32::to_le_bytes).collect();
        let indices: Vec<u32> = (0..200).collect();

        let result = unsafe { take_avx2(&values, &indices) };
        assert_eq!(values.as_slice(), result.as_slice());
    }

    /// 2-byte values have no AVX2 gather, so they take the scalar fallback path and must still be
    /// correct.
    #[test]
    fn test_avx2_take_scalar_fallback_u16() {
        let values: Vec<u16> = (1..=300).collect();
        let indices: Vec<u32> = (0..300).collect();

        let result = unsafe { take_avx2(&values, &indices) };
        assert_eq!(values.as_slice(), result.as_slice());
    }

    /// Values wider than 8 bytes (e.g. `i128`/decimal backing) exceed the gather lane and fall
    /// back to the scalar kernel.
    #[test]
    fn test_avx2_take_scalar_fallback_array_u8x16() {
        let values: Vec<[u8; 16]> = (0u128..200).map(u128::to_le_bytes).collect();
        let indices: Vec<u32> = (0..200).collect();

        let result = unsafe { take_avx2(&values, &indices) };
        assert_eq!(values.as_slice(), result.as_slice());
    }
}
