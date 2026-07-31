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
use std::arch::x86_64::_mm256_and_si256;
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
use std::arch::x86_64::_mm256_xor_si256;
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
use crate::dtype::PType;
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
/// The gather copies the bytes of an existing `V` without changing them. Invalid lanes write zero
/// into uninitialized output memory, but that memory is never exposed because the function panics
/// before setting the output length.
///
/// # Panics
///
/// This function panics if any of the provided `indices` are out of bounds for `values`.
///
/// # Safety
///
/// The caller must ensure the `avx2` feature is enabled. Four- and eight-byte `V` types must not
/// contain uninitialized padding because the gather reads their entire object representation as an
/// integer lane. The production caller only supplies [`NativePType`] values, which satisfy this.
#[target_feature(enable = "avx2")]
#[doc(hidden)]
unsafe fn take_avx2<V: Copy, I: UnsignedPType>(buffer: &[V], indices: &[I]) -> Buffer<V> {
    if buffer.is_empty() {
        assert!(indices.is_empty(), "take index out of bounds");
        return Buffer::empty();
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
        // The i32 gather interprets u32 lanes as signed offsets. If the values slice is longer
        // than its non-negative addressable range, a valid high u32 index must use the scalar path
        // instead.
        4 if I::PTYPE == PType::U32 && !i32_gather_can_address(buffer.len()) => {
            take_primitive_scalar(buffer, indices)
        }
        4 => dispatch!(u32),
        8 => dispatch!(u64),
        // 1/2-byte and >8-byte values have no AVX2 gather lane, so fall back to scalar.
        _ => take_primitive_scalar(buffer, indices),
    }
}

const fn i32_gather_can_address(values_len: usize) -> bool {
    values_len <= i32::MAX as usize + 1
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
    ///
    /// Returns a vector mask with all lanes set when every gathered index is valid.
    unsafe fn gather(
        indices: *const Idx,
        max_idx: usize,
        src: *const Values,
        dst: *mut Values,
    ) -> __m256i;
}

/// AVX2 version of [`GatherFn`] defined for 32- and 64-bit value types.
enum AVX2Gather {}

macro_rules! cmpgt_epu32 {
    ($lhs:expr, $rhs:expr) => {{
        // AVX2 only supplies a signed integer comparison. XORing each lane with its sign bit
        // maps the unsigned ordering into the signed ordering without changing the relative
        // order.
        let sign_bit = _mm256_set1_epi32(i32::MIN);
        _mm256_cmpgt_epi32(
            _mm256_xor_si256($lhs, sign_bit),
            _mm256_xor_si256($rhs, sign_bit),
        )
    }};
}

macro_rules! cmpgt_epu64 {
    ($lhs:expr, $rhs:expr) => {{
        // AVX2 only supplies a signed integer comparison. XORing each lane with its sign bit
        // maps the unsigned ordering into the signed ordering without changing the relative
        // order.
        let sign_bit = _mm256_set1_epi64x(i64::MIN);
        _mm256_cmpgt_epi64(
            _mm256_xor_si256($lhs, sign_bit),
            _mm256_xor_si256($rhs, sign_bit),
        )
    }};
}

macro_rules! pack_i64_mask_for_i32_gather {
    ($mask:expr) => {{
        let lo_bits = _mm256_extracti128_si256::<0>($mask);
        let hi_bits = _mm256_extracti128_si256::<1>($mask);
        let lo_packed = pack_i64_mask_half_for_i32_gather!(lo_bits);
        let hi_packed = pack_i64_mask_half_for_i32_gather!(hi_bits);
        _mm_unpacklo_epi64(lo_packed, hi_packed)
    }};
}

macro_rules! pack_i64_mask_half_for_i32_gather {
    ($mask:expr) => {
        _mm_shuffle_epi32::<0b11_01_11_01>($mask)
    };
}

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
                unsafe fn gather(indices: *const $idx, max_idx: usize, src: *const $value, dst: *mut $value) -> __m256i {
                    const {
                        assert!($WIDTH <= $STRIDE, "dst cannot advance by more than the stride");
                    }

                    const SCALE: i32 = std::mem::size_of::<$value>() as i32;

                    let indices_vec = unsafe { $load(indices.cast()) };
                    // Extend indices to fill vector register.
                    let indices_vec = unsafe { $extend(indices_vec) };

                    let max_idx_vec = unsafe { $splat(max_idx as _) };
                    // Passing the valid mask to the gather masks every invalid lane before it can
                    // access `src`.
                    let valid_mask = unsafe { $mask_indices!(max_idx_vec, indices_vec) };
                    let $mask_var = valid_mask;
                    let gather_mask = $mask_cvt;
                    let zero_vec = unsafe { $zero_vec() };

                    // Gather the values into new vector register, for masked positions
                    // it substitutes zero instead of accessing the src.
                    let values_vec = unsafe { $masked_gather::<SCALE>(zero_vec, src.cast(), indices_vec, gather_mask) };

                    // Write the vec out to dst.
                    unsafe { $store(dst.cast(), values_vec) };

                    valid_mask
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
        mask_indices: cmpgt_epu32,
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
        mask_indices: cmpgt_epu64,
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
        mask_indices: cmpgt_epu32,
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
        mask_indices: cmpgt_epu64,
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
        mask_indices: cmpgt_epu32,
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
        mask_indices: cmpgt_epu64,
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
        mask_indices: cmpgt_epu64,
        mask_cvt: |m| { unsafe { pack_i64_mask_for_i32_gather!(m) } },
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
        mask_indices: cmpgt_epu64,
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
/// `size_of::<Out>()`). The gather copies each valid `Out` object's bytes unchanged. Pointers into
/// the `Out`-typed slices are cast to `*const Lane`/`*mut Lane`; gather tolerates the (possibly
/// weaker) `Out` alignment.
#[inline(always)]
fn exec_take<Out, Lane, Idx, Gather>(values: &[Out], indices: &[Idx]) -> Buffer<Out>
where
    Out: Copy,
    Idx: UnsignedPType,
    Gather: GatherFn<Idx, Lane>,
{
    assert_eq!(
        size_of::<Out>(),
        size_of::<Lane>(),
        "gather lane and output element must have the same size"
    );

    let indices_len = indices.len();
    // Lift the representability branch out of the SIMD loop. When the length does not fit in the
    // index type, max + 1 is a widened exclusive bound that accepts every possible index.
    let max_index: usize = if Idx::from(values.len()).is_some() {
        values.len()
    } else {
        Idx::max_value().as_() + 1
    };
    let mut buffer =
        BufferMut::<Out>::with_capacity_aligned(indices_len, Alignment::of::<__m256i>());
    let buf_uninit = buffer.spare_capacity_mut();

    let mut offset = 0;
    // SAFETY: `exec_take` is only called by `take_avx2`, whose caller guarantees AVX2 support.
    let mut all_indices_valid = unsafe { _mm256_set1_epi32(-1) };
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
        let valid_mask = unsafe {
            Gather::gather(
                indices.as_ptr().add(offset),
                max_index,
                values.as_ptr().cast::<Lane>(),
                buf_uninit.as_mut_ptr().add(offset).cast::<Lane>(),
            )
        };
        // SAFETY: `exec_take` is only called by `take_avx2`, whose caller guarantees AVX2 support.
        all_indices_valid = unsafe { _mm256_and_si256(all_indices_valid, valid_mask) };
        offset += Gather::WIDTH;
    }

    // Invalid lanes were masked before gathering, so it is safe to defer the bounds failure until
    // after the SIMD loop and avoid a conditional branch on every iteration.
    assert!(
        // SAFETY: `exec_take` is only called by `take_avx2`, whose caller guarantees AVX2 support.
        unsafe { _mm256_movemask_epi8(all_indices_valid) } == -1,
        "take index out of bounds"
    );

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
    use std::arch::x86_64::_mm_movemask_epi8;
    use std::arch::x86_64::_mm_set_epi64x;
    use std::panic::RefUnwindSafe;
    use std::panic::catch_unwind;

    use super::*;

    fn take_avx2_if_supported<V: Copy, I: UnsignedPType>(
        values: &[V],
        indices: &[I],
    ) -> Option<Buffer<V>> {
        if !is_x86_feature_detected!("avx2") {
            return None;
        }

        // SAFETY: AVX2 support was detected above, and every test value type has no uninitialized
        // padding.
        Some(unsafe { take_avx2(values, indices) })
    }

    fn assert_avx2_take_panics<V, I>(values: &[V], indices: &[I])
    where
        V: Copy + RefUnwindSafe,
        I: UnsignedPType + RefUnwindSafe,
    {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        // SAFETY: AVX2 support was detected above, and every test value type has no uninitialized
        // padding.
        let result = catch_unwind(|| unsafe { take_avx2(values, indices) });
        let Err(payload) = result else {
            panic!("take should panic for an invalid index");
        };
        let message = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str));
        assert_eq!(message, Some("take index out of bounds"));
    }

    #[test]
    fn test_pack_i64_mask_for_i32_gather_preserves_lanes() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }

        for lane_bits in 0u8..16 {
            let lane_mask = |lane: u32| {
                if lane_bits & (1u8 << lane) == 0 {
                    0
                } else {
                    -1
                }
            };
            let actual = unsafe {
                let lo_bits = _mm_set_epi64x(lane_mask(1), lane_mask(0));
                let hi_bits = _mm_set_epi64x(lane_mask(3), lane_mask(2));
                let lo_packed = pack_i64_mask_half_for_i32_gather!(lo_bits);
                let hi_packed = pack_i64_mask_half_for_i32_gather!(hi_bits);
                let packed = _mm_unpacklo_epi64(lo_packed, hi_packed);
                _mm_movemask_epi8(packed)
            };
            let expected = (0u32..4).fold(0, |bits, lane| {
                bits | (((lane_bits >> lane) & 1) as i32 * (0b1111 << (lane * 4)))
            });

            assert_eq!(actual, expected, "lane mask {lane_bits:04b}");
        }
    }

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

                        let Some(result) = take_avx2_if_supported(&values, &indices) else {
                            return;
                        };
                        assert_eq!(&values, result.as_slice());
                    }

                    // Test take on empty array.
                    #[test]
                    #[allow(clippy::cast_possible_truncation)]
                    fn [<test_avx2_take_empty_ $IDX _ $VAL>]() {
                        let values: Vec<$VAL> = vec![];
                        let indices: Vec<$IDX> = (0..127).collect();

                        assert_avx2_take_panics(&values, &indices);
                    }

                    // Test all invalid take indices mapping to zeros.
                    #[test]
                    #[allow(clippy::cast_possible_truncation)]
                    fn [<test_avx2_take_invalid_ $IDX _ $VAL>]() {
                        let values: Vec<$VAL> = (1..=127).map(|x| x as $VAL).collect();
                        // All out of bounds indices.
                        let indices: Vec<$IDX> = (127..=254).collect();

                        assert_avx2_take_panics(&values, &indices);
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

        let Some(result) = take_avx2_if_supported(&values, &indices) else {
            return;
        };
        assert_eq!(&vec![255; indices.len()], result.as_slice());
    }

    #[test]
    fn test_avx2_take_last_valid_index_u16() {
        let values: Vec<i64> = (0..(65535 + 1)).collect();
        let indices: Vec<u16> = vec![65535; 20];

        let Some(result) = take_avx2_if_supported(&values, &indices) else {
            return;
        };
        assert_eq!(&vec![65535; indices.len()], result.as_slice());
    }

    #[test]
    fn test_avx2_take_empty_values_and_indices() {
        let Some(result) = take_avx2_if_supported::<u32, u32>(&[], &[]) else {
            return;
        };
        assert!(result.is_empty());
    }

    #[test]
    fn test_i32_gather_addressable_length_boundary() {
        assert!(i32_gather_can_address(i32::MAX as usize + 1));
        assert!(!i32_gather_can_address(i32::MAX as usize + 2));
    }

    #[test]
    fn test_avx2_take_u32_max_index_u32_lane() {
        let values = vec![0u32; 8];
        // The first eight indices execute in the SIMD loop; the scalar remainder is valid.
        let indices = vec![0, u32::MAX, 2, 3, 4, 5, 6, 7, 0];

        assert_avx2_take_panics(&values, &indices);
    }

    #[test]
    fn test_avx2_take_u64_max_index_u32_lane() {
        let values = vec![0u32; 8];
        // The first four indices execute in the SIMD loop; the scalar remainder is valid.
        let indices = vec![0, u64::MAX, 2, 3, 0];

        assert_avx2_take_panics(&values, &indices);
    }

    #[test]
    fn test_avx2_take_u64_max_index_u64_lane() {
        let values = vec![0u64; 8];
        // The first four indices execute in the SIMD loop; the scalar remainder is valid.
        let indices = vec![0, u64::MAX, 2, 3, 0];

        assert_avx2_take_panics(&values, &indices);
    }

    /// A `[u8; 4]` is a 4-byte `Copy` POD that is not a `NativePType`. This proves the kernel
    /// gathers an arbitrary 4-byte value type through the `u32` SIMD lane.
    #[test]
    fn test_avx2_take_simd_array_u8x4() {
        let values: Vec<[u8; 4]> = (1u32..=200).map(u32::to_le_bytes).collect();
        let indices: Vec<u32> = (0..200).collect();

        let Some(result) = take_avx2_if_supported(&values, &indices) else {
            return;
        };
        assert_eq!(values.as_slice(), result.as_slice());
    }

    /// 2-byte values have no AVX2 gather, so they take the scalar fallback path and must still be
    /// correct.
    #[test]
    fn test_avx2_take_scalar_fallback_u16() {
        let values: Vec<u16> = (1..=300).collect();
        let indices: Vec<u32> = (0..300).collect();

        let Some(result) = take_avx2_if_supported(&values, &indices) else {
            return;
        };
        assert_eq!(values.as_slice(), result.as_slice());
    }

    /// Values wider than 8 bytes (e.g. `i128`/decimal backing) exceed the gather lane and fall
    /// back to the scalar kernel.
    #[test]
    fn test_avx2_take_scalar_fallback_array_u8x16() {
        let values: Vec<[u8; 16]> = (0u128..200).map(u128::to_le_bytes).collect();
        let indices: Vec<u32> = (0..200).collect();

        let Some(result) = take_avx2_if_supported(&values, &indices) else {
            return;
        };
        assert_eq!(values.as_slice(), result.as_slice());
    }
}
