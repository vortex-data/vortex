// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::arch::x86_64::__m256i;
use std::arch::x86_64::_mm_loadu_si128;
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
use std::arch::x86_64::_mm256_set1_epi32;
use std::arch::x86_64::_mm256_set1_epi64x;
use std::arch::x86_64::_mm256_setzero_si256;
use std::arch::x86_64::_mm256_storeu_si256;
use std::arch::x86_64::_mm256_xor_si256;
use std::convert::identity;

/// Moves one SIMD block of fixed-width values into an output buffer.
pub(super) trait GatherFn<Index, Lane> {
    /// The number of data elements written on each iteration.
    const WIDTH: usize;
    /// The number of indices read on each iteration.
    const STRIDE: usize = Self::WIDTH;

    /// Gather values from `src` into `dst`.
    ///
    /// `max_idx` is the exclusive upper bound on valid indices. `None` means the bound does not
    /// fit in the index type, so every representable index is in-bounds.
    ///
    /// # Safety
    ///
    /// `indices` must be readable for `STRIDE` elements. `src` must contain `max_idx` elements,
    /// or more than `Index::MAX` elements when `max_idx` is `None`, and `dst` must be writable
    /// for `WIDTH` elements.
    ///
    /// Returns a vector mask with all lanes set when every gathered index was in-bounds.
    unsafe fn gather(
        indices: *const Index,
        max_idx: Option<Index>,
        src: *const Lane,
        dst: *mut Lane,
    ) -> __m256i;
}

/// AVX2 gather implementations for 32- and 64-bit value lanes.
pub(super) enum Avx2Gather {}

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
        impl GatherFn<$idx, $value> for Avx2Gather {
            const WIDTH: usize = $WIDTH;
            const STRIDE: usize = $STRIDE;

            #[allow(unused_unsafe, clippy::cast_possible_truncation)]
            #[inline(always)]
            unsafe fn gather(
                indices: *const $idx,
                max_idx: Option<$idx>,
                src: *const $value,
                dst: *mut $value,
            ) -> __m256i {
                const {
                    assert!($WIDTH <= $STRIDE, "dst cannot advance by more than the stride");
                }

                const SCALE: i32 = std::mem::size_of::<$value>() as i32;

                let indices_vec = unsafe { $load(indices.cast()) };
                let indices_vec = unsafe { $extend(indices_vec) };
                // An absent exclusive bound means it does not fit in the index type, so every
                // representable index is in-bounds. Passing the valid mask to the gather masks
                // every invalid lane before it can access `src`.
                let valid_mask = if let Some(max_idx) = max_idx {
                    let max_idx_vec = unsafe { $splat(max_idx as _) };
                    unsafe { $mask_indices!(max_idx_vec, indices_vec) }
                } else {
                    unsafe { $splat(-1) }
                };
                let gather_mask = {
                    let $mask_var = valid_mask;
                    $mask_cvt
                };
                let values_vec = unsafe {
                    $masked_gather::<SCALE>(
                        $zero_vec(),
                        src.cast(),
                        indices_vec,
                        gather_mask,
                    )
                };
                unsafe { $store(dst.cast(), values_vec) };
                valid_mask
            }
        }
    };
}

impl_gather!(u8,
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

impl_gather!(u16,
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

impl_gather!(u32,
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

impl_gather!(u64,
    { u32 =>
        load: _mm256_loadu_si256,
        extend: identity,
        splat: _mm256_set1_epi64x,
        zero_vec: _mm_setzero_si128,
        mask_indices: cmpgt_epu64,
        mask_cvt: |mask| { unsafe { pack_i64_mask_for_i32_gather!(mask) } },
        gather: _mm256_mask_i64gather_epi32,
        store: _mm_storeu_si128,
        WIDTH = 4, STRIDE = 4
    },
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

#[cfg(test)]
mod tests {
    use std::arch::x86_64::_mm_movemask_epi8;
    use std::arch::x86_64::_mm_set_epi64x;

    use super::*;

    #[test]
    fn pack_i64_mask_for_i32_gather_preserves_lanes() {
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
}
