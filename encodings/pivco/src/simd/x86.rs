// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-FileCopyrightText: 2026 Marcin Żukowski
// SPDX-FileComment: Derived from pivco-huffman at commit 16b48fcf at https://github.com/MarcinZukowski/pivco-huffman

//! `x86_64` runtime-dispatched SSSE3 / SSE4.1 / AVX2 SIMD kernels for `vortex-pivco`.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use core::arch::x86_64::__m128i;
use core::arch::x86_64::__m256i;
use core::arch::x86_64::_mm_add_epi8;
use core::arch::x86_64::_mm_and_si128;
use core::arch::x86_64::_mm_andnot_si128;
use core::arch::x86_64::_mm_blendv_epi8;
use core::arch::x86_64::_mm_cmpeq_epi8;
use core::arch::x86_64::_mm_cvtsi32_si128;
use core::arch::x86_64::_mm_load_si128;
use core::arch::x86_64::_mm_loadl_epi64;
use core::arch::x86_64::_mm_loadu_si128;
use core::arch::x86_64::_mm_min_epu8;
use core::arch::x86_64::_mm_movemask_epi8;
use core::arch::x86_64::_mm_mullo_epi16;
use core::arch::x86_64::_mm_or_si128;
use core::arch::x86_64::_mm_packus_epi16;
use core::arch::x86_64::_mm_set1_epi8;
use core::arch::x86_64::_mm_set1_epi32;
use core::arch::x86_64::_mm_setr_epi8;
use core::arch::x86_64::_mm_setr_epi16;
use core::arch::x86_64::_mm_setr_epi32;
use core::arch::x86_64::_mm_setzero_si128;
use core::arch::x86_64::_mm_shuffle_epi8;
use core::arch::x86_64::_mm_slli_epi16;
use core::arch::x86_64::_mm_slli_si128;
use core::arch::x86_64::_mm_srli_epi16;
use core::arch::x86_64::_mm_srli_si128;
use core::arch::x86_64::_mm_srlv_epi32;
use core::arch::x86_64::_mm_storel_epi64;
use core::arch::x86_64::_mm_storeu_si128;
use core::arch::x86_64::_mm_unpackhi_epi8;
use core::arch::x86_64::_mm_unpackhi_epi16;
use core::arch::x86_64::_mm_unpacklo_epi8;
use core::arch::x86_64::_mm_unpacklo_epi16;
use core::arch::x86_64::_mm_xor_si128;
use core::arch::x86_64::_mm256_add_epi8;
use core::arch::x86_64::_mm256_and_si256;
use core::arch::x86_64::_mm256_andnot_si256;
use core::arch::x86_64::_mm256_blend_epi32;
use core::arch::x86_64::_mm256_blendv_epi8;
use core::arch::x86_64::_mm256_broadcastq_epi64;
use core::arch::x86_64::_mm256_broadcastsi128_si256;
use core::arch::x86_64::_mm256_castsi128_si256;
use core::arch::x86_64::_mm256_castsi256_si128;
use core::arch::x86_64::_mm256_cmpeq_epi8;
use core::arch::x86_64::_mm256_extracti128_si256;
use core::arch::x86_64::_mm256_inserti128_si256;
use core::arch::x86_64::_mm256_loadu_si256;
use core::arch::x86_64::_mm256_madd_epi16;
use core::arch::x86_64::_mm256_maddubs_epi16;
use core::arch::x86_64::_mm256_movemask_epi8;
use core::arch::x86_64::_mm256_or_si256;
use core::arch::x86_64::_mm256_permute2x128_si256;
use core::arch::x86_64::_mm256_set1_epi8;
use core::arch::x86_64::_mm256_set1_epi16;
use core::arch::x86_64::_mm256_set1_epi32;
use core::arch::x86_64::_mm256_set1_epi64x;
use core::arch::x86_64::_mm256_setr_epi8;
use core::arch::x86_64::_mm256_setzero_si256;
use core::arch::x86_64::_mm256_shuffle_epi8;
use core::arch::x86_64::_mm256_srli_epi16;
use core::arch::x86_64::_mm256_srli_epi64;
use core::arch::x86_64::_mm256_storeu_si256;
use core::arch::x86_64::_mm256_sub_epi8;
use core::arch::x86_64::_mm256_unpackhi_epi8;
use core::arch::x86_64::_mm256_unpacklo_epi8;
use core::arch::x86_64::_mm256_xor_si256;
use core::arch::x86_64::_popcnt64;
use std::sync::LazyLock;

use super::COMPRESS_LEFT_TAB;
use super::COMPRESS_RIGHT_TAB;
use super::POPCNT8;
use super::X86_MERGE_SHUF0;
use super::X86_MERGE_SHUF1;
use super::merge_cst_side_tail_raw;
use super::merge_vec_vec_tail_raw;
use super::unpack_flat_tail_scalar;

#[derive(Clone, Copy)]
pub(super) struct X86Features {
    pub(super) popcnt: bool,
    pub(super) ssse3: bool,
    pub(super) sse41: bool,
    pub(super) avx2: bool,
}

static X86_FEATURES: LazyLock<X86Features> = LazyLock::new(|| X86Features {
    popcnt: std::arch::is_x86_feature_detected!("popcnt"),
    ssse3: std::arch::is_x86_feature_detected!("ssse3"),
    sse41: std::arch::is_x86_feature_detected!("sse4.1"),
    avx2: std::arch::is_x86_feature_detected!("avx2"),
});

#[inline]
pub(super) fn features() -> X86Features {
    *X86_FEATURES
}

/// 4-way unrolled `POPCNT` (`x86.h:68-105`).
#[target_feature(enable = "popcnt")]
pub(super) unsafe fn popcount_bits_x86(bm: &[u8], k: usize) -> usize {
    let full_bytes = k >> 3;
    let partial_bits = k & 7;
    let bm_ptr = bm.as_ptr();
    let mut pos = 0usize;
    let mut a0 = 0usize;
    let mut a1 = 0usize;
    let mut a2 = 0usize;
    let mut a3 = 0usize;

    while pos + 32 <= full_bytes {
        let v0 = unsafe { bm_ptr.add(pos).cast::<u64>().read_unaligned() };
        let v1 = unsafe { bm_ptr.add(pos + 8).cast::<u64>().read_unaligned() };
        let v2 = unsafe { bm_ptr.add(pos + 16).cast::<u64>().read_unaligned() };
        let v3 = unsafe { bm_ptr.add(pos + 24).cast::<u64>().read_unaligned() };
        a0 += _popcnt64(v0 as i64) as usize;
        a1 += _popcnt64(v1 as i64) as usize;
        a2 += _popcnt64(v2 as i64) as usize;
        a3 += _popcnt64(v3 as i64) as usize;
        pos += 32;
    }
    let mut sum = a0 + a1 + a2 + a3;
    while pos + 8 <= full_bytes {
        let word = unsafe { bm_ptr.add(pos).cast::<u64>().read_unaligned() };
        sum += _popcnt64(word as i64) as usize;
        pos += 8;
    }
    while pos < full_bytes {
        sum += usize::from(POPCNT8[usize::from(bm[pos])]);
        pos += 1;
    }
    if partial_bits > 0 {
        let mask = (1u8 << partial_bits) - 1;
        sum += usize::from(POPCNT8[usize::from(bm[full_bytes] & mask)]);
    }
    sum
}

/// 16B PSHUFB-complement + OR merge (`x86.h:200-217`).
#[target_feature(enable = "ssse3")]
#[inline]
unsafe fn merge_vv_step16(dest: *mut u8, l_ptr: *const u8, r_ptr: *const u8, lo: usize, hi: usize) {
    let ones = _mm_set1_epi8(-1);
    let shuf0 = unsafe { _mm_load_si128(X86_MERGE_SHUF0.0[lo].as_ptr().cast()) };
    let shuf1 =
        _mm_slli_si128::<8>(unsafe { _mm_loadl_epi64(X86_MERGE_SHUF1.0[hi].as_ptr().cast()) });
    let merged = _mm_add_epi8(shuf0, shuf1);
    let vr = unsafe { _mm_loadu_si128(r_ptr.cast()) };
    let vl = unsafe { _mm_loadu_si128(l_ptr.cast()) };
    let rr = _mm_shuffle_epi8(vr, merged);
    let rl = _mm_shuffle_epi8(vl, _mm_xor_si128(merged, ones));
    unsafe { _mm_storeu_si128(dest.cast(), _mm_or_si128(rr, rl)) };
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_halves_256(s0: *const u8, s1: *const u8) -> __m256i {
    let lo = unsafe { _mm_loadu_si128(s0.cast()) };
    let hi = unsafe { _mm_loadu_si128(s1.cast()) };
    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn build_shuf32_avx2(m0: usize, m1: usize, m2: usize, m3: usize) -> __m256i {
    let zeros = _mm256_setzero_si256();
    let shuf02 = unsafe {
        load_halves_256(
            X86_MERGE_SHUF0.0[m0].as_ptr(),
            X86_MERGE_SHUF0.0[m2].as_ptr(),
        )
    };
    let shuf1 =
        _mm256_broadcastq_epi64(unsafe { _mm_loadl_epi64(X86_MERGE_SHUF1.0[m1].as_ptr().cast()) });
    let shuf3 =
        _mm256_broadcastq_epi64(unsafe { _mm_loadl_epi64(X86_MERGE_SHUF1.0[m3].as_ptr().cast()) });
    let shuf13 = _mm256_blend_epi32::<0xF0>(shuf1, shuf3);
    let shuf13_masked = _mm256_blend_epi32::<0xCC>(zeros, shuf13);
    _mm256_add_epi8(shuf02, shuf13_masked)
}

#[target_feature(enable = "ssse3,popcnt")]
pub(super) unsafe fn merge_vec_vec_ssse3(
    bm: &[u8],
    k: usize,
    l_base: *const u8,
    l_len: usize,
    r_base: *const u8,
    r_len: usize,
    out_ptr: *mut u8,
) {
    let mut lc = 0usize;
    let mut rc = 0usize;
    let mut idx = 0usize;
    let bm_ptr = bm.as_ptr();

    while idx + 16 <= k && lc + 16 <= l_len && rc + 16 <= r_len {
        let m16 = unsafe { bm_ptr.add(idx >> 3).cast::<u16>().read_unaligned() }.to_le() as usize;
        unsafe {
            merge_vv_step16(
                out_ptr.add(idx),
                l_base.add(lc),
                r_base.add(rc),
                m16 & 0xFF,
                m16 >> 8,
            );
        }
        let pop = _popcnt64(m16 as i64) as usize;
        rc += pop;
        lc += 16 - pop;
        idx += 16;
    }
    unsafe { merge_vec_vec_tail_raw(bm, idx, k, l_base, lc, r_base, rc, out_ptr) };
}

/// 32B/step AVX2 PSHUFB-complement + OR merge (`x86.h:149-173`).
#[target_feature(enable = "avx2,popcnt")]
pub(super) unsafe fn merge_vec_vec_avx2(
    bm: &[u8],
    k: usize,
    l_base: *const u8,
    l_len: usize,
    r_base: *const u8,
    r_len: usize,
    out_ptr: *mut u8,
) {
    let ones = _mm256_set1_epi8(-1);
    let mut lc = 0usize;
    let mut rc = 0usize;
    let mut idx = 0usize;
    let bm_ptr = bm.as_ptr();

    while idx + 32 <= k {
        let mask = unsafe { bm_ptr.add(idx >> 3).cast::<u32>().read_unaligned() }.to_le();
        let lo_pop = _popcnt64(i64::from(mask & 0xFFFF)) as usize;
        if lc + (16 - lo_pop) + 16 > l_len || rc + lo_pop + 16 > r_len {
            break;
        }
        let m0 = (mask & 0xFF) as usize;
        let m1 = ((mask >> 8) & 0xFF) as usize;
        let m2 = ((mask >> 16) & 0xFF) as usize;
        let m3 = ((mask >> 24) & 0xFF) as usize;
        let vshuf = unsafe { build_shuf32_avx2(m0, m1, m2, m3) };
        let vr = unsafe { load_halves_256(r_base.add(rc), r_base.add(rc + lo_pop)) };
        let vl = unsafe { load_halves_256(l_base.add(lc), l_base.add(lc + 16 - lo_pop)) };
        let rr = _mm256_shuffle_epi8(vr, vshuf);
        let rl = _mm256_shuffle_epi8(vl, _mm256_xor_si256(vshuf, ones));
        unsafe { _mm256_storeu_si256(out_ptr.add(idx).cast(), _mm256_or_si256(rl, rr)) };
        let pop = _popcnt64(i64::from(mask)) as usize;
        rc += pop;
        lc += 32 - pop;
        idx += 32;
    }
    while idx + 16 <= k && lc + 16 <= l_len && rc + 16 <= r_len {
        let m16 = unsafe { bm_ptr.add(idx >> 3).cast::<u16>().read_unaligned() }.to_le() as usize;
        unsafe {
            merge_vv_step16(
                out_ptr.add(idx),
                l_base.add(lc),
                r_base.add(rc),
                m16 & 0xFF,
                m16 >> 8,
            );
        }
        let pop = _popcnt64(m16 as i64) as usize;
        rc += pop;
        lc += 16 - pop;
        idx += 16;
    }
    unsafe { merge_vec_vec_tail_raw(bm, idx, k, l_base, lc, r_base, rc, out_ptr) };
}

#[target_feature(enable = "ssse3")]
#[inline]
unsafe fn merge_cv_step16(dest: *mut u8, vcst: __m128i, v_ptr: *const u8, lo: usize, hi: usize) {
    let ones = _mm_set1_epi8(-1);
    let shuf0 = unsafe { _mm_load_si128(X86_MERGE_SHUF0.0[lo].as_ptr().cast()) };
    let shuf1 =
        _mm_slli_si128::<8>(unsafe { _mm_loadl_epi64(X86_MERGE_SHUF1.0[hi].as_ptr().cast()) });
    let merged = _mm_add_epi8(shuf0, shuf1);
    let vv = unsafe { _mm_loadu_si128(v_ptr.cast()) };
    let rv = _mm_shuffle_epi8(vv, merged);
    let rc = _mm_shuffle_epi8(vcst, _mm_xor_si128(merged, ones));
    unsafe { _mm_storeu_si128(dest.cast(), _mm_or_si128(rv, rc)) };
}

/// Unified SSSE3 constant-plus-vector merge (`INVERT = false` => `cst_vec`, `true` => `vec_cst`).
#[target_feature(enable = "ssse3,popcnt")]
pub(super) unsafe fn merge_cst_side_ssse3<const INVERT: bool>(
    bm: &[u8],
    k: usize,
    cst_sym: u8,
    v_base: *const u8,
    v_len: usize,
    out_ptr: *mut u8,
) {
    let vcst = _mm_set1_epi8(cst_sym as i8);
    let mut vc = 0usize;
    let mut idx = 0usize;
    let bm_ptr = bm.as_ptr();

    while idx + 16 <= k && vc + 16 <= v_len {
        let raw_m16 = unsafe { bm_ptr.add(idx >> 3).cast::<u16>().read_unaligned() }.to_le();
        let m16 = usize::from(if INVERT { !raw_m16 } else { raw_m16 });
        unsafe {
            merge_cv_step16(out_ptr.add(idx), vcst, v_base.add(vc), m16 & 0xFF, m16 >> 8);
        }
        vc += _popcnt64(m16 as i64) as usize;
        idx += 16;
    }
    unsafe { merge_cst_side_tail_raw::<INVERT>(bm, idx, k, cst_sym, v_base, vc, out_ptr) };
}

/// Unified AVX2 constant-plus-vector merge (`INVERT = false` => `cst_vec`, `true` => `vec_cst`).
#[target_feature(enable = "avx2,popcnt")]
pub(super) unsafe fn merge_cst_side_avx2<const INVERT: bool>(
    bm: &[u8],
    k: usize,
    cst_sym: u8,
    v_base: *const u8,
    v_len: usize,
    out_ptr: *mut u8,
) {
    let ones = _mm256_set1_epi8(-1);
    let vcst256 = _mm256_set1_epi8(cst_sym as i8);
    let vcst128 = _mm_set1_epi8(cst_sym as i8);
    let mut vc = 0usize;
    let mut idx = 0usize;
    let bm_ptr = bm.as_ptr();

    while idx + 32 <= k {
        let raw_mask = unsafe { bm_ptr.add(idx >> 3).cast::<u32>().read_unaligned() }.to_le();
        let mask = if INVERT { !raw_mask } else { raw_mask };
        let lo_pop = _popcnt64(i64::from(mask & 0xFFFF)) as usize;
        if vc + lo_pop + 16 > v_len {
            break;
        }
        let vshuf = unsafe {
            build_shuf32_avx2(
                (mask & 0xFF) as usize,
                ((mask >> 8) & 0xFF) as usize,
                ((mask >> 16) & 0xFF) as usize,
                ((mask >> 24) & 0xFF) as usize,
            )
        };
        let vv = unsafe { load_halves_256(v_base.add(vc), v_base.add(vc + lo_pop)) };
        let rv = _mm256_shuffle_epi8(vv, vshuf);
        let rc = _mm256_shuffle_epi8(vcst256, _mm256_xor_si256(vshuf, ones));
        unsafe { _mm256_storeu_si256(out_ptr.add(idx).cast(), _mm256_or_si256(rc, rv)) };
        vc += _popcnt64(i64::from(mask)) as usize;
        idx += 32;
    }
    while idx + 16 <= k && vc + 16 <= v_len {
        let raw_m16 = unsafe { bm_ptr.add(idx >> 3).cast::<u16>().read_unaligned() }.to_le();
        let m16 = usize::from(if INVERT { !raw_m16 } else { raw_m16 });
        unsafe {
            merge_cv_step16(
                out_ptr.add(idx),
                vcst128,
                v_base.add(vc),
                m16 & 0xFF,
                m16 >> 8,
            );
        }
        vc += _popcnt64(m16 as i64) as usize;
        idx += 16;
    }
    unsafe { merge_cst_side_tail_raw::<INVERT>(bm, idx, k, cst_sym, v_base, vc, out_ptr) };
}

/// Table-free SSE4.1 `merge_cst_cst` (`x86.h:309-351`).
#[target_feature(enable = "sse4.1")]
pub(super) unsafe fn merge_cst_cst_sse41(
    bm: &[u8],
    k: usize,
    left_sym: u8,
    right_sym: u8,
    out: &mut [u8],
) {
    let vleft = _mm_set1_epi8(left_sym as i8);
    let vright = _mm_set1_epi8(right_sym as i8);
    let bits = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128);
    let shuf = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    let bm_ptr = bm.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut pos = 0usize;

    while pos + 16 <= k {
        let m16 = unsafe { bm_ptr.add(pos >> 3).cast::<u16>().read_unaligned() }.to_le();
        let dup = _mm_shuffle_epi8(_mm_cvtsi32_si128(i32::from(m16)), shuf);
        let mask8 = _mm_cmpeq_epi8(_mm_and_si128(dup, bits), bits);
        unsafe {
            _mm_storeu_si128(
                out_ptr.add(pos).cast(),
                _mm_blendv_epi8(vleft, vright, mask8),
            );
        }
        pos += 16;
    }
    while pos < k {
        let mb = (bm[pos >> 3] >> (pos & 7)) & 1;
        out[pos] = if mb != 0 { right_sym } else { left_sym };
        pos += 1;
    }
}

/// Table-free AVX2 32B/iter `merge_cst_cst` (`x86.h:321-337`).
#[target_feature(enable = "avx2")]
pub(super) unsafe fn merge_cst_cst_avx2(
    bm: &[u8],
    k: usize,
    left_sym: u8,
    right_sym: u8,
    out: &mut [u8],
) {
    let vleft = _mm256_set1_epi8(left_sym as i8);
    let vright = _mm256_set1_epi8(right_sym as i8);
    let shuf = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3,
    );
    let bit_mask = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
        1, 2, 4, 8, 16, 32, 64, -128,
    );
    let bm_ptr = bm.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut pos = 0usize;

    while pos + 32 <= k {
        let m32 = unsafe { bm_ptr.add(pos >> 3).cast::<u32>().read_unaligned() }.to_le() as i32;
        let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(m32), shuf);
        let eq = _mm256_cmpeq_epi8(_mm256_and_si256(bytes, bit_mask), bit_mask);
        unsafe {
            _mm256_storeu_si256(
                out_ptr.add(pos).cast(),
                _mm256_blendv_epi8(vleft, vright, eq),
            );
        }
        pos += 32;
    }
    while pos < k {
        let mb = (bm[pos >> 3] >> (pos & 7)) & 1;
        out[pos] = if mb != 0 { right_sym } else { left_sym };
        pos += 1;
    }
}

/// AVX2 `vpsrlvd` transpose D=2 flat unpack (`x86_flat.h:130-142`).
#[target_feature(enable = "avx2")]
pub(super) unsafe fn unpack_flat_d2_avx2(
    len: usize,
    packed: &[u8],
    c2s: &[u8; 64],
    out: &mut [u8],
) {
    let c2s_vec = unsafe { _mm_loadl_epi64(c2s.as_ptr().cast()) };
    let shifts = _mm_setr_epi32(0, 2, 4, 6);
    let mask3 = _mm_set1_epi8(0x03);
    let transpose = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    let p_ptr = packed.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut idx = 0usize;

    while idx + 16 <= len {
        let word = unsafe { p_ptr.add(idx >> 2).cast::<u32>().read_unaligned() }.to_le() as i32;
        let shifted = _mm_srlv_epi32(_mm_set1_epi32(word), shifts);
        let codes = _mm_and_si128(_mm_shuffle_epi8(shifted, transpose), mask3);
        unsafe { _mm_storeu_si128(out_ptr.add(idx).cast(), _mm_shuffle_epi8(c2s_vec, codes)) };
        idx += 16;
    }
    unpack_flat_tail_scalar(2, idx, len, packed, c2s, out);
}

#[target_feature(enable = "ssse3")]
pub(super) unsafe fn unpack_flat_d2_ssse3(
    len: usize,
    packed: &[u8],
    c2s: &[u8; 64],
    out: &mut [u8],
) {
    let word = u32::from_le_bytes([c2s[0], c2s[1], c2s[2], c2s[3]]) as i32;
    let tab_lo = _mm_set1_epi32(word);
    let tab_hi = _mm_shuffle_epi8(
        tab_lo,
        _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
    );
    let m0f = _mm_set1_epi8(0x0F);
    let p_ptr = packed.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut idx = 0usize;

    while idx + 64 <= len {
        let vec = unsafe { _mm_loadu_si128(p_ptr.add(idx >> 2).cast()) };
        let lo = _mm_and_si128(vec, m0f);
        let hi = _mm_and_si128(_mm_srli_epi16::<4>(vec), m0f);
        let s0 = _mm_shuffle_epi8(tab_lo, lo);
        let s1 = _mm_shuffle_epi8(tab_hi, lo);
        let s2 = _mm_shuffle_epi8(tab_lo, hi);
        let s3 = _mm_shuffle_epi8(tab_hi, hi);
        let p01_lo = _mm_unpacklo_epi8(s0, s1);
        let p01_hi = _mm_unpackhi_epi8(s0, s1);
        let p23_lo = _mm_unpacklo_epi8(s2, s3);
        let p23_hi = _mm_unpackhi_epi8(s2, s3);
        unsafe {
            _mm_storeu_si128(out_ptr.add(idx).cast(), _mm_unpacklo_epi16(p01_lo, p23_lo));
            _mm_storeu_si128(
                out_ptr.add(idx + 16).cast(),
                _mm_unpackhi_epi16(p01_lo, p23_lo),
            );
            _mm_storeu_si128(
                out_ptr.add(idx + 32).cast(),
                _mm_unpacklo_epi16(p01_hi, p23_hi),
            );
            _mm_storeu_si128(
                out_ptr.add(idx + 48).cast(),
                _mm_unpackhi_epi16(p01_hi, p23_hi),
            );
        }
        idx += 64;
    }
    unpack_flat_tail_scalar(2, idx, len, packed, c2s, out);
}

#[target_feature(enable = "ssse3")]
pub(super) unsafe fn unpack_flat_d4_ssse3(
    len: usize,
    packed: &[u8],
    c2s: &[u8; 64],
    out: &mut [u8],
) {
    let tab = unsafe { _mm_loadu_si128(c2s.as_ptr().cast()) };
    let m0f = _mm_set1_epi8(0x0F);
    let p_ptr = packed.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut idx = 0usize;

    while idx + 32 <= len {
        let vec = unsafe { _mm_loadu_si128(p_ptr.add(idx >> 1).cast()) };
        let s_lo = _mm_shuffle_epi8(tab, _mm_and_si128(vec, m0f));
        let s_hi = _mm_shuffle_epi8(tab, _mm_and_si128(_mm_srli_epi16::<4>(vec), m0f));
        unsafe {
            _mm_storeu_si128(out_ptr.add(idx).cast(), _mm_unpacklo_epi8(s_lo, s_hi));
            _mm_storeu_si128(out_ptr.add(idx + 16).cast(), _mm_unpackhi_epi8(s_lo, s_hi));
        }
        idx += 32;
    }
    while idx + 16 <= len {
        let raw = unsafe { _mm_loadl_epi64(p_ptr.add(idx >> 1).cast()) };
        let merged = _mm_unpacklo_epi8(raw, _mm_srli_epi16::<4>(raw));
        let codes = _mm_and_si128(merged, m0f);
        unsafe { _mm_storeu_si128(out_ptr.add(idx).cast(), _mm_shuffle_epi8(tab, codes)) };
        idx += 16;
    }
    unpack_flat_tail_scalar(4, idx, len, packed, c2s, out);
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn unpack_flat_d4_avx2(
    len: usize,
    packed: &[u8],
    c2s: &[u8; 64],
    out: &mut [u8],
) {
    let tab = _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(c2s.as_ptr().cast()) });
    let m0f = _mm256_set1_epi8(0x0F);
    let p_ptr = packed.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut idx = 0usize;

    while idx + 64 <= len {
        let vec = unsafe { _mm256_loadu_si256(p_ptr.add(idx >> 1).cast()) };
        let s_lo = _mm256_shuffle_epi8(tab, _mm256_and_si256(vec, m0f));
        let s_hi = _mm256_shuffle_epi8(tab, _mm256_and_si256(_mm256_srli_epi16::<4>(vec), m0f));
        let u_lo = _mm256_unpacklo_epi8(s_lo, s_hi);
        let u_hi = _mm256_unpackhi_epi8(s_lo, s_hi);
        unsafe {
            _mm256_storeu_si256(
                out_ptr.add(idx).cast(),
                _mm256_permute2x128_si256::<0x20>(u_lo, u_hi),
            );
            _mm256_storeu_si256(
                out_ptr.add(idx + 32).cast(),
                _mm256_permute2x128_si256::<0x31>(u_lo, u_hi),
            );
        }
        idx += 64;
    }
    unpack_flat_tail_scalar(4, idx, len, packed, c2s, out);
}

#[target_feature(enable = "ssse3")]
#[inline]
unsafe fn extract_16_codes_d3(vec: __m128i) -> __m128i {
    let shuf0 = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 2, -128, 2, -128);
    let shuf1 = _mm_setr_epi8(3, 4, 3, 4, 3, 4, 4, 5, 4, 5, 4, 5, 5, -128, 5, -128);
    let mul = _mm_setr_epi16(
        1 << 13,
        1 << 10,
        1 << 7,
        1 << 12,
        1 << 9,
        1 << 6,
        1 << 11,
        1 << 8,
    );
    let lo = _mm_srli_epi16::<13>(_mm_mullo_epi16(_mm_shuffle_epi8(vec, shuf0), mul));
    let hi = _mm_srli_epi16::<13>(_mm_mullo_epi16(_mm_shuffle_epi8(vec, shuf1), mul));
    _mm_packus_epi16(lo, hi)
}

#[target_feature(enable = "ssse3")]
#[inline]
unsafe fn extract_16_codes_d5(vec: __m128i) -> __m128i {
    let shuf0 = _mm_setr_epi8(0, 1, 0, 1, 1, 2, 1, 2, 2, 3, 3, 4, 3, 4, 4, -128);
    let shuf1 = _mm_setr_epi8(5, 6, 5, 6, 6, 7, 6, 7, 7, 8, 8, 9, 8, 9, 9, -128);
    let mul = _mm_setr_epi16(
        1 << 11,
        1 << 6,
        1 << 9,
        1 << 4,
        1 << 7,
        1 << 10,
        1 << 5,
        1 << 8,
    );
    let lo = _mm_srli_epi16::<11>(_mm_mullo_epi16(_mm_shuffle_epi8(vec, shuf0), mul));
    let hi = _mm_srli_epi16::<11>(_mm_mullo_epi16(_mm_shuffle_epi8(vec, shuf1), mul));
    _mm_packus_epi16(lo, hi)
}

#[target_feature(enable = "ssse3")]
#[inline]
unsafe fn extract_16_codes_d6(vec: __m128i) -> __m128i {
    let shuf0 = _mm_setr_epi8(0, 1, 0, 1, 1, 2, 2, -128, 3, 4, 3, 4, 4, 5, 5, -128);
    let shuf1 = _mm_setr_epi8(6, 7, 6, 7, 7, 8, 8, -128, 9, 10, 9, 10, 10, 11, 11, -128);
    let mul = _mm_setr_epi16(
        1 << 10,
        1 << 4,
        1 << 6,
        1 << 8,
        1 << 10,
        1 << 4,
        1 << 6,
        1 << 8,
    );
    let lo = _mm_srli_epi16::<10>(_mm_mullo_epi16(_mm_shuffle_epi8(vec, shuf0), mul));
    let hi = _mm_srli_epi16::<10>(_mm_mullo_epi16(_mm_shuffle_epi8(vec, shuf1), mul));
    _mm_packus_epi16(lo, hi)
}

#[target_feature(enable = "ssse3")]
pub(super) unsafe fn unpack_flat_d3_ssse3(
    len: usize,
    packed: &[u8],
    c2s: &[u8; 64],
    out: &mut [u8],
) {
    let tab = unsafe { _mm_loadu_si128(c2s.as_ptr().cast()) };
    let p_ptr = packed.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut idx = 0usize;
    let mut byte_pos = 0usize;

    while idx + 16 <= len && byte_pos + 8 <= packed.len() {
        let vec = unsafe { _mm_loadl_epi64(p_ptr.add(byte_pos).cast()) };
        let codes = unsafe { extract_16_codes_d3(vec) };
        unsafe { _mm_storeu_si128(out_ptr.add(idx).cast(), _mm_shuffle_epi8(tab, codes)) };
        idx += 16;
        byte_pos += 6;
    }
    unpack_flat_tail_scalar(3, idx, len, packed, c2s, out);
}

#[target_feature(enable = "ssse3")]
pub(super) unsafe fn unpack_flat_d5_ssse3(
    len: usize,
    packed: &[u8],
    c2s: &[u8; 64],
    out: &mut [u8],
) {
    let tab0 = unsafe { _mm_loadu_si128(c2s.as_ptr().cast()) };
    let tab1 = unsafe { _mm_loadu_si128(c2s.as_ptr().add(16).cast()) };
    let bit4 = _mm_set1_epi8(0x10);
    let bit7 = _mm_set1_epi8(i8::MIN);
    let p_ptr = packed.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut idx = 0usize;
    let mut byte_pos = 0usize;

    while idx + 16 <= len && byte_pos + 16 <= packed.len() {
        let vec = unsafe { _mm_loadu_si128(p_ptr.add(byte_pos).cast()) };
        let codes = unsafe { extract_16_codes_d5(vec) };
        let hi_flag = _mm_slli_epi16::<3>(_mm_and_si128(codes, bit4));
        let idx0 = _mm_or_si128(codes, hi_flag);
        let idx1 = _mm_xor_si128(idx0, bit7);
        let syms = _mm_or_si128(_mm_shuffle_epi8(tab0, idx0), _mm_shuffle_epi8(tab1, idx1));
        unsafe { _mm_storeu_si128(out_ptr.add(idx).cast(), syms) };
        idx += 16;
        byte_pos += 10;
    }
    unpack_flat_tail_scalar(5, idx, len, packed, c2s, out);
}

#[target_feature(enable = "ssse3")]
pub(super) unsafe fn unpack_flat_d6_ssse3(
    len: usize,
    packed: &[u8],
    c2s: &[u8; 64],
    out: &mut [u8],
) {
    let tab0 = unsafe { _mm_loadu_si128(c2s.as_ptr().cast()) };
    let tab1 = unsafe { _mm_loadu_si128(c2s.as_ptr().add(16).cast()) };
    let tab2 = unsafe { _mm_loadu_si128(c2s.as_ptr().add(32).cast()) };
    let tab3 = unsafe { _mm_loadu_si128(c2s.as_ptr().add(48).cast()) };
    let m0f = _mm_set1_epi8(0x0F);
    let m30 = _mm_set1_epi8(0x30);
    let bit7 = _mm_set1_epi8(i8::MIN);
    let q0 = _mm_setzero_si128();
    let q1 = _mm_set1_epi8(0x10);
    let q2 = _mm_set1_epi8(0x20);
    let q3 = _mm_set1_epi8(0x30);
    let p_ptr = packed.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut idx = 0usize;
    let mut byte_pos = 0usize;

    while idx + 16 <= len && byte_pos + 16 <= packed.len() {
        let vec = unsafe { _mm_loadu_si128(p_ptr.add(byte_pos).cast()) };
        let codes = unsafe { extract_16_codes_d6(vec) };
        let lo4 = _mm_and_si128(codes, m0f);
        let qv = _mm_and_si128(codes, m30);
        let r0 = _mm_shuffle_epi8(
            tab0,
            _mm_or_si128(lo4, _mm_andnot_si128(_mm_cmpeq_epi8(qv, q0), bit7)),
        );
        let r1 = _mm_shuffle_epi8(
            tab1,
            _mm_or_si128(lo4, _mm_andnot_si128(_mm_cmpeq_epi8(qv, q1), bit7)),
        );
        let r2 = _mm_shuffle_epi8(
            tab2,
            _mm_or_si128(lo4, _mm_andnot_si128(_mm_cmpeq_epi8(qv, q2), bit7)),
        );
        let r3 = _mm_shuffle_epi8(
            tab3,
            _mm_or_si128(lo4, _mm_andnot_si128(_mm_cmpeq_epi8(qv, q3), bit7)),
        );
        let syms = _mm_or_si128(_mm_or_si128(r0, r1), _mm_or_si128(r2, r3));
        unsafe { _mm_storeu_si128(out_ptr.add(idx).cast(), syms) };
        idx += 16;
        byte_pos += 12;
    }
    unpack_flat_tail_scalar(6, idx, len, packed, c2s, out);
}

#[target_feature(enable = "ssse3")]
pub(super) unsafe fn partition_ranks_ssse3(
    ranks: &[u8],
    pivot_rank: u8,
    bm_out: &mut [u8],
    left_out: &mut [u8],
    right_out: &mut [u8],
) -> usize {
    let len = ranks.len();
    let vpivot = _mm_set1_epi8(pivot_rank as i8);
    let r_ptr = ranks.as_ptr();
    let l_dst = left_out.as_mut_ptr();
    let r_dst = right_out.as_mut_ptr();
    let mut lc = 0usize;
    let mut rc = 0usize;
    let mut idx = 0usize;

    while idx + 16 <= len {
        let vec = unsafe { _mm_loadu_si128(r_ptr.add(idx).cast()) };
        // Unsigned `vec >= vpivot` iff `min_epu8(vec, vpivot) == vpivot`.
        let cmp = _mm_cmpeq_epi8(_mm_min_epu8(vec, vpivot), vpivot);
        let m16 = _mm_movemask_epi8(cmp) as u32;
        let m0 = (m16 & 0xFF) as usize;
        let m1 = ((m16 >> 8) & 0xFF) as usize;
        bm_out[idx >> 3] = m0 as u8;
        bm_out[(idx >> 3) + 1] = m1 as u8;

        let l_shuf0 = unsafe { _mm_loadl_epi64(COMPRESS_LEFT_TAB.0[m0].as_ptr().cast()) };
        let r_shuf0 = unsafe { _mm_loadl_epi64(COMPRESS_RIGHT_TAB.0[m0].as_ptr().cast()) };
        unsafe {
            _mm_storel_epi64(l_dst.add(lc).cast(), _mm_shuffle_epi8(vec, l_shuf0));
            _mm_storel_epi64(r_dst.add(rc).cast(), _mm_shuffle_epi8(vec, r_shuf0));
        }
        let pop0 = usize::from(POPCNT8[m0]);
        lc += 8 - pop0;
        rc += pop0;

        let vhi = _mm_srli_si128::<8>(vec);
        let l_shuf1 = unsafe { _mm_loadl_epi64(COMPRESS_LEFT_TAB.0[m1].as_ptr().cast()) };
        let r_shuf1 = unsafe { _mm_loadl_epi64(COMPRESS_RIGHT_TAB.0[m1].as_ptr().cast()) };
        unsafe {
            _mm_storel_epi64(l_dst.add(lc).cast(), _mm_shuffle_epi8(vhi, l_shuf1));
            _mm_storel_epi64(r_dst.add(rc).cast(), _mm_shuffle_epi8(vhi, r_shuf1));
        }
        let pop1 = usize::from(POPCNT8[m1]);
        lc += 8 - pop1;
        rc += pop1;

        idx += 16;
    }

    if idx < len {
        bm_out[idx >> 3..len.div_ceil(8)].fill(0);
        while idx < len {
            let rank = ranks[idx];
            if rank >= pivot_rank {
                bm_out[idx >> 3] |= 1u8 << (idx & 7);
                right_out[rc] = rank;
                rc += 1;
            } else {
                left_out[lc] = rank;
                lc += 1;
            }
            idx += 1;
        }
    }
    rc
}

/// AVX2 `vpmaddubsw` + `vpmaddwd` flat packer (`avx2_pack.h:60-93`).
#[target_feature(enable = "avx2")]
pub(super) unsafe fn pack_flat_codes_avx2(
    depth: u8,
    ranks: &[u8],
    base_rank: u8,
    out: &mut [u8],
) -> usize {
    let len = ranks.len();
    let total_bytes = out.len();
    let r_ptr = ranks.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let vbase = _mm256_set1_epi8(base_rank as i8);
    let vmask = _mm256_set1_epi8(((1u8 << depth) - 1) as i8);
    let mut idx = 0usize;

    match depth {
        2 | 3 | 5 | 6 => {
            let d_u32 = u32::from(depth);
            let d_usize = usize::from(depth);
            let c0 = _mm256_set1_epi16(((1i16 << d_u32) << 8) | 1);
            let c1 = _mm256_set1_epi32(((1i32 << (2 * d_u32)) << 16) | 1);
            let c3 = _mm256_set1_epi64x((1i64 << (4 * d_u32)) - 1);
            let compact = match depth {
                2 => _mm256_setr_epi8(
                    0, 1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 8, 9, -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                ),
                3 => _mm256_setr_epi8(
                    0, 1, 2, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 8, 9, 10,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                ),
                5 => _mm256_setr_epi8(
                    0, 1, 2, 3, 4, 8, 9, 10, 11, 12, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 8, 9,
                    10, 11, 12, -1, -1, -1, -1, -1, -1,
                ),
                _ => _mm256_setr_epi8(
                    0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 8, 9,
                    10, 11, 12, 13, -1, -1, -1, -1,
                ),
            };
            while idx + 32 <= len && ((idx * d_usize) >> 3) + 2 * d_usize + 16 <= total_bytes {
                let cb = _mm256_and_si256(
                    _mm256_sub_epi8(unsafe { _mm256_loadu_si256(r_ptr.add(idx).cast()) }, vbase),
                    vmask,
                );
                let mut val = _mm256_maddubs_epi16(c0, cb);
                val = _mm256_madd_epi16(val, c1);
                let shifted = match depth {
                    2 => _mm256_srli_epi64::<24>(val),
                    3 => _mm256_srli_epi64::<20>(val),
                    5 => _mm256_srli_epi64::<12>(val),
                    _ => _mm256_srli_epi64::<8>(val),
                };
                val = _mm256_or_si256(_mm256_and_si256(val, c3), _mm256_andnot_si256(c3, shifted));
                let out_y = _mm256_shuffle_epi8(val, compact);
                let outpos = (idx * d_usize) >> 3;
                unsafe {
                    _mm_storeu_si128(out_ptr.add(outpos).cast(), _mm256_castsi256_si128(out_y));
                    _mm_storeu_si128(
                        out_ptr.add(outpos + 2 * d_usize).cast(),
                        _mm256_extracti128_si256::<1>(out_y),
                    );
                }
                idx += 32;
            }
        }
        4 => {
            let c0 = _mm256_set1_epi16((16 << 8) | 1);
            let compact = _mm256_setr_epi8(
                0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, 10, 12,
                14, -1, -1, -1, -1, -1, -1, -1, -1,
            );
            while idx + 32 <= len {
                let cb = _mm256_and_si256(
                    _mm256_sub_epi8(unsafe { _mm256_loadu_si256(r_ptr.add(idx).cast()) }, vbase),
                    vmask,
                );
                let packed = _mm256_shuffle_epi8(_mm256_maddubs_epi16(c0, cb), compact);
                let outpos = idx >> 1;
                unsafe {
                    _mm_storel_epi64(out_ptr.add(outpos).cast(), _mm256_castsi256_si128(packed));
                    _mm_storel_epi64(
                        out_ptr.add(outpos + 8).cast(),
                        _mm256_extracti128_si256::<1>(packed),
                    );
                }
                idx += 32;
            }
        }
        _ => {}
    }
    idx
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn pack_64_bytes01_avx2(chunk64: &[u8]) -> u64 {
    let one = _mm256_set1_epi8(1);
    let ptr = chunk64.as_ptr();
    let v0 = unsafe { _mm256_loadu_si256(ptr.cast()) };
    let v1 = unsafe { _mm256_loadu_si256(ptr.add(32).cast()) };
    let m0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(v0, one), one)) as u32;
    let m1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(v1, one), one)) as u32;
    u64::from(m0) | (u64::from(m1) << 32)
}
