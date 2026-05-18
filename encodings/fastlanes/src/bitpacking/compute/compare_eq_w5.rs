// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::many_single_char_names)]

//! Hand-tuned in-range Eq/Lt kernel for `bit_width = 5` on `u32` storage.
//!
//! Five words per lane, alignment phase rotates by 3 bits per word (phases 0, 3, 1, 4, 2).
//! 28 of the 32 rows are fully contained in one word; the four straddlers are rows 6, 12,
//! 19 and 25.
//!
//! ```text
//!   word 0: rows  0..6  (6 full),  row 6 lower 2 bits
//!   word 1: rows  7..12 (5 full),  row 6 upper 3 bits, row 12 lower 4 bits
//!   word 2: rows 13..19 (6 full),  row 12 upper 1 bit, row 19 lower 1 bit
//!   word 3: rows 20..25 (5 full),  row 19 upper 4 bits, row 25 lower 3 bits
//!   word 4: rows 26..32 (6 full),  row 25 upper 2 bits
//! ```
//!
//! FL_ORDER buckets each contain exactly 8 rows; each bucket includes exactly one
//! straddler.

// Per-word L/H masks. L has bit 0 of each field set, H has bit 4 of each field set.
const L_W0: u32 = 0x0210_8421;
const H_W0: u32 = 0x2108_4210;
const L_W1: u32 = 0x0084_2108;
const H_W1: u32 = 0x0842_1080;
const L_W2: u32 = 0x0421_0842;
const H_W2: u32 = 0x4210_8420;
const L_W3: u32 = 0x0108_4210;
const H_W3: u32 = 0x1084_2100;
const L_W4: u32 = 0x0842_1084;
const H_W4: u32 = 0x8421_0840;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
pub(crate) fn swar_eq_w5_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    // SAFETY: target_feature=avx2 gates compilation here.
    unsafe { simd_eq_w5_avx2(packed_chunk, c, out) }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
pub(crate) fn swar_eq_w5_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    scalar_w5(packed_chunk, c, out, /* eq= */ true);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
pub(crate) fn swar_lt_w5_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    // SAFETY: target_feature=avx2 gates compilation here.
    unsafe { simd_lt_w5_avx2(packed_chunk, c, out) }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
pub(crate) fn swar_lt_w5_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    scalar_w5(packed_chunk, c, out, /* eq= */ false);
}

// ---------------------------------------------------------------------------
// AVX2 implementation.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn simd_eq_w5_avx2(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    use std::arch::x86_64::*;

    use super::compare_eq_w4::extract_slot;
    use super::compare_eq_w4::scatter_rows;

    debug_assert_eq!(packed_chunk.len(), 160);

    let c5 = u32::from(c & 0x1F);

    // SAFETY: AVX2 guaranteed by cfg gate; pointer arithmetic stays in bounds because
    // `packed_chunk.len() == 160`.
    unsafe {
        let c_vec = _mm256_set1_epi32(c5 as i32);
        let c_w0 = _mm256_set1_epi32((c5 * L_W0) as i32);
        let c_w1 = _mm256_set1_epi32((c5 * L_W1) as i32);
        let c_w2 = _mm256_set1_epi32((c5 * L_W2) as i32);
        let c_w3 = _mm256_set1_epi32((c5 * L_W3) as i32);
        let c_w4 = _mm256_set1_epi32((c5 * L_W4) as i32);
        let l_w0 = _mm256_set1_epi32(L_W0 as i32);
        let l_w1 = _mm256_set1_epi32(L_W1 as i32);
        let l_w2 = _mm256_set1_epi32(L_W2 as i32);
        let l_w3 = _mm256_set1_epi32(L_W3 as i32);
        let l_w4 = _mm256_set1_epi32(L_W4 as i32);
        let h_w0 = _mm256_set1_epi32(H_W0 as i32);
        let h_w1 = _mm256_set1_epi32(H_W1 as i32);
        let h_w2 = _mm256_set1_epi32(H_W2 as i32);
        let h_w3 = _mm256_set1_epi32(H_W3 as i32);
        let h_w4 = _mm256_set1_epi32(H_W4 as i32);

        macro_rules! load4 {
            ($word_idx:literal) => {{
                let base = packed_chunk.as_ptr().add($word_idx * 32).cast::<__m256i>();
                (
                    _mm256_loadu_si256(base.add(0)),
                    _mm256_loadu_si256(base.add(1)),
                    _mm256_loadu_si256(base.add(2)),
                    _mm256_loadu_si256(base.add(3)),
                )
            }};
        }

        let (p0_w0, p1_w0, p2_w0, p3_w0) = load4!(0);
        let (p0_w1, p1_w1, p2_w1, p3_w1) = load4!(1);
        let (p0_w2, p1_w2, p2_w2, p3_w2) = load4!(2);
        let (p0_w3, p1_w3, p2_w3, p3_w3) = load4!(3);
        let (p0_w4, p1_w4, p2_w4, p3_w4) = load4!(4);

        macro_rules! knuth {
            ($p:ident, $c:ident, $l:ident, $h:ident) => {{
                let xor = _mm256_xor_si256($p, $c);
                _mm256_and_si256(_mm256_sub_epi32(xor, $l), _mm256_andnot_si256(xor, $h))
            }};
        }
        let z0_w0 = knuth!(p0_w0, c_w0, l_w0, h_w0);
        let z1_w0 = knuth!(p1_w0, c_w0, l_w0, h_w0);
        let z2_w0 = knuth!(p2_w0, c_w0, l_w0, h_w0);
        let z3_w0 = knuth!(p3_w0, c_w0, l_w0, h_w0);
        let z0_w1 = knuth!(p0_w1, c_w1, l_w1, h_w1);
        let z1_w1 = knuth!(p1_w1, c_w1, l_w1, h_w1);
        let z2_w1 = knuth!(p2_w1, c_w1, l_w1, h_w1);
        let z3_w1 = knuth!(p3_w1, c_w1, l_w1, h_w1);
        let z0_w2 = knuth!(p0_w2, c_w2, l_w2, h_w2);
        let z1_w2 = knuth!(p1_w2, c_w2, l_w2, h_w2);
        let z2_w2 = knuth!(p2_w2, c_w2, l_w2, h_w2);
        let z3_w2 = knuth!(p3_w2, c_w2, l_w2, h_w2);
        let z0_w3 = knuth!(p0_w3, c_w3, l_w3, h_w3);
        let z1_w3 = knuth!(p1_w3, c_w3, l_w3, h_w3);
        let z2_w3 = knuth!(p2_w3, c_w3, l_w3, h_w3);
        let z3_w3 = knuth!(p3_w3, c_w3, l_w3, h_w3);
        let z0_w4 = knuth!(p0_w4, c_w4, l_w4, h_w4);
        let z1_w4 = knuth!(p1_w4, c_w4, l_w4, h_w4);
        let z2_w4 = knuth!(p2_w4, c_w4, l_w4, h_w4);
        let z3_w4 = knuth!(p3_w4, c_w4, l_w4, h_w4);

        // Straddlers — reconstruct val per lane then `cmpeq_epi32`.
        // Row 6:  val = ((w0 >> 30) & 3) | ((w1 & 7) << 2)
        // Row 12: val = ((w1 >> 28) & 15) | ((w2 & 1) << 4)
        // Row 19: val = ((w2 >> 31) & 1) | ((w3 & 15) << 1)
        // Row 25: val = ((w3 >> 29) & 7) | ((w4 & 3) << 3)
        let straddler_6 = straddler_eq::<30, 2, 3, 7>(
            p0_w0, p1_w0, p2_w0, p3_w0, p0_w1, p1_w1, p2_w1, p3_w1, c_vec,
        );
        let straddler_12 = straddler_eq::<28, 4, 15, 1>(
            p0_w1, p1_w1, p2_w1, p3_w1, p0_w2, p1_w2, p2_w2, p3_w2, c_vec,
        );
        let straddler_19 = straddler_eq::<31, 1, 1, 15>(
            p0_w2, p1_w2, p2_w2, p3_w2, p0_w3, p1_w3, p2_w3, p3_w3, c_vec,
        );
        let straddler_25 = straddler_eq::<29, 3, 7, 3>(
            p0_w3, p1_w3, p2_w3, p3_w3, p0_w4, p1_w4, p2_w4, p3_w4, c_vec,
        );

        // Bucket 0 (rows 0..8): word 0 slots 0..6, straddler 6, word 1 slot 0.
        //   Word 0 SHIFTs: 27, 22, 17, 12, 7, 2 (rows 0..5).
        //   Word 1 slot 0 (row 7) SHIFT: 24.
        scatter_rows(0, out, [
            extract_slot::<27>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<22>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<17>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<12>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<7>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<2>(z0_w0, z1_w0, z2_w0, z3_w0),
            straddler_6,
            extract_slot::<24>(z0_w1, z1_w1, z2_w1, z3_w1),
        ]);

        // Bucket 1 (rows 8..16): word 1 slots 1..4, straddler 12, word 2 slots 0..2.
        //   Word 1 slots 1..4 (rows 8..11) SHIFTs: 19, 14, 9, 4.
        //   Word 2 slots 0..2 (rows 13..15) SHIFTs: 26, 21, 16.
        scatter_rows(1, out, [
            extract_slot::<19>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<14>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<9>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<4>(z0_w1, z1_w1, z2_w1, z3_w1),
            straddler_12,
            extract_slot::<26>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<21>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<16>(z0_w2, z1_w2, z2_w2, z3_w2),
        ]);

        // Bucket 2 (rows 16..24): word 2 slots 3..5, straddler 19, word 3 slots 0..3.
        //   Word 2 slots 3..5 (rows 16..18) SHIFTs: 11, 6, 1.
        //   Word 3 slots 0..3 (rows 20..23) SHIFTs: 23, 18, 13, 8.
        scatter_rows(2, out, [
            extract_slot::<11>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<6>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<1>(z0_w2, z1_w2, z2_w2, z3_w2),
            straddler_19,
            extract_slot::<23>(z0_w3, z1_w3, z2_w3, z3_w3),
            extract_slot::<18>(z0_w3, z1_w3, z2_w3, z3_w3),
            extract_slot::<13>(z0_w3, z1_w3, z2_w3, z3_w3),
            extract_slot::<8>(z0_w3, z1_w3, z2_w3, z3_w3),
        ]);

        // Bucket 3 (rows 24..32): word 3 slot 4, straddler 25, word 4 slots 0..5.
        //   Word 3 slot 4 (row 24) SHIFT: 3.
        //   Word 4 slots 0..5 (rows 26..31) SHIFTs: 25, 20, 15, 10, 5, 0.
        scatter_rows(3, out, [
            extract_slot::<3>(z0_w3, z1_w3, z2_w3, z3_w3),
            straddler_25,
            extract_slot::<25>(z0_w4, z1_w4, z2_w4, z3_w4),
            extract_slot::<20>(z0_w4, z1_w4, z2_w4, z3_w4),
            extract_slot::<15>(z0_w4, z1_w4, z2_w4, z3_w4),
            extract_slot::<10>(z0_w4, z1_w4, z2_w4, z3_w4),
            extract_slot::<5>(z0_w4, z1_w4, z2_w4, z3_w4),
            extract_slot::<0>(z0_w4, z1_w4, z2_w4, z3_w4),
        ]);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn straddler_eq<
    const LO_SHR: i32,
    const HI_SHL: i32,
    const LO_MASK: i32,
    const HI_MASK: i32,
>(
    p0_lo: std::arch::x86_64::__m256i,
    p1_lo: std::arch::x86_64::__m256i,
    p2_lo: std::arch::x86_64::__m256i,
    p3_lo: std::arch::x86_64::__m256i,
    p0_hi: std::arch::x86_64::__m256i,
    p1_hi: std::arch::x86_64::__m256i,
    p2_hi: std::arch::x86_64::__m256i,
    p3_hi: std::arch::x86_64::__m256i,
    c_vec: std::arch::x86_64::__m256i,
) -> u32 {
    use std::arch::x86_64::*;
    // SAFETY: AVX2 guaranteed.
    unsafe {
        let lo_mask = _mm256_set1_epi32(LO_MASK);
        let hi_mask = _mm256_set1_epi32(HI_MASK);
        macro_rules! one {
            ($lo:ident, $hi:ident) => {{
                let lo = _mm256_and_si256(_mm256_srli_epi32::<LO_SHR>($lo), lo_mask);
                let hi = _mm256_slli_epi32::<HI_SHL>(_mm256_and_si256($hi, hi_mask));
                let val = _mm256_or_si256(lo, hi);
                let cmp = _mm256_cmpeq_epi32(val, c_vec);
                _mm256_movemask_ps(_mm256_castsi256_ps(cmp)) as u32 & 0xFF
            }};
        }
        let m0 = one!(p0_lo, p0_hi);
        let m1 = one!(p1_lo, p1_hi);
        let m2 = one!(p2_lo, p2_hi);
        let m3 = one!(p3_lo, p3_hi);
        m0 | (m1 << 8) | (m2 << 16) | (m3 << 24)
    }
}

// Lt: per-word Knuth high/low split (high = bit 4, low = bits 0..3 of each slot). Straddlers
// reconstruct val per lane then use the `min_epu32` identity.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn simd_lt_w5_avx2(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    use std::arch::x86_64::*;

    use super::compare_eq_w4::extract_slot;
    use super::compare_eq_w4::scatter_rows;

    debug_assert_eq!(packed_chunk.len(), 160);

    let c5 = u32::from(c & 0x1F);

    unsafe {
        let c_vec = _mm256_set1_epi32(c5 as i32);

        macro_rules! load4 {
            ($word_idx:literal) => {{
                let base = packed_chunk.as_ptr().add($word_idx * 32).cast::<__m256i>();
                (
                    _mm256_loadu_si256(base.add(0)),
                    _mm256_loadu_si256(base.add(1)),
                    _mm256_loadu_si256(base.add(2)),
                    _mm256_loadu_si256(base.add(3)),
                )
            }};
        }
        let (p0_w0, p1_w0, p2_w0, p3_w0) = load4!(0);
        let (p0_w1, p1_w1, p2_w1, p3_w1) = load4!(1);
        let (p0_w2, p1_w2, p2_w2, p3_w2) = load4!(2);
        let (p0_w3, p1_w3, p2_w3, p3_w3) = load4!(3);
        let (p0_w4, p1_w4, p2_w4, p3_w4) = load4!(4);

        // For each W-bit field, the "lo" mask in Knuth is bits 0..W-1 minus the H bit. For
        // W=5, lo positions in a field = bits 0..3 → mask = L | (L<<1) | (L<<2) | (L<<3).
        macro_rules! lt_word {
            ($p:ident, $L:expr, $H:expr) => {{
                let c_packed = c5 * $L;
                let c_hi_word = c_packed & $H;
                let lo_mask_word = $L | ($L << 1) | ($L << 2) | ($L << 3);
                let c_lo_word = c_packed & lo_mask_word;
                let l_vec = _mm256_set1_epi32($L as i32);
                let h_vec = _mm256_set1_epi32($H as i32);
                let m_vec = _mm256_set1_epi32(lo_mask_word as i32);
                let c_hi = _mm256_set1_epi32(c_hi_word as i32);
                let c_lo = _mm256_set1_epi32(c_lo_word as i32);
                let c_lo_or_h = _mm256_set1_epi32((c_lo_word | $H) as i32);

                let a_hi = _mm256_and_si256($p, h_vec);
                let a_lo = _mm256_and_si256($p, m_vec);
                let hi_lt = _mm256_andnot_si256(a_hi, c_hi);
                let hi_eq = _mm256_andnot_si256(_mm256_xor_si256(a_hi, c_hi), h_vec);
                let lo_le = _mm256_and_si256(_mm256_sub_epi32(c_lo_or_h, a_lo), h_vec);
                let xor_lo = _mm256_xor_si256(a_lo, c_lo);
                let lo_eq = _mm256_and_si256(
                    _mm256_sub_epi32(xor_lo, l_vec),
                    _mm256_andnot_si256(xor_lo, h_vec),
                );
                let lo_lt = _mm256_andnot_si256(lo_eq, lo_le);
                _mm256_or_si256(hi_lt, _mm256_and_si256(hi_eq, lo_lt))
            }};
        }

        let z0_w0 = lt_word!(p0_w0, L_W0, H_W0);
        let z1_w0 = lt_word!(p1_w0, L_W0, H_W0);
        let z2_w0 = lt_word!(p2_w0, L_W0, H_W0);
        let z3_w0 = lt_word!(p3_w0, L_W0, H_W0);
        let z0_w1 = lt_word!(p0_w1, L_W1, H_W1);
        let z1_w1 = lt_word!(p1_w1, L_W1, H_W1);
        let z2_w1 = lt_word!(p2_w1, L_W1, H_W1);
        let z3_w1 = lt_word!(p3_w1, L_W1, H_W1);
        let z0_w2 = lt_word!(p0_w2, L_W2, H_W2);
        let z1_w2 = lt_word!(p1_w2, L_W2, H_W2);
        let z2_w2 = lt_word!(p2_w2, L_W2, H_W2);
        let z3_w2 = lt_word!(p3_w2, L_W2, H_W2);
        let z0_w3 = lt_word!(p0_w3, L_W3, H_W3);
        let z1_w3 = lt_word!(p1_w3, L_W3, H_W3);
        let z2_w3 = lt_word!(p2_w3, L_W3, H_W3);
        let z3_w3 = lt_word!(p3_w3, L_W3, H_W3);
        let z0_w4 = lt_word!(p0_w4, L_W4, H_W4);
        let z1_w4 = lt_word!(p1_w4, L_W4, H_W4);
        let z2_w4 = lt_word!(p2_w4, L_W4, H_W4);
        let z3_w4 = lt_word!(p3_w4, L_W4, H_W4);

        let straddler_6 = straddler_lt::<30, 2, 3, 7>(
            p0_w0, p1_w0, p2_w0, p3_w0, p0_w1, p1_w1, p2_w1, p3_w1, c_vec,
        );
        let straddler_12 = straddler_lt::<28, 4, 15, 1>(
            p0_w1, p1_w1, p2_w1, p3_w1, p0_w2, p1_w2, p2_w2, p3_w2, c_vec,
        );
        let straddler_19 = straddler_lt::<31, 1, 1, 15>(
            p0_w2, p1_w2, p2_w2, p3_w2, p0_w3, p1_w3, p2_w3, p3_w3, c_vec,
        );
        let straddler_25 = straddler_lt::<29, 3, 7, 3>(
            p0_w3, p1_w3, p2_w3, p3_w3, p0_w4, p1_w4, p2_w4, p3_w4, c_vec,
        );

        scatter_rows(0, out, [
            extract_slot::<27>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<22>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<17>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<12>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<7>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<2>(z0_w0, z1_w0, z2_w0, z3_w0),
            straddler_6,
            extract_slot::<24>(z0_w1, z1_w1, z2_w1, z3_w1),
        ]);
        scatter_rows(1, out, [
            extract_slot::<19>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<14>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<9>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<4>(z0_w1, z1_w1, z2_w1, z3_w1),
            straddler_12,
            extract_slot::<26>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<21>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<16>(z0_w2, z1_w2, z2_w2, z3_w2),
        ]);
        scatter_rows(2, out, [
            extract_slot::<11>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<6>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<1>(z0_w2, z1_w2, z2_w2, z3_w2),
            straddler_19,
            extract_slot::<23>(z0_w3, z1_w3, z2_w3, z3_w3),
            extract_slot::<18>(z0_w3, z1_w3, z2_w3, z3_w3),
            extract_slot::<13>(z0_w3, z1_w3, z2_w3, z3_w3),
            extract_slot::<8>(z0_w3, z1_w3, z2_w3, z3_w3),
        ]);
        scatter_rows(3, out, [
            extract_slot::<3>(z0_w3, z1_w3, z2_w3, z3_w3),
            straddler_25,
            extract_slot::<25>(z0_w4, z1_w4, z2_w4, z3_w4),
            extract_slot::<20>(z0_w4, z1_w4, z2_w4, z3_w4),
            extract_slot::<15>(z0_w4, z1_w4, z2_w4, z3_w4),
            extract_slot::<10>(z0_w4, z1_w4, z2_w4, z3_w4),
            extract_slot::<5>(z0_w4, z1_w4, z2_w4, z3_w4),
            extract_slot::<0>(z0_w4, z1_w4, z2_w4, z3_w4),
        ]);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn straddler_lt<
    const LO_SHR: i32,
    const HI_SHL: i32,
    const LO_MASK: i32,
    const HI_MASK: i32,
>(
    p0_lo: std::arch::x86_64::__m256i,
    p1_lo: std::arch::x86_64::__m256i,
    p2_lo: std::arch::x86_64::__m256i,
    p3_lo: std::arch::x86_64::__m256i,
    p0_hi: std::arch::x86_64::__m256i,
    p1_hi: std::arch::x86_64::__m256i,
    p2_hi: std::arch::x86_64::__m256i,
    p3_hi: std::arch::x86_64::__m256i,
    c_vec: std::arch::x86_64::__m256i,
) -> u32 {
    use std::arch::x86_64::*;
    unsafe {
        let lo_mask = _mm256_set1_epi32(LO_MASK);
        let hi_mask = _mm256_set1_epi32(HI_MASK);
        macro_rules! one {
            ($lo:ident, $hi:ident) => {{
                let lo = _mm256_and_si256(_mm256_srli_epi32::<LO_SHR>($lo), lo_mask);
                let hi = _mm256_slli_epi32::<HI_SHL>(_mm256_and_si256($hi, hi_mask));
                let val = _mm256_or_si256(lo, hi);
                let min = _mm256_min_epu32(val, c_vec);
                let le = _mm256_cmpeq_epi32(min, val);
                let eq = _mm256_cmpeq_epi32(val, c_vec);
                let lt = _mm256_andnot_si256(eq, le);
                _mm256_movemask_ps(_mm256_castsi256_ps(lt)) as u32 & 0xFF
            }};
        }
        let m0 = one!(p0_lo, p0_hi);
        let m1 = one!(p1_lo, p1_hi);
        let m2 = one!(p2_lo, p2_hi);
        let m3 = one!(p3_lo, p3_hi);
        m0 | (m1 << 8) | (m2 << 16) | (m3 << 24)
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback (works for both Eq and Lt; W=5 uses the FastLanes index formula
// directly to avoid replicating the rotation logic in scalar code).
// ---------------------------------------------------------------------------

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn scalar_w5(packed_chunk: &[u32], c: u8, out: &mut [u64; 16], eq: bool) {
    debug_assert_eq!(packed_chunk.len(), 160);
    let c5 = u32::from(c & 0x1F);
    for row in 0..32usize {
        let start_bit = row * 5;
        let word = start_bit / 32;
        let bit = start_bit % 32;
        for lane in 0..32usize {
            let lo = packed_chunk[word * 32 + lane] >> bit;
            let val = if bit + 5 <= 32 {
                lo & 0x1F
            } else {
                let hi = packed_chunk[(word + 1) * 32 + lane] << (32 - bit);
                (lo | hi) & 0x1F
            };
            let i = fastlanes::FL_ORDER[row / 8] * 16 + (row % 8) * 128 + lane;
            let matches = if eq { val == c5 } else { val < c5 };
            if matches {
                out[i / 64] |= 1u64 << (i % 64);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use fastlanes::BitPacking;

    use super::*;

    fn pack_u32(values: &[u32; 1024]) -> [u32; 160] {
        let mut out = [0u32; 160];
        // SAFETY: `out` matches `128 * W / size_of::<u32>() = 160` for W=5.
        unsafe {
            BitPacking::unchecked_pack(5, values, &mut out);
        }
        out
    }

    fn bit(chunk_bits: &[u64; 16], i: usize) -> bool {
        (chunk_bits[i / 64] >> (i % 64)) & 1 != 0
    }

    #[test]
    fn eq_w5_matches_naive() {
        let mut values = [0u32; 1024];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 17 + 3) & 0x1F) as u32;
        }
        let packed = pack_u32(&values);

        for c in 0..32u8 {
            let mut got = [0u64; 16];
            swar_eq_w5_u32(&packed, c, &mut got);
            for i in 0..1024 {
                let expected = values[i] == u32::from(c);
                assert_eq!(bit(&got, i), expected, "i={i}, c={c}, value={}", values[i]);
            }
        }
    }

    #[test]
    fn lt_w5_matches_naive() {
        let mut values = [0u32; 1024];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 31 + 5) & 0x1F) as u32;
        }
        let packed = pack_u32(&values);

        for c in 0..32u8 {
            let mut got = [0u64; 16];
            swar_lt_w5_u32(&packed, c, &mut got);
            for i in 0..1024 {
                let expected = values[i] < u32::from(c);
                assert_eq!(bit(&got, i), expected, "i={i}, c={c}, value={}", values[i]);
            }
        }
    }
}
