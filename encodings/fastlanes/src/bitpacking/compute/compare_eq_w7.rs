// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::many_single_char_names)]

//! Hand-tuned in-range Eq/Lt kernel for `bit_width = 7` on `u32` storage.
//!
//! Seven words per lane, alignment phase rotates by 4 bits per word
//! (phases 0, 3, 6, 2, 5, 1, 4). 26 of the 32 rows are fully contained in one word; six are
//! straddlers (rows 4, 9, 13, 18, 22, 27).
//!
//! ```text
//!   word 0: rows  0..4  (4 full), row  4 lower 4 bits
//!   word 1: rows  5..9  (4 full), row  4 upper 3 bits, row  9 lower 1 bit
//!   word 2: rows 10..13 (3 full), row  9 upper 6 bits, row 13 lower 5 bits
//!   word 3: rows 14..18 (4 full), row 13 upper 2 bits, row 18 lower 2 bits
//!   word 4: rows 19..22 (3 full), row 18 upper 5 bits, row 22 lower 6 bits
//!   word 5: rows 23..27 (4 full), row 22 upper 1 bit,  row 27 lower 3 bits
//!   word 6: rows 28..32 (4 full), row 27 upper 4 bits
//! ```
//!
//! Buckets 1 and 2 each contain two straddler rows; buckets 0 and 3 contain one each.

const L_W0: u32 = 0x0020_4081;
const H_W0: u32 = 0x0810_2040;
const L_W1: u32 = 0x0102_0408;
const H_W1: u32 = 0x4081_0200;
const L_W2: u32 = 0x0010_2040;
const H_W2: u32 = 0x0408_1000;
const L_W3: u32 = 0x0081_0204;
const H_W3: u32 = 0x2040_8100;
const L_W4: u32 = 0x0008_1020;
const H_W4: u32 = 0x0204_0800;
const L_W5: u32 = 0x0040_8102;
const H_W5: u32 = 0x1020_4080;
const L_W6: u32 = 0x0204_0810;
const H_W6: u32 = 0x8102_0400;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
pub(crate) fn swar_eq_w7_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    // SAFETY: target_feature=avx2 gates compilation here.
    unsafe { simd_eq_w7_avx2(packed_chunk, c, out) }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
pub(crate) fn swar_eq_w7_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    scalar_w7(packed_chunk, c, out, /* eq= */ true);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
pub(crate) fn swar_lt_w7_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    // SAFETY: target_feature=avx2 gates compilation here.
    unsafe { simd_lt_w7_avx2(packed_chunk, c, out) }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
pub(crate) fn swar_lt_w7_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    scalar_w7(packed_chunk, c, out, /* eq= */ false);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn simd_eq_w7_avx2(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    use std::arch::x86_64::*;

    use super::compare_eq_w4::extract_slot;
    use super::compare_eq_w4::scatter_rows;
    use super::compare_eq_w5::straddler_eq;

    debug_assert_eq!(packed_chunk.len(), 224);

    let c7 = u32::from(c & 0x7F);

    // SAFETY: AVX2 guaranteed by cfg gate; pointer arithmetic stays in bounds because
    // `packed_chunk.len() == 224`.
    unsafe {
        let c_vec = _mm256_set1_epi32(c7 as i32);

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
        let (p0_w5, p1_w5, p2_w5, p3_w5) = load4!(5);
        let (p0_w6, p1_w6, p2_w6, p3_w6) = load4!(6);

        macro_rules! knuth {
            ($p:ident, $L:expr, $H:expr) => {{
                let c_packed = _mm256_set1_epi32((c7 * $L) as i32);
                let l_vec = _mm256_set1_epi32($L as i32);
                let h_vec = _mm256_set1_epi32($H as i32);
                let xor = _mm256_xor_si256($p, c_packed);
                _mm256_and_si256(_mm256_sub_epi32(xor, l_vec), _mm256_andnot_si256(xor, h_vec))
            }};
        }
        let z0_w0 = knuth!(p0_w0, L_W0, H_W0);
        let z1_w0 = knuth!(p1_w0, L_W0, H_W0);
        let z2_w0 = knuth!(p2_w0, L_W0, H_W0);
        let z3_w0 = knuth!(p3_w0, L_W0, H_W0);
        let z0_w1 = knuth!(p0_w1, L_W1, H_W1);
        let z1_w1 = knuth!(p1_w1, L_W1, H_W1);
        let z2_w1 = knuth!(p2_w1, L_W1, H_W1);
        let z3_w1 = knuth!(p3_w1, L_W1, H_W1);
        let z0_w2 = knuth!(p0_w2, L_W2, H_W2);
        let z1_w2 = knuth!(p1_w2, L_W2, H_W2);
        let z2_w2 = knuth!(p2_w2, L_W2, H_W2);
        let z3_w2 = knuth!(p3_w2, L_W2, H_W2);
        let z0_w3 = knuth!(p0_w3, L_W3, H_W3);
        let z1_w3 = knuth!(p1_w3, L_W3, H_W3);
        let z2_w3 = knuth!(p2_w3, L_W3, H_W3);
        let z3_w3 = knuth!(p3_w3, L_W3, H_W3);
        let z0_w4 = knuth!(p0_w4, L_W4, H_W4);
        let z1_w4 = knuth!(p1_w4, L_W4, H_W4);
        let z2_w4 = knuth!(p2_w4, L_W4, H_W4);
        let z3_w4 = knuth!(p3_w4, L_W4, H_W4);
        let z0_w5 = knuth!(p0_w5, L_W5, H_W5);
        let z1_w5 = knuth!(p1_w5, L_W5, H_W5);
        let z2_w5 = knuth!(p2_w5, L_W5, H_W5);
        let z3_w5 = knuth!(p3_w5, L_W5, H_W5);
        let z0_w6 = knuth!(p0_w6, L_W6, H_W6);
        let z1_w6 = knuth!(p1_w6, L_W6, H_W6);
        let z2_w6 = knuth!(p2_w6, L_W6, H_W6);
        let z3_w6 = knuth!(p3_w6, L_W6, H_W6);

        // Straddler eqs.
        //   Row  4: ((w0 >> 28) & 15) | ((w1 & 7) << 4)
        //   Row  9: ((w1 >> 31) & 1)  | ((w2 & 63) << 1)
        //   Row 13: ((w2 >> 27) & 31) | ((w3 & 3)  << 5)
        //   Row 18: ((w3 >> 30) & 3)  | ((w4 & 31) << 2)
        //   Row 22: ((w4 >> 26) & 63) | ((w5 & 1)  << 6)
        //   Row 27: ((w5 >> 29) & 7)  | ((w6 & 15) << 3)
        let s4 = straddler_eq::<28, 4, 15, 7>(
            p0_w0, p1_w0, p2_w0, p3_w0, p0_w1, p1_w1, p2_w1, p3_w1, c_vec,
        );
        let s9 = straddler_eq::<31, 1, 1, 63>(
            p0_w1, p1_w1, p2_w1, p3_w1, p0_w2, p1_w2, p2_w2, p3_w2, c_vec,
        );
        let s13 = straddler_eq::<27, 5, 31, 3>(
            p0_w2, p1_w2, p2_w2, p3_w2, p0_w3, p1_w3, p2_w3, p3_w3, c_vec,
        );
        let s18 = straddler_eq::<30, 2, 3, 31>(
            p0_w3, p1_w3, p2_w3, p3_w3, p0_w4, p1_w4, p2_w4, p3_w4, c_vec,
        );
        let s22 = straddler_eq::<26, 6, 63, 1>(
            p0_w4, p1_w4, p2_w4, p3_w4, p0_w5, p1_w5, p2_w5, p3_w5, c_vec,
        );
        let s27 = straddler_eq::<29, 3, 7, 15>(
            p0_w5, p1_w5, p2_w5, p3_w5, p0_w6, p1_w6, p2_w6, p3_w6, c_vec,
        );

        // Bucket 0 (rows 0..8): w0 slots 0..4, straddler 4, w1 slots 0..2.
        scatter_rows(0, out, [
            extract_slot::<25>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<18>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<11>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<4>(z0_w0, z1_w0, z2_w0, z3_w0),
            s4,
            extract_slot::<22>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<15>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<8>(z0_w1, z1_w1, z2_w1, z3_w1),
        ]);

        // Bucket 1 (rows 8..16): w1 slot 3, straddler 9, w2 slots 0..2, straddler 13, w3 slots 0..1.
        scatter_rows(1, out, [
            extract_slot::<1>(z0_w1, z1_w1, z2_w1, z3_w1),
            s9,
            extract_slot::<19>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<12>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<5>(z0_w2, z1_w2, z2_w2, z3_w2),
            s13,
            extract_slot::<23>(z0_w3, z1_w3, z2_w3, z3_w3),
            extract_slot::<16>(z0_w3, z1_w3, z2_w3, z3_w3),
        ]);

        // Bucket 2 (rows 16..24): w3 slots 2..3, straddler 18, w4 slots 0..2, straddler 22, w5 slot 0.
        scatter_rows(2, out, [
            extract_slot::<9>(z0_w3, z1_w3, z2_w3, z3_w3),
            extract_slot::<2>(z0_w3, z1_w3, z2_w3, z3_w3),
            s18,
            extract_slot::<20>(z0_w4, z1_w4, z2_w4, z3_w4),
            extract_slot::<13>(z0_w4, z1_w4, z2_w4, z3_w4),
            extract_slot::<6>(z0_w4, z1_w4, z2_w4, z3_w4),
            s22,
            extract_slot::<24>(z0_w5, z1_w5, z2_w5, z3_w5),
        ]);

        // Bucket 3 (rows 24..32): w5 slots 1..3, straddler 27, w6 slots 0..3.
        scatter_rows(3, out, [
            extract_slot::<17>(z0_w5, z1_w5, z2_w5, z3_w5),
            extract_slot::<10>(z0_w5, z1_w5, z2_w5, z3_w5),
            extract_slot::<3>(z0_w5, z1_w5, z2_w5, z3_w5),
            s27,
            extract_slot::<21>(z0_w6, z1_w6, z2_w6, z3_w6),
            extract_slot::<14>(z0_w6, z1_w6, z2_w6, z3_w6),
            extract_slot::<7>(z0_w6, z1_w6, z2_w6, z3_w6),
            extract_slot::<0>(z0_w6, z1_w6, z2_w6, z3_w6),
        ]);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn simd_lt_w7_avx2(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    use std::arch::x86_64::*;

    use super::compare_eq_w4::extract_slot;
    use super::compare_eq_w4::scatter_rows;
    use super::compare_eq_w5::straddler_lt;

    debug_assert_eq!(packed_chunk.len(), 224);

    let c7 = u32::from(c & 0x7F);

    unsafe {
        let c_vec = _mm256_set1_epi32(c7 as i32);

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
        let (p0_w5, p1_w5, p2_w5, p3_w5) = load4!(5);
        let (p0_w6, p1_w6, p2_w6, p3_w6) = load4!(6);

        // For W=7, lo mask = bits 0..5 of each field (6 bits) = L | L<<1 | ... | L<<5.
        macro_rules! lt_word {
            ($p:ident, $L:expr, $H:expr) => {{
                let c_packed = c7 * $L;
                let c_hi_word = c_packed & $H;
                let lo_mask_word =
                    $L | ($L << 1) | ($L << 2) | ($L << 3) | ($L << 4) | ($L << 5);
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
        let z0_w5 = lt_word!(p0_w5, L_W5, H_W5);
        let z1_w5 = lt_word!(p1_w5, L_W5, H_W5);
        let z2_w5 = lt_word!(p2_w5, L_W5, H_W5);
        let z3_w5 = lt_word!(p3_w5, L_W5, H_W5);
        let z0_w6 = lt_word!(p0_w6, L_W6, H_W6);
        let z1_w6 = lt_word!(p1_w6, L_W6, H_W6);
        let z2_w6 = lt_word!(p2_w6, L_W6, H_W6);
        let z3_w6 = lt_word!(p3_w6, L_W6, H_W6);

        let s4 = straddler_lt::<28, 4, 15, 7>(
            p0_w0, p1_w0, p2_w0, p3_w0, p0_w1, p1_w1, p2_w1, p3_w1, c_vec,
        );
        let s9 = straddler_lt::<31, 1, 1, 63>(
            p0_w1, p1_w1, p2_w1, p3_w1, p0_w2, p1_w2, p2_w2, p3_w2, c_vec,
        );
        let s13 = straddler_lt::<27, 5, 31, 3>(
            p0_w2, p1_w2, p2_w2, p3_w2, p0_w3, p1_w3, p2_w3, p3_w3, c_vec,
        );
        let s18 = straddler_lt::<30, 2, 3, 31>(
            p0_w3, p1_w3, p2_w3, p3_w3, p0_w4, p1_w4, p2_w4, p3_w4, c_vec,
        );
        let s22 = straddler_lt::<26, 6, 63, 1>(
            p0_w4, p1_w4, p2_w4, p3_w4, p0_w5, p1_w5, p2_w5, p3_w5, c_vec,
        );
        let s27 = straddler_lt::<29, 3, 7, 15>(
            p0_w5, p1_w5, p2_w5, p3_w5, p0_w6, p1_w6, p2_w6, p3_w6, c_vec,
        );

        scatter_rows(0, out, [
            extract_slot::<25>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<18>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<11>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<4>(z0_w0, z1_w0, z2_w0, z3_w0),
            s4,
            extract_slot::<22>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<15>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<8>(z0_w1, z1_w1, z2_w1, z3_w1),
        ]);
        scatter_rows(1, out, [
            extract_slot::<1>(z0_w1, z1_w1, z2_w1, z3_w1),
            s9,
            extract_slot::<19>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<12>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<5>(z0_w2, z1_w2, z2_w2, z3_w2),
            s13,
            extract_slot::<23>(z0_w3, z1_w3, z2_w3, z3_w3),
            extract_slot::<16>(z0_w3, z1_w3, z2_w3, z3_w3),
        ]);
        scatter_rows(2, out, [
            extract_slot::<9>(z0_w3, z1_w3, z2_w3, z3_w3),
            extract_slot::<2>(z0_w3, z1_w3, z2_w3, z3_w3),
            s18,
            extract_slot::<20>(z0_w4, z1_w4, z2_w4, z3_w4),
            extract_slot::<13>(z0_w4, z1_w4, z2_w4, z3_w4),
            extract_slot::<6>(z0_w4, z1_w4, z2_w4, z3_w4),
            s22,
            extract_slot::<24>(z0_w5, z1_w5, z2_w5, z3_w5),
        ]);
        scatter_rows(3, out, [
            extract_slot::<17>(z0_w5, z1_w5, z2_w5, z3_w5),
            extract_slot::<10>(z0_w5, z1_w5, z2_w5, z3_w5),
            extract_slot::<3>(z0_w5, z1_w5, z2_w5, z3_w5),
            s27,
            extract_slot::<21>(z0_w6, z1_w6, z2_w6, z3_w6),
            extract_slot::<14>(z0_w6, z1_w6, z2_w6, z3_w6),
            extract_slot::<7>(z0_w6, z1_w6, z2_w6, z3_w6),
            extract_slot::<0>(z0_w6, z1_w6, z2_w6, z3_w6),
        ]);
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn scalar_w7(packed_chunk: &[u32], c: u8, out: &mut [u64; 16], eq: bool) {
    debug_assert_eq!(packed_chunk.len(), 224);
    let c7 = u32::from(c & 0x7F);
    for row in 0..32usize {
        let start_bit = row * 7;
        let word = start_bit / 32;
        let bit = start_bit % 32;
        for lane in 0..32usize {
            let lo = packed_chunk[word * 32 + lane] >> bit;
            let val = if bit + 7 <= 32 {
                lo & 0x7F
            } else {
                let hi = packed_chunk[(word + 1) * 32 + lane] << (32 - bit);
                (lo | hi) & 0x7F
            };
            let i = fastlanes::FL_ORDER[row / 8] * 16 + (row % 8) * 128 + lane;
            let matches = if eq { val == c7 } else { val < c7 };
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

    fn pack_u32(values: &[u32; 1024]) -> [u32; 224] {
        let mut out = [0u32; 224];
        // SAFETY: `out` matches `128 * W / size_of::<u32>() = 224` for W=7.
        unsafe {
            BitPacking::unchecked_pack(7, values, &mut out);
        }
        out
    }

    fn bit(chunk_bits: &[u64; 16], i: usize) -> bool {
        (chunk_bits[i / 64] >> (i % 64)) & 1 != 0
    }

    #[test]
    fn eq_w7_matches_naive() {
        let mut values = [0u32; 1024];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 17 + 3) & 0x7F) as u32;
        }
        let packed = pack_u32(&values);

        for c in (0..128u8).step_by(7) {
            let mut got = [0u64; 16];
            swar_eq_w7_u32(&packed, c, &mut got);
            for i in 0..1024 {
                let expected = values[i] == u32::from(c);
                assert_eq!(bit(&got, i), expected, "i={i}, c={c}, value={}", values[i]);
            }
        }
    }

    #[test]
    fn lt_w7_matches_naive() {
        let mut values = [0u32; 1024];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 31 + 5) & 0x7F) as u32;
        }
        let packed = pack_u32(&values);

        for c in (0..128u8).step_by(7) {
            let mut got = [0u64; 16];
            swar_lt_w7_u32(&packed, c, &mut got);
            for i in 0..1024 {
                let expected = values[i] < u32::from(c);
                assert_eq!(bit(&got, i), expected, "i={i}, c={c}, value={}", values[i]);
            }
        }
    }
}
