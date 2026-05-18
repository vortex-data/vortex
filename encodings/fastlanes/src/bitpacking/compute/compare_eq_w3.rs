// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::many_single_char_names)]

//! Hand-tuned in-range Eq/Lt kernel for `bit_width = 3` on `u32` storage.
//!
//! With W=3 and T=32 the lane bit stream is 96 bits / 3 words. Slot starts rotate by 1 bit
//! per word (phases 0, 1, 2). Of the 32 rows per lane, 30 are "fully contained" in a single
//! word and 2 straddle the word boundary: row 10 spans word 0 bits 30..31 + word 1 bit 0,
//! and row 21 spans word 1 bit 31 + word 2 bits 0..1.
//!
//! Approach:
//!  * Per-word Knuth zero-test with rotated `(L, H)` masks gives the per-slot match bit at
//!    the slot's high position; `extract_slot::<SHIFT>` projects each slot to a 32-bit
//!    row bitmap.
//!  * Straddler rows are stitched scalar-style per lane in SIMD: combine the low/high bit
//!    fragments into a 3-bit `u32` per lane, then `cmpeq_epi32`/`min_epu32` against the
//!    broadcast constant.
//!
//! FL_ORDER buckets cross word boundaries for buckets 1 and 2 (which contain the straddler
//! rows). Buckets 0 and 3 are entirely within one word.

const L_W0: u32 = 0x0924_9249;
const H_W0: u32 = 0x2492_4924;
const L_W1: u32 = 0x1249_2492;
const H_W1: u32 = 0x4924_9248;
const L_W2: u32 = 0x2492_4924;
const H_W2: u32 = 0x9249_2490;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
pub(crate) fn swar_eq_w3_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    // SAFETY: target_feature=avx2 gates compilation here.
    unsafe { simd_eq_w3_avx2(packed_chunk, c, out) }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
pub(crate) fn swar_eq_w3_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    scalar_eq_w3(packed_chunk, c, out);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
pub(crate) fn swar_lt_w3_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    // SAFETY: target_feature=avx2 gates compilation here.
    unsafe { simd_lt_w3_avx2(packed_chunk, c, out) }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
pub(crate) fn swar_lt_w3_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    scalar_lt_w3(packed_chunk, c, out);
}

// ---------------------------------------------------------------------------
// AVX2 implementation.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn simd_eq_w3_avx2(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    use std::arch::x86_64::*;

    use super::compare_eq_w4::extract_slot;
    use super::compare_eq_w4::scatter_rows;

    debug_assert_eq!(packed_chunk.len(), 96);

    let c3 = u32::from(c & 0x07);

    // SAFETY: AVX2 guaranteed by cfg gate; pointer arithmetic stays in bounds because
    // `packed_chunk.len() == 96`.
    unsafe {
        let c_vec = _mm256_set1_epi32(c3 as i32);
        let c_w0 = _mm256_set1_epi32((c3 * L_W0) as i32);
        let c_w1 = _mm256_set1_epi32((c3 * L_W1) as i32);
        let c_w2 = _mm256_set1_epi32((c3 * L_W2) as i32);
        let l_w0 = _mm256_set1_epi32(L_W0 as i32);
        let l_w1 = _mm256_set1_epi32(L_W1 as i32);
        let l_w2 = _mm256_set1_epi32(L_W2 as i32);
        let h_w0 = _mm256_set1_epi32(H_W0 as i32);
        let h_w1 = _mm256_set1_epi32(H_W1 as i32);
        let h_w2 = _mm256_set1_epi32(H_W2 as i32);

        // Load 3 banks of 32 lane words.
        let base = packed_chunk.as_ptr().cast::<__m256i>();
        let p0_w0 = _mm256_loadu_si256(base.add(0));
        let p1_w0 = _mm256_loadu_si256(base.add(1));
        let p2_w0 = _mm256_loadu_si256(base.add(2));
        let p3_w0 = _mm256_loadu_si256(base.add(3));
        let p0_w1 = _mm256_loadu_si256(base.add(4));
        let p1_w1 = _mm256_loadu_si256(base.add(5));
        let p2_w1 = _mm256_loadu_si256(base.add(6));
        let p3_w1 = _mm256_loadu_si256(base.add(7));
        let p0_w2 = _mm256_loadu_si256(base.add(8));
        let p1_w2 = _mm256_loadu_si256(base.add(9));
        let p2_w2 = _mm256_loadu_si256(base.add(10));
        let p3_w2 = _mm256_loadu_si256(base.add(11));

        // Per-word Knuth zero-test.
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

        // Straddler eq: combine 3 bits per lane into a single 3-bit value then `cmpeq`.
        let straddler_10 = straddler_eq_row10(
            p0_w0, p1_w0, p2_w0, p3_w0, p0_w1, p1_w1, p2_w1, p3_w1, c_vec,
        );
        let straddler_21 = straddler_eq_row21(
            p0_w1, p1_w1, p2_w1, p3_w1, p0_w2, p1_w2, p2_w2, p3_w2, c_vec,
        );

        // Bucket 0: rows 0..8 = word 0 slots 0..8.
        // SHIFT = 29 - 3s for s in 0..8.
        scatter_rows(0, out, [
            extract_slot::<29>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<26>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<23>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<20>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<17>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<14>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<11>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<8>(z0_w0, z1_w0, z2_w0, z3_w0),
        ]);

        // Bucket 1: rows 8..16.
        //   row  8 = word 0 slot 8  → SHIFT 5
        //   row  9 = word 0 slot 9  → SHIFT 2
        //   row 10 = straddler
        //   row 11..15 = word 1 slots 0..4 (phase 1; SHIFT = 28 - 3s)
        scatter_rows(1, out, [
            extract_slot::<5>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<2>(z0_w0, z1_w0, z2_w0, z3_w0),
            straddler_10,
            extract_slot::<28>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<25>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<22>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<19>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<16>(z0_w1, z1_w1, z2_w1, z3_w1),
        ]);

        // Bucket 2: rows 16..24.
        //   row 16..20 = word 1 slots 5..9 (SHIFT 13, 10, 7, 4, 1)
        //   row 21 = straddler
        //   row 22 = word 2 slot 0 → SHIFT 27
        //   row 23 = word 2 slot 1 → SHIFT 24
        scatter_rows(2, out, [
            extract_slot::<13>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<10>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<7>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<4>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<1>(z0_w1, z1_w1, z2_w1, z3_w1),
            straddler_21,
            extract_slot::<27>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<24>(z0_w2, z1_w2, z2_w2, z3_w2),
        ]);

        // Bucket 3: rows 24..32 = word 2 slots 2..10 (SHIFT 21..0 in steps of 3).
        scatter_rows(3, out, [
            extract_slot::<21>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<18>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<15>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<12>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<9>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<6>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<3>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<0>(z0_w2, z1_w2, z2_w2, z3_w2),
        ]);
    }
}

// Row 10 straddler: value = ((word_0 >> 30) & 3) | ((word_1 & 1) << 2). Eq via cmpeq_epi32.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn straddler_eq_row10(
    p0_w0: std::arch::x86_64::__m256i,
    p1_w0: std::arch::x86_64::__m256i,
    p2_w0: std::arch::x86_64::__m256i,
    p3_w0: std::arch::x86_64::__m256i,
    p0_w1: std::arch::x86_64::__m256i,
    p1_w1: std::arch::x86_64::__m256i,
    p2_w1: std::arch::x86_64::__m256i,
    p3_w1: std::arch::x86_64::__m256i,
    c_vec: std::arch::x86_64::__m256i,
) -> u32 {
    use std::arch::x86_64::*;
    // SAFETY: AVX2 guaranteed.
    unsafe {
        let three = _mm256_set1_epi32(3);
        let one = _mm256_set1_epi32(1);
        macro_rules! row10 {
            ($w0:ident, $w1:ident) => {{
                let lo = _mm256_and_si256(_mm256_srli_epi32::<30>($w0), three);
                let hi = _mm256_slli_epi32::<2>(_mm256_and_si256($w1, one));
                let val = _mm256_or_si256(lo, hi);
                let cmp = _mm256_cmpeq_epi32(val, c_vec);
                _mm256_movemask_ps(_mm256_castsi256_ps(cmp)) as u32 & 0xFF
            }};
        }
        let m0 = row10!(p0_w0, p0_w1);
        let m1 = row10!(p1_w0, p1_w1);
        let m2 = row10!(p2_w0, p2_w1);
        let m3 = row10!(p3_w0, p3_w1);
        m0 | (m1 << 8) | (m2 << 16) | (m3 << 24)
    }
}

// Row 21 straddler: value = ((word_1 >> 31) & 1) | ((word_2 & 3) << 1).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn straddler_eq_row21(
    p0_w1: std::arch::x86_64::__m256i,
    p1_w1: std::arch::x86_64::__m256i,
    p2_w1: std::arch::x86_64::__m256i,
    p3_w1: std::arch::x86_64::__m256i,
    p0_w2: std::arch::x86_64::__m256i,
    p1_w2: std::arch::x86_64::__m256i,
    p2_w2: std::arch::x86_64::__m256i,
    p3_w2: std::arch::x86_64::__m256i,
    c_vec: std::arch::x86_64::__m256i,
) -> u32 {
    use std::arch::x86_64::*;
    unsafe {
        let three = _mm256_set1_epi32(3);
        macro_rules! row21 {
            ($w1:ident, $w2:ident) => {{
                let lo = _mm256_srli_epi32::<31>($w1);
                let hi = _mm256_slli_epi32::<1>(_mm256_and_si256($w2, three));
                let val = _mm256_or_si256(lo, hi);
                let cmp = _mm256_cmpeq_epi32(val, c_vec);
                _mm256_movemask_ps(_mm256_castsi256_ps(cmp)) as u32 & 0xFF
            }};
        }
        let m0 = row21!(p0_w1, p0_w2);
        let m1 = row21!(p1_w1, p1_w2);
        let m2 = row21!(p2_w1, p2_w2);
        let m3 = row21!(p3_w1, p3_w2);
        m0 | (m1 << 8) | (m2 << 16) | (m3 << 24)
    }
}

// Lt for W=3 uses the same SHIFT layout. Per word: Knuth high/low split per 3-bit field.
//   a_hi (bit 3s+2), a_lo (bits 3s, 3s+1).
//   hi_lt = !a_hi & c_hi
//   hi_eq = !(a_hi ^ c_hi) & H_w
//   For 2-bit lo: lo_le = ((c_lo | H_w) - a_lo) & H_w
//                 lo_eq = ((a_lo ^ c_lo) - L_w) & !(a_lo ^ c_lo) & H_w
//                 lo_lt = lo_le & !lo_eq
//   lt = hi_lt | (hi_eq & lo_lt)
// Result lands at the high bit (3s+2) of each slot.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn simd_lt_w3_avx2(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    use std::arch::x86_64::*;

    use super::compare_eq_w4::extract_slot;
    use super::compare_eq_w4::scatter_rows;

    debug_assert_eq!(packed_chunk.len(), 96);

    let c3 = u32::from(c & 0x07);

    unsafe {
        let c_vec = _mm256_set1_epi32(c3 as i32);

        macro_rules! lt_one_word {
            ($p:ident, $L:expr, $H:expr) => {{
                let c_packed = c3 * $L;
                let c_hi_word = c_packed & $H;
                let c_lo_word = c_packed & ($L | ($L << 1)); // bits 3s, 3s+1 of each field
                let l_vec = _mm256_set1_epi32($L as i32);
                let h_vec = _mm256_set1_epi32($H as i32);
                let m_vec = _mm256_set1_epi32(($L | ($L << 1)) as i32);
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

        // Load 3 banks.
        let base = packed_chunk.as_ptr().cast::<__m256i>();
        let p0_w0 = _mm256_loadu_si256(base.add(0));
        let p1_w0 = _mm256_loadu_si256(base.add(1));
        let p2_w0 = _mm256_loadu_si256(base.add(2));
        let p3_w0 = _mm256_loadu_si256(base.add(3));
        let p0_w1 = _mm256_loadu_si256(base.add(4));
        let p1_w1 = _mm256_loadu_si256(base.add(5));
        let p2_w1 = _mm256_loadu_si256(base.add(6));
        let p3_w1 = _mm256_loadu_si256(base.add(7));
        let p0_w2 = _mm256_loadu_si256(base.add(8));
        let p1_w2 = _mm256_loadu_si256(base.add(9));
        let p2_w2 = _mm256_loadu_si256(base.add(10));
        let p3_w2 = _mm256_loadu_si256(base.add(11));

        let z0_w0 = lt_one_word!(p0_w0, L_W0, H_W0);
        let z1_w0 = lt_one_word!(p1_w0, L_W0, H_W0);
        let z2_w0 = lt_one_word!(p2_w0, L_W0, H_W0);
        let z3_w0 = lt_one_word!(p3_w0, L_W0, H_W0);
        let z0_w1 = lt_one_word!(p0_w1, L_W1, H_W1);
        let z1_w1 = lt_one_word!(p1_w1, L_W1, H_W1);
        let z2_w1 = lt_one_word!(p2_w1, L_W1, H_W1);
        let z3_w1 = lt_one_word!(p3_w1, L_W1, H_W1);
        let z0_w2 = lt_one_word!(p0_w2, L_W2, H_W2);
        let z1_w2 = lt_one_word!(p1_w2, L_W2, H_W2);
        let z2_w2 = lt_one_word!(p2_w2, L_W2, H_W2);
        let z3_w2 = lt_one_word!(p3_w2, L_W2, H_W2);

        // Straddler lt: reconstruct val per lane then `min_epu32` identity.
        let straddler_10 = straddler_lt_row10(
            p0_w0, p1_w0, p2_w0, p3_w0, p0_w1, p1_w1, p2_w1, p3_w1, c_vec,
        );
        let straddler_21 = straddler_lt_row21(
            p0_w1, p1_w1, p2_w1, p3_w1, p0_w2, p1_w2, p2_w2, p3_w2, c_vec,
        );

        scatter_rows(0, out, [
            extract_slot::<29>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<26>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<23>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<20>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<17>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<14>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<11>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<8>(z0_w0, z1_w0, z2_w0, z3_w0),
        ]);
        scatter_rows(1, out, [
            extract_slot::<5>(z0_w0, z1_w0, z2_w0, z3_w0),
            extract_slot::<2>(z0_w0, z1_w0, z2_w0, z3_w0),
            straddler_10,
            extract_slot::<28>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<25>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<22>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<19>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<16>(z0_w1, z1_w1, z2_w1, z3_w1),
        ]);
        scatter_rows(2, out, [
            extract_slot::<13>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<10>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<7>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<4>(z0_w1, z1_w1, z2_w1, z3_w1),
            extract_slot::<1>(z0_w1, z1_w1, z2_w1, z3_w1),
            straddler_21,
            extract_slot::<27>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<24>(z0_w2, z1_w2, z2_w2, z3_w2),
        ]);
        scatter_rows(3, out, [
            extract_slot::<21>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<18>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<15>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<12>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<9>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<6>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<3>(z0_w2, z1_w2, z2_w2, z3_w2),
            extract_slot::<0>(z0_w2, z1_w2, z2_w2, z3_w2),
        ]);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn straddler_lt_row10(
    p0_w0: std::arch::x86_64::__m256i,
    p1_w0: std::arch::x86_64::__m256i,
    p2_w0: std::arch::x86_64::__m256i,
    p3_w0: std::arch::x86_64::__m256i,
    p0_w1: std::arch::x86_64::__m256i,
    p1_w1: std::arch::x86_64::__m256i,
    p2_w1: std::arch::x86_64::__m256i,
    p3_w1: std::arch::x86_64::__m256i,
    c_vec: std::arch::x86_64::__m256i,
) -> u32 {
    use std::arch::x86_64::*;
    unsafe {
        let three = _mm256_set1_epi32(3);
        let one = _mm256_set1_epi32(1);
        macro_rules! row10 {
            ($w0:ident, $w1:ident) => {{
                let lo = _mm256_and_si256(_mm256_srli_epi32::<30>($w0), three);
                let hi = _mm256_slli_epi32::<2>(_mm256_and_si256($w1, one));
                let val = _mm256_or_si256(lo, hi);
                let min = _mm256_min_epu32(val, c_vec);
                let le = _mm256_cmpeq_epi32(min, val);
                let eq = _mm256_cmpeq_epi32(val, c_vec);
                let lt = _mm256_andnot_si256(eq, le);
                _mm256_movemask_ps(_mm256_castsi256_ps(lt)) as u32 & 0xFF
            }};
        }
        let m0 = row10!(p0_w0, p0_w1);
        let m1 = row10!(p1_w0, p1_w1);
        let m2 = row10!(p2_w0, p2_w1);
        let m3 = row10!(p3_w0, p3_w1);
        m0 | (m1 << 8) | (m2 << 16) | (m3 << 24)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn straddler_lt_row21(
    p0_w1: std::arch::x86_64::__m256i,
    p1_w1: std::arch::x86_64::__m256i,
    p2_w1: std::arch::x86_64::__m256i,
    p3_w1: std::arch::x86_64::__m256i,
    p0_w2: std::arch::x86_64::__m256i,
    p1_w2: std::arch::x86_64::__m256i,
    p2_w2: std::arch::x86_64::__m256i,
    p3_w2: std::arch::x86_64::__m256i,
    c_vec: std::arch::x86_64::__m256i,
) -> u32 {
    use std::arch::x86_64::*;
    unsafe {
        let three = _mm256_set1_epi32(3);
        macro_rules! row21 {
            ($w1:ident, $w2:ident) => {{
                let lo = _mm256_srli_epi32::<31>($w1);
                let hi = _mm256_slli_epi32::<1>(_mm256_and_si256($w2, three));
                let val = _mm256_or_si256(lo, hi);
                let min = _mm256_min_epu32(val, c_vec);
                let le = _mm256_cmpeq_epi32(min, val);
                let eq = _mm256_cmpeq_epi32(val, c_vec);
                let lt = _mm256_andnot_si256(eq, le);
                _mm256_movemask_ps(_mm256_castsi256_ps(lt)) as u32 & 0xFF
            }};
        }
        let m0 = row21!(p0_w1, p0_w2);
        let m1 = row21!(p1_w1, p1_w2);
        let m2 = row21!(p2_w1, p2_w2);
        let m3 = row21!(p3_w1, p3_w2);
        m0 | (m1 << 8) | (m2 << 16) | (m3 << 24)
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback.
// ---------------------------------------------------------------------------

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn scalar_eq_w3(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    // Walk rows directly via the FastLanes index formula.
    debug_assert_eq!(packed_chunk.len(), 96);
    let c3 = u32::from(c & 0x07);
    for row in 0..32usize {
        let start_bit = row * 3;
        let word = start_bit / 32;
        let bit = start_bit % 32;
        for lane in 0..32usize {
            let lo = packed_chunk[word * 32 + lane] >> bit;
            let val = if bit + 3 <= 32 {
                lo & 0x7
            } else {
                let hi = packed_chunk[(word + 1) * 32 + lane] << (32 - bit);
                (lo | hi) & 0x7
            };
            let i = fastlanes::FL_ORDER[row / 8] * 16 + (row % 8) * 128 + lane;
            if val == c3 {
                out[i / 64] |= 1u64 << (i % 64);
            }
        }
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn scalar_lt_w3(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    debug_assert_eq!(packed_chunk.len(), 96);
    let c3 = u32::from(c & 0x07);
    for row in 0..32usize {
        let start_bit = row * 3;
        let word = start_bit / 32;
        let bit = start_bit % 32;
        for lane in 0..32usize {
            let lo = packed_chunk[word * 32 + lane] >> bit;
            let val = if bit + 3 <= 32 {
                lo & 0x7
            } else {
                let hi = packed_chunk[(word + 1) * 32 + lane] << (32 - bit);
                (lo | hi) & 0x7
            };
            let i = fastlanes::FL_ORDER[row / 8] * 16 + (row % 8) * 128 + lane;
            if val < c3 {
                out[i / 64] |= 1u64 << (i % 64);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use fastlanes::BitPacking;

    use super::*;

    fn pack_u32(values: &[u32; 1024]) -> [u32; 96] {
        let mut out = [0u32; 96];
        // SAFETY: `out` matches `128 * W / size_of::<u32>() = 96` for W=3.
        unsafe {
            BitPacking::unchecked_pack(3, values, &mut out);
        }
        out
    }

    fn bit(chunk_bits: &[u64; 16], i: usize) -> bool {
        (chunk_bits[i / 64] >> (i % 64)) & 1 != 0
    }

    #[test]
    fn eq_w3_matches_naive() {
        let mut values = [0u32; 1024];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 17 + 3) & 0x7) as u32;
        }
        let packed = pack_u32(&values);

        for c in 0..8u8 {
            let mut got = [0u64; 16];
            swar_eq_w3_u32(&packed, c, &mut got);
            for i in 0..1024 {
                let expected = values[i] == u32::from(c);
                assert_eq!(bit(&got, i), expected, "i={i}, c={c}, value={}", values[i]);
            }
        }
    }

    #[test]
    fn lt_w3_matches_naive() {
        let mut values = [0u32; 1024];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 31 + 5) & 0x7) as u32;
        }
        let packed = pack_u32(&values);

        for c in 0..8u8 {
            let mut got = [0u64; 16];
            swar_lt_w3_u32(&packed, c, &mut got);
            for i in 0..1024 {
                let expected = values[i] < u32::from(c);
                assert_eq!(bit(&got, i), expected, "i={i}, c={c}, value={}", values[i]);
            }
        }
    }
}
