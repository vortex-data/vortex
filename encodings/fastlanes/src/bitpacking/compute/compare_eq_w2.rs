// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::many_single_char_names)]

//! Hand-tuned in-range Eq/Lt kernel for `bit_width = 2` on `u32` storage.
//!
//! Each packed `u32` holds 16 2-bit elements (one lane × half-row group).  Knuth's
//! broadword zero-test `(xor - L) & !xor & H` with `L = 0x5555_5555`,
//! `H = 0xAAAA_AAAA` writes the match bit at the high bit of each 2-bit pair
//! (positions `2s + 1`).
//!
//! Two AVX2 blocks (one per `word_in_lane ∈ {0, 1}`), each producing 2 buckets of 8
//! row-bitmaps via `extract_slot::<SHIFT>`. The `Lt` kernel uses the standard Knuth
//! high/low split with the result projected up to the high bit of the pair so the
//! same SIMD extractor applies.

const L: u32 = 0x5555_5555;
const H: u32 = 0xAAAA_AAAA;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
pub(crate) fn swar_eq_w2_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    // SAFETY: target_feature=avx2 gates compilation here.
    unsafe { simd_eq_w2_avx2(packed_chunk, c, out) }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
pub(crate) fn swar_eq_w2_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    scalar_eq_w2(packed_chunk, c, out);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
pub(crate) fn swar_lt_w2_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    // SAFETY: target_feature=avx2 gates compilation here.
    unsafe { simd_lt_w2_avx2(packed_chunk, c, out) }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
pub(crate) fn swar_lt_w2_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    scalar_lt_w2(packed_chunk, c, out);
}

// ---------------------------------------------------------------------------
// AVX2 implementation.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn simd_eq_w2_avx2(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    use std::arch::x86_64::*;

    use super::compare_eq_w4::extract_slot;
    use super::compare_eq_w4::scatter_rows;

    debug_assert_eq!(packed_chunk.len(), 64);

    let c_low = u32::from(c & 0x03);
    let c_packed_word = c_low * L;

    // SAFETY: AVX2 guaranteed by cfg gate; pointer arithmetic stays in bounds because
    // `packed_chunk.len() == 64`.
    unsafe {
        let c_packed = _mm256_set1_epi32(c_packed_word as i32);
        let l_vec = _mm256_set1_epi32(L as i32);
        let h_vec = _mm256_set1_epi32(H as i32);

        // word_in_lane = 0 holds buckets 0 (rows 0..8) and 1 (rows 8..16).
        // word_in_lane = 1 holds buckets 2 (rows 16..24) and 3 (rows 24..32).
        // Inside a word, bucket b ∈ {0, 1} covers slots `b*8..b*8+8`, with result bit at
        // position `2*(b*8 + s) + 1`. SHIFT to MSB = `31 - (2*(b*8+s)+1) = 30 - 2*(b*8+s)`.
        //   b=0: SHIFTs 30, 28, 26, 24, 22, 20, 18, 16
        //   b=1: SHIFTs 14, 12, 10,  8,  6,  4,  2,  0
        macro_rules! emit_bucket {
            ($k:literal, $base:literal, $z0:ident, $z1:ident, $z2:ident, $z3:ident) => {{
                let row0 = extract_slot::<{ $base }>($z0, $z1, $z2, $z3);
                let row1 = extract_slot::<{ $base - 2 }>($z0, $z1, $z2, $z3);
                let row2 = extract_slot::<{ $base - 4 }>($z0, $z1, $z2, $z3);
                let row3 = extract_slot::<{ $base - 6 }>($z0, $z1, $z2, $z3);
                let row4 = extract_slot::<{ $base - 8 }>($z0, $z1, $z2, $z3);
                let row5 = extract_slot::<{ $base - 10 }>($z0, $z1, $z2, $z3);
                let row6 = extract_slot::<{ $base - 12 }>($z0, $z1, $z2, $z3);
                let row7 = extract_slot::<{ $base - 14 }>($z0, $z1, $z2, $z3);
                scatter_rows(
                    $k,
                    out,
                    [row0, row1, row2, row3, row4, row5, row6, row7],
                );
            }};
        }

        macro_rules! word_block {
            ($word_idx:literal, $k_lo:literal, $k_hi:literal) => {{
                let base_ptr = packed_chunk.as_ptr().add($word_idx * 32).cast::<__m256i>();
                let p0 = _mm256_loadu_si256(base_ptr.add(0));
                let p1 = _mm256_loadu_si256(base_ptr.add(1));
                let p2 = _mm256_loadu_si256(base_ptr.add(2));
                let p3 = _mm256_loadu_si256(base_ptr.add(3));

                let x0 = _mm256_xor_si256(p0, c_packed);
                let x1 = _mm256_xor_si256(p1, c_packed);
                let x2 = _mm256_xor_si256(p2, c_packed);
                let x3 = _mm256_xor_si256(p3, c_packed);
                let z0 = _mm256_and_si256(
                    _mm256_sub_epi32(x0, l_vec),
                    _mm256_andnot_si256(x0, h_vec),
                );
                let z1 = _mm256_and_si256(
                    _mm256_sub_epi32(x1, l_vec),
                    _mm256_andnot_si256(x1, h_vec),
                );
                let z2 = _mm256_and_si256(
                    _mm256_sub_epi32(x2, l_vec),
                    _mm256_andnot_si256(x2, h_vec),
                );
                let z3 = _mm256_and_si256(
                    _mm256_sub_epi32(x3, l_vec),
                    _mm256_andnot_si256(x3, h_vec),
                );

                emit_bucket!($k_lo, 30, z0, z1, z2, z3);
                emit_bucket!($k_hi, 14, z0, z1, z2, z3);
            }};
        }

        word_block!(0, 0, 1);
        word_block!(1, 2, 3);
    }
}

// AVX2 Lt for W=2.
//
// Per 2-bit pair:
//   a_hi = a & H            (bit 2s+1 of each pair)
//   a_lo = a & L            (bit 2s of each pair)
//   c_hi = c_packed & H, c_lo = c_packed & L
//   hi_lt = !a_hi & c_hi                    (set at bit 2s+1 where a_hi=0 and c_hi=1)
//   hi_eq = !(a_hi ^ c_hi) & H              (set at bit 2s+1 where the high bits match)
//   lo_lt = !a_lo & c_lo                    (set at bit 2s where a_lo=0 and c_lo=1)
//   lo_lt_hi = lo_lt << 1                   (project up to bit 2s+1)
//   lt = hi_lt | (hi_eq & lo_lt_hi)         (result at bit 2s+1)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn simd_lt_w2_avx2(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    use std::arch::x86_64::*;

    use super::compare_eq_w4::extract_slot;
    use super::compare_eq_w4::scatter_rows;

    debug_assert_eq!(packed_chunk.len(), 64);

    let c_low = u32::from(c & 0x03);
    let c_packed_word = c_low * L;
    let c_hi_word = c_packed_word & H;
    let c_lo_word = c_packed_word & L;

    // SAFETY: AVX2 guaranteed by cfg gate.
    unsafe {
        let c_hi = _mm256_set1_epi32(c_hi_word as i32);
        let c_lo = _mm256_set1_epi32(c_lo_word as i32);
        let h_vec = _mm256_set1_epi32(H as i32);
        let l_vec = _mm256_set1_epi32(L as i32);

        macro_rules! lt_one {
            ($a:ident) => {{
                let a_hi = _mm256_and_si256($a, h_vec);
                let a_lo = _mm256_and_si256($a, l_vec);
                let hi_lt = _mm256_andnot_si256(a_hi, c_hi);
                let hi_eq = _mm256_andnot_si256(_mm256_xor_si256(a_hi, c_hi), h_vec);
                let lo_lt = _mm256_andnot_si256(a_lo, c_lo);
                let lo_lt_hi = _mm256_slli_epi32::<1>(lo_lt);
                _mm256_or_si256(hi_lt, _mm256_and_si256(hi_eq, lo_lt_hi))
            }};
        }

        macro_rules! emit_bucket {
            ($k:literal, $base:literal, $z0:ident, $z1:ident, $z2:ident, $z3:ident) => {{
                let row0 = extract_slot::<{ $base }>($z0, $z1, $z2, $z3);
                let row1 = extract_slot::<{ $base - 2 }>($z0, $z1, $z2, $z3);
                let row2 = extract_slot::<{ $base - 4 }>($z0, $z1, $z2, $z3);
                let row3 = extract_slot::<{ $base - 6 }>($z0, $z1, $z2, $z3);
                let row4 = extract_slot::<{ $base - 8 }>($z0, $z1, $z2, $z3);
                let row5 = extract_slot::<{ $base - 10 }>($z0, $z1, $z2, $z3);
                let row6 = extract_slot::<{ $base - 12 }>($z0, $z1, $z2, $z3);
                let row7 = extract_slot::<{ $base - 14 }>($z0, $z1, $z2, $z3);
                scatter_rows(
                    $k,
                    out,
                    [row0, row1, row2, row3, row4, row5, row6, row7],
                );
            }};
        }

        macro_rules! word_block {
            ($word_idx:literal, $k_lo:literal, $k_hi:literal) => {{
                let base_ptr = packed_chunk.as_ptr().add($word_idx * 32).cast::<__m256i>();
                let p0 = _mm256_loadu_si256(base_ptr.add(0));
                let p1 = _mm256_loadu_si256(base_ptr.add(1));
                let p2 = _mm256_loadu_si256(base_ptr.add(2));
                let p3 = _mm256_loadu_si256(base_ptr.add(3));
                let z0 = lt_one!(p0);
                let z1 = lt_one!(p1);
                let z2 = lt_one!(p2);
                let z3 = lt_one!(p3);
                emit_bucket!($k_lo, 30, z0, z1, z2, z3);
                emit_bucket!($k_hi, 14, z0, z1, z2, z3);
            }};
        }

        word_block!(0, 0, 1);
        word_block!(1, 2, 3);
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback.
// ---------------------------------------------------------------------------

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn scalar_eq_w2(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    use super::compare_eq_w4::scatter_rows;

    debug_assert_eq!(packed_chunk.len(), 64);
    let c_packed = u32::from(c & 0x03) * L;

    for k in 0..4usize {
        let word_in_lane = k / 2;
        let slot_off = (k % 2) * 8;
        let base = (2 * slot_off + 1) as u32;

        let mut a = [0u32; 8];
        for lane in 0..32usize {
            let w = packed_chunk[word_in_lane * 32 + lane];
            let xor = w ^ c_packed;
            let zeros = xor.wrapping_sub(L) & !xor & H;

            a[0] |= ((zeros >> base) & 1) << lane;
            a[1] |= ((zeros >> (base + 2)) & 1) << lane;
            a[2] |= ((zeros >> (base + 4)) & 1) << lane;
            a[3] |= ((zeros >> (base + 6)) & 1) << lane;
            a[4] |= ((zeros >> (base + 8)) & 1) << lane;
            a[5] |= ((zeros >> (base + 10)) & 1) << lane;
            a[6] |= ((zeros >> (base + 12)) & 1) << lane;
            a[7] |= ((zeros >> (base + 14)) & 1) << lane;
        }
        scatter_rows(k, out, a);
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn scalar_lt_w2(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    use super::compare_eq_w4::scatter_rows;

    debug_assert_eq!(packed_chunk.len(), 64);
    let c_packed = u32::from(c & 0x03) * L;
    let c_hi = c_packed & H;
    let c_lo = c_packed & L;

    for k in 0..4usize {
        let word_in_lane = k / 2;
        let slot_off = (k % 2) * 8;
        let base = (2 * slot_off + 1) as u32;

        let mut a = [0u32; 8];
        for lane in 0..32usize {
            let w = packed_chunk[word_in_lane * 32 + lane];
            let a_hi = w & H;
            let a_lo = w & L;
            let hi_lt = !a_hi & c_hi;
            let hi_eq = !(a_hi ^ c_hi) & H;
            let lo_lt = !a_lo & c_lo;
            let lo_lt_hi = lo_lt << 1;
            let lt = hi_lt | (hi_eq & lo_lt_hi);

            a[0] |= ((lt >> base) & 1) << lane;
            a[1] |= ((lt >> (base + 2)) & 1) << lane;
            a[2] |= ((lt >> (base + 4)) & 1) << lane;
            a[3] |= ((lt >> (base + 6)) & 1) << lane;
            a[4] |= ((lt >> (base + 8)) & 1) << lane;
            a[5] |= ((lt >> (base + 10)) & 1) << lane;
            a[6] |= ((lt >> (base + 12)) & 1) << lane;
            a[7] |= ((lt >> (base + 14)) & 1) << lane;
        }
        scatter_rows(k, out, a);
    }
}

#[cfg(test)]
mod tests {
    use fastlanes::BitPacking;

    use super::*;

    fn pack_u32(values: &[u32; 1024]) -> [u32; 64] {
        let mut out = [0u32; 64];
        // SAFETY: `out` matches `128 * W / size_of::<u32>() = 64` for W=2.
        unsafe {
            BitPacking::unchecked_pack(2, values, &mut out);
        }
        out
    }

    fn bit(chunk_bits: &[u64; 16], i: usize) -> bool {
        (chunk_bits[i / 64] >> (i % 64)) & 1 != 0
    }

    #[test]
    fn eq_w2_matches_naive() {
        let mut values = [0u32; 1024];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 17 + 3) & 3) as u32;
        }
        let packed = pack_u32(&values);

        for c in 0..4u8 {
            let mut got = [0u64; 16];
            swar_eq_w2_u32(&packed, c, &mut got);
            for i in 0..1024 {
                let expected = values[i] == u32::from(c);
                assert_eq!(bit(&got, i), expected, "i={i}, c={c}, value={}", values[i]);
            }
        }
    }

    #[test]
    fn lt_w2_matches_naive() {
        let mut values = [0u32; 1024];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 31 + 5) & 3) as u32;
        }
        let packed = pack_u32(&values);

        for c in 0..4u8 {
            let mut got = [0u64; 16];
            swar_lt_w2_u32(&packed, c, &mut got);
            for i in 0..1024 {
                let expected = values[i] < u32::from(c);
                assert_eq!(bit(&got, i), expected, "i={i}, c={c}, value={}", values[i]);
            }
        }
    }
}
