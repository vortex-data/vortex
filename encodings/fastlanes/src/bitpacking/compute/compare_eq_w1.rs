// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::many_single_char_names)]

//! Hand-tuned in-range Eq/Lt kernel for `bit_width = 1` on `u32` storage.
//!
//! Each packed `u32` holds 32 1-bit elements for one lane. With `c ∈ {0, 1}` the entire
//! per-element compare reduces to bit-tests on the lane word: `Eq` is `bit_r == c_bit`,
//! `Lt` is either all-zero (c=0) or `bit_r == 0` (c=1).
//!
//! AVX2 path: 4 ymm cover all 32 lane words; per row `r` we shift bit `r` to position 31
//! and `movemask_ps` extracts 8 lane bits, exactly as in the W=4 kernel.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
pub(crate) fn swar_eq_w1_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    // SAFETY: target_feature=avx2 gates compilation here.
    unsafe { simd_eq_w1_avx2(packed_chunk, c, out) }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
pub(crate) fn swar_eq_w1_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    scalar_eq_w1(packed_chunk, c, out);
}

#[inline]
pub(crate) fn swar_lt_w1_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    // `c==0`: nothing in `[0, 1]` is `< 0`. The caller pre-zeroes `out`.
    // `c==1`: `a < 1` iff `a == 0`.
    if c & 1 == 0 {
        return;
    }
    swar_eq_w1_u32(packed_chunk, 0, out);
}

// ---------------------------------------------------------------------------
// AVX2 implementation.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn simd_eq_w1_avx2(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    use std::arch::x86_64::*;

    use super::compare_eq_w4::extract_slot;
    use super::compare_eq_w4::scatter_rows;

    debug_assert_eq!(packed_chunk.len(), 32);

    // SAFETY: AVX2 guaranteed by cfg gate; pointer arithmetic stays in bounds because
    // `packed_chunk.len() == 32`.
    unsafe {
        let p = packed_chunk.as_ptr().cast::<__m256i>();
        let p0 = _mm256_loadu_si256(p.add(0));
        let p1 = _mm256_loadu_si256(p.add(1));
        let p2 = _mm256_loadu_si256(p.add(2));
        let p3 = _mm256_loadu_si256(p.add(3));

        // For each row r: result bit = (lane word bit r) == c_bit.
        //   c_bit == 1  →  z = p
        //   c_bit == 0  →  z = !p
        let (z0, z1, z2, z3) = if c & 1 == 1 {
            (p0, p1, p2, p3)
        } else {
            let ones = _mm256_set1_epi32(-1);
            (
                _mm256_xor_si256(p0, ones),
                _mm256_xor_si256(p1, ones),
                _mm256_xor_si256(p2, ones),
                _mm256_xor_si256(p3, ones),
            )
        };

        // For bucket k ∈ 0..4, the 8 rows correspond to bits (k*8)..(k*8+8).
        // SHIFT = 31 - bit_index moves that bit to MSB for `movemask_ps` extraction.
        macro_rules! bucket {
            ($k:literal, $base:literal) => {{
                let row0 = extract_slot::<{ $base }>(z0, z1, z2, z3);
                let row1 = extract_slot::<{ $base - 1 }>(z0, z1, z2, z3);
                let row2 = extract_slot::<{ $base - 2 }>(z0, z1, z2, z3);
                let row3 = extract_slot::<{ $base - 3 }>(z0, z1, z2, z3);
                let row4 = extract_slot::<{ $base - 4 }>(z0, z1, z2, z3);
                let row5 = extract_slot::<{ $base - 5 }>(z0, z1, z2, z3);
                let row6 = extract_slot::<{ $base - 6 }>(z0, z1, z2, z3);
                let row7 = extract_slot::<{ $base - 7 }>(z0, z1, z2, z3);
                scatter_rows(
                    $k,
                    out,
                    [row0, row1, row2, row3, row4, row5, row6, row7],
                );
            }};
        }
        bucket!(0, 31);
        bucket!(1, 23);
        bucket!(2, 15);
        bucket!(3, 7);
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback.
// ---------------------------------------------------------------------------

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn scalar_eq_w1(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    use super::compare_eq_w4::scatter_rows;

    debug_assert_eq!(packed_chunk.len(), 32);
    // z[lane] has bit r set iff value at (row r, lane) equals c.
    let xor_mask = if c & 1 == 1 { 0u32 } else { !0u32 };

    for k in 0..4usize {
        let mut a0 = 0u32;
        let mut a1 = 0u32;
        let mut a2 = 0u32;
        let mut a3 = 0u32;
        let mut a4 = 0u32;
        let mut a5 = 0u32;
        let mut a6 = 0u32;
        let mut a7 = 0u32;

        let base = k * 8;
        for lane in 0..32usize {
            let z = packed_chunk[lane] ^ xor_mask;
            a0 |= ((z >> base) & 1) << lane;
            a1 |= ((z >> (base + 1)) & 1) << lane;
            a2 |= ((z >> (base + 2)) & 1) << lane;
            a3 |= ((z >> (base + 3)) & 1) << lane;
            a4 |= ((z >> (base + 4)) & 1) << lane;
            a5 |= ((z >> (base + 5)) & 1) << lane;
            a6 |= ((z >> (base + 6)) & 1) << lane;
            a7 |= ((z >> (base + 7)) & 1) << lane;
        }

        scatter_rows(k, out, [a0, a1, a2, a3, a4, a5, a6, a7]);
    }
}

#[cfg(test)]
mod tests {
    use fastlanes::BitPacking;

    use super::*;

    fn pack_u32(values: &[u32; 1024]) -> [u32; 32] {
        let mut out = [0u32; 32];
        // SAFETY: `out` matches `128 * W / size_of::<u32>() = 32` for W=1.
        unsafe {
            BitPacking::unchecked_pack(1, values, &mut out);
        }
        out
    }

    fn bit(chunk_bits: &[u64; 16], i: usize) -> bool {
        (chunk_bits[i / 64] >> (i % 64)) & 1 != 0
    }

    #[test]
    fn eq_w1_matches_naive() {
        let mut values = [0u32; 1024];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 17 + 3) & 1) as u32;
        }
        let packed = pack_u32(&values);

        for c in 0..2u8 {
            let mut got = [0u64; 16];
            swar_eq_w1_u32(&packed, c, &mut got);
            for i in 0..1024 {
                let expected = values[i] == u32::from(c);
                assert_eq!(bit(&got, i), expected, "i={i}, c={c}, value={}", values[i]);
            }
        }
    }

    #[test]
    fn lt_w1_matches_naive() {
        let mut values = [0u32; 1024];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 31 + 5) & 1) as u32;
        }
        let packed = pack_u32(&values);

        for c in 0..2u8 {
            let mut got = [0u64; 16];
            swar_lt_w1_u32(&packed, c, &mut got);
            for i in 0..1024 {
                let expected = values[i] < u32::from(c);
                assert_eq!(bit(&got, i), expected, "i={i}, c={c}, value={}", values[i]);
            }
        }
    }
}
