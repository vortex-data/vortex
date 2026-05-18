// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::many_single_char_names)]

//! Hand-tuned in-range Eq kernel for `bit_width = 4` on `u32` storage.
//!
//! Each packed `u32` word holds 8 nibbles. Per word, Knuth's broadword zero-test
//! `(xor - 0x1111_1111) & !xor & 0x8888_8888` writes the per-slot matched bit at the
//! high bit of each nibble (positions 3, 7, 11, ..., 31).
//!
//! Two paths:
//!  * AVX2 fast path: 4 ymm registers of `zeros` per word, then 8 ×
//!    (`slli_epi32`+`movemask_ps`) to extract one row-bitmap per slot.
//!  * Scalar fallback: hand-unrolled 32-lane inner loop with 8 independent accumulators.

use fastlanes::FL_ORDER;

const L: u32 = 0x11111111;
const H: u32 = 0x88888888;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
pub(crate) fn swar_eq_w4_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    unsafe { simd_eq_w4_avx2(packed_chunk, c, out) }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
pub(crate) fn swar_eq_w4_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    scalar_eq_w4(packed_chunk, c, out);
}

// ---------------------------------------------------------------------------
// AVX2 implementation.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn simd_eq_w4_avx2(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(packed_chunk.len(), 128);

    let c_packed_word: u32 = u32::from(c & 0x0F) * L;
    // SAFETY: target_feature=avx2 guaranteed by the cfg gate.
    unsafe {
        let c_packed = _mm256_set1_epi32(c_packed_word as i32);
        let l_mask = _mm256_set1_epi32(L as i32);
        let h_mask = _mm256_set1_epi32(H as i32);

        for k in 0..4usize {
            let p = packed_chunk.as_ptr().add(k * 32).cast::<__m256i>();
            // Load 4 ymm covering 32 lanes' worth of packed u32 for this k.
            let p0 = _mm256_loadu_si256(p.add(0));
            let p1 = _mm256_loadu_si256(p.add(1));
            let p2 = _mm256_loadu_si256(p.add(2));
            let p3 = _mm256_loadu_si256(p.add(3));

            // XOR with c_packed and apply Knuth broadword zero-test per nibble.
            let x0 = _mm256_xor_si256(p0, c_packed);
            let x1 = _mm256_xor_si256(p1, c_packed);
            let x2 = _mm256_xor_si256(p2, c_packed);
            let x3 = _mm256_xor_si256(p3, c_packed);
            let z0 = _mm256_and_si256(
                _mm256_sub_epi32(x0, l_mask),
                _mm256_andnot_si256(x0, h_mask),
            );
            let z1 = _mm256_and_si256(
                _mm256_sub_epi32(x1, l_mask),
                _mm256_andnot_si256(x1, h_mask),
            );
            let z2 = _mm256_and_si256(
                _mm256_sub_epi32(x2, l_mask),
                _mm256_andnot_si256(x2, h_mask),
            );
            let z3 = _mm256_and_si256(
                _mm256_sub_epi32(x3, l_mask),
                _mm256_andnot_si256(x3, h_mask),
            );

            // For each slot s ∈ 0..8: shift bit (4s+3) of each u32 to bit 31, then
            // movemask_ps to collect 8 lane-bits. Combine the 4 ymms → 32-bit row bitmap.
            let row0 = extract_slot::<28>(z0, z1, z2, z3);
            let row1 = extract_slot::<24>(z0, z1, z2, z3);
            let row2 = extract_slot::<20>(z0, z1, z2, z3);
            let row3 = extract_slot::<16>(z0, z1, z2, z3);
            let row4 = extract_slot::<12>(z0, z1, z2, z3);
            let row5 = extract_slot::<8>(z0, z1, z2, z3);
            let row6 = extract_slot::<4>(z0, z1, z2, z3);
            let row7 = extract_slot::<0>(z0, z1, z2, z3);

            scatter_rows(k, out, [row0, row1, row2, row3, row4, row5, row6, row7]);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn extract_slot<const SHIFT: i32>(
    z0: std::arch::x86_64::__m256i,
    z1: std::arch::x86_64::__m256i,
    z2: std::arch::x86_64::__m256i,
    z3: std::arch::x86_64::__m256i,
) -> u32 {
    use std::arch::x86_64::*;
    unsafe {
        let m0 =
            _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32::<SHIFT>(z0))) as u32 & 0xFF;
        let m1 =
            _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32::<SHIFT>(z1))) as u32 & 0xFF;
        let m2 =
            _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32::<SHIFT>(z2))) as u32 & 0xFF;
        let m3 =
            _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32::<SHIFT>(z3))) as u32 & 0xFF;
        m0 | (m1 << 8) | (m2 << 16) | (m3 << 24)
    }
}

#[inline(always)]
fn scatter_rows(k: usize, out: &mut [u64; 16], rows: [u32; 8]) {
    // For W=4 row r=k*8+s; elem_base = FL_ORDER[k]*16 + s*128. Per-k the 8 row
    // bitmaps go to 8 different u64s at the same bit offset:
    //   k=0 (FL_ORDER[0]=0) → u_base=0, bit_off= 0
    //   k=1 (FL_ORDER[1]=4) → u_base=1, bit_off= 0
    //   k=2 (FL_ORDER[2]=2) → u_base=0, bit_off=32
    //   k=3 (FL_ORDER[3]=6) → u_base=1, bit_off=32
    let fl = FL_ORDER[k];
    let u_base = fl / 4;
    let bit_offset = ((fl % 4) * 16) as u64;

    out[u_base] |= (rows[0] as u64) << bit_offset;
    out[u_base + 2] |= (rows[1] as u64) << bit_offset;
    out[u_base + 4] |= (rows[2] as u64) << bit_offset;
    out[u_base + 6] |= (rows[3] as u64) << bit_offset;
    out[u_base + 8] |= (rows[4] as u64) << bit_offset;
    out[u_base + 10] |= (rows[5] as u64) << bit_offset;
    out[u_base + 12] |= (rows[6] as u64) << bit_offset;
    out[u_base + 14] |= (rows[7] as u64) << bit_offset;
}

// ---------------------------------------------------------------------------
// Scalar fallback.
// ---------------------------------------------------------------------------

#[inline]
fn scalar_eq_w4(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    debug_assert_eq!(packed_chunk.len(), 128);
    let c_packed: u32 = u32::from(c & 0x0F) * L;

    for k in 0..4usize {
        let mut a0: u32 = 0;
        let mut a1: u32 = 0;
        let mut a2: u32 = 0;
        let mut a3: u32 = 0;
        let mut a4: u32 = 0;
        let mut a5: u32 = 0;
        let mut a6: u32 = 0;
        let mut a7: u32 = 0;

        for lane in 0..32usize {
            let word = packed_chunk[k * 32 + lane];
            let xor = word ^ c_packed;
            let zeros = xor.wrapping_sub(L) & !xor & H;

            a0 |= ((zeros >> 3) & 1) << lane;
            a1 |= ((zeros >> 7) & 1) << lane;
            a2 |= ((zeros >> 11) & 1) << lane;
            a3 |= ((zeros >> 15) & 1) << lane;
            a4 |= ((zeros >> 19) & 1) << lane;
            a5 |= ((zeros >> 23) & 1) << lane;
            a6 |= ((zeros >> 27) & 1) << lane;
            a7 |= ((zeros >> 31) & 1) << lane;
        }

        scatter_rows(k, out, [a0, a1, a2, a3, a4, a5, a6, a7]);
    }
}

#[cfg(test)]
mod tests {
    use fastlanes::BitPacking;

    use super::*;

    fn pack_u32(values: &[u32; 1024]) -> [u32; 128] {
        let mut out = [0u32; 128];
        unsafe {
            BitPacking::unchecked_pack(4, values, &mut out);
        }
        out
    }

    fn bit(chunk_bits: &[u64; 16], i: usize) -> bool {
        (chunk_bits[i / 64] >> (i % 64)) & 1 != 0
    }

    #[test]
    fn eq_w4_matches_naive() {
        let mut values = [0u32; 1024];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 31 + 7) % 16) as u32;
        }
        let packed = pack_u32(&values);

        for c in 0..16u8 {
            let mut got = [0u64; 16];
            swar_eq_w4_u32(&packed, c, &mut got);
            for i in 0..1024 {
                let expected = values[i] == c as u32;
                assert_eq!(bit(&got, i), expected, "i={i}, c={c}, value={}", values[i]);
            }
        }
    }
}
