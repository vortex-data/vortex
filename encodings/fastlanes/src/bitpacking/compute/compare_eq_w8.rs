// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::many_single_char_names)]

//! Hand-tuned in-range Eq kernel for `bit_width = 8` on `u32` storage.
//!
//! Each packed `u32` holds 4 byte slots = 4 element rows. Byte-aligned, so no Knuth
//! broadword tricks — `_mm256_cmpeq_epi8` against a broadcast `c` does the per-element
//! compare in one SIMD op, and `_mm256_movemask_epi8` packs 32 byte-results into 32 bits.
//! BMI2 `_pext_u32` deinterleaves those 32 bits into 4 row-groups (one per byte position
//! within a packed `u32`).

use fastlanes::FL_ORDER;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "bmi2"
))]
#[inline]
pub(crate) fn swar_eq_w8_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    unsafe { simd_eq_w8_avx2_bmi2(packed_chunk, c, out) }
}

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "bmi2"
)))]
#[inline]
pub(crate) fn swar_eq_w8_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    scalar_eq_w8(packed_chunk, c, out);
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "bmi2"
))]
#[inline]
pub(crate) fn swar_lt_w8_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    unsafe { simd_lt_w8_avx2_bmi2(packed_chunk, c, out) }
}

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "bmi2"
)))]
#[inline]
pub(crate) fn swar_lt_w8_u32(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    scalar_lt_w8(packed_chunk, c, out);
}

// ---------------------------------------------------------------------------
// AVX2 + BMI2 implementation.
// ---------------------------------------------------------------------------

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "bmi2"
))]
#[inline]
unsafe fn simd_eq_w8_avx2_bmi2(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(packed_chunk.len(), 256);

    unsafe {
        let c_vec = _mm256_set1_epi8(c as i8);

        for k in 0..8usize {
            // 4 ymm cover 32 lanes' packed u32 = 32 bytes per ymm = 8 lanes × 4 rows.
            let p = packed_chunk.as_ptr().add(k * 32).cast::<__m256i>();

            let mut row0 = 0u32;
            let mut row1 = 0u32;
            let mut row2 = 0u32;
            let mut row3 = 0u32;

            for half in 0..4 {
                let p_ymm = _mm256_loadu_si256(p.add(half));
                let cmp = _mm256_cmpeq_epi8(p_ymm, c_vec);
                let mask = _mm256_movemask_epi8(cmp) as u32;

                // mask bit (lane_in_half*4 + byte_in_word) = (byte == c).
                // Deinterleave by byte_in_word:
                //   byte 0 (row k*4+0): stride-4 starting at bit 0 = 0x11111111
                //   byte 1 (row k*4+1): stride-4 starting at bit 1 = 0x22222222
                //   byte 2 (row k*4+2): stride-4 starting at bit 2 = 0x44444444
                //   byte 3 (row k*4+3): stride-4 starting at bit 3 = 0x88888888
                let shift = half * 8;
                row0 |= _pext_u32(mask, 0x1111_1111) << shift;
                row1 |= _pext_u32(mask, 0x2222_2222) << shift;
                row2 |= _pext_u32(mask, 0x4444_4444) << shift;
                row3 |= _pext_u32(mask, 0x8888_8888) << shift;
            }

            scatter_w8(k, out, [row0, row1, row2, row3]);
        }
    }
}

#[inline(always)]
fn scatter_w8(k: usize, out: &mut [u64; 16], rows: [u32; 4]) {
    // For W=8 row r = k*4 + s for s ∈ 0..4. elem_base = FL_ORDER[r/8] * 16 + (r%8) * 128.
    //   k=0 (rows 0..3): r/8=0, FL_ORDER[0]=0 → bases 0,128,256,384 → u64 0,2,4,6 / off 0
    //   k=1 (rows 4..7): r/8=0, FL_ORDER[0]=0 → bases 512,640,768,896 → u64 8,10,12,14 / off 0
    //   k=2 (rows 8..11): r/8=1, FL_ORDER[1]=4 → bases 64,192,320,448 → u64 1,3,5,7 / off 0
    //   k=3 (rows 12..15): r/8=1, FL_ORDER[1]=4 → bases 576,704,832,960 → u64 9,11,13,15 / off 0
    //   k=4 (rows 16..19): r/8=2, FL_ORDER[2]=2 → bases 32,160,288,416 → u64 0,2,4,6 / off 32
    //   k=5 (rows 20..23): r/8=2, FL_ORDER[2]=2 → bases 544,672,800,928 → u64 8,10,12,14 / off 32
    //   k=6 (rows 24..27): r/8=3, FL_ORDER[3]=6 → bases 96,224,352,480 → u64 1,3,5,7 / off 32
    //   k=7 (rows 28..31): r/8=3, FL_ORDER[3]=6 → bases 608,736,864,992 → u64 9,11,13,15 / off 32
    let row_base = (k % 2) * 4; // k=0,1 → 0,4 hmm no. Let me derive from FL_ORDER.
    let fl = FL_ORDER[k / 2]; // r/8 = k/2 because each k has 4 rows and rows 0-7 are r/8=0.
    let r_mod8 = (k % 2) * 4; // rows 0,1,2,3 (k=0) or 4,5,6,7 (k=1) all have r%8 in [0..8).
    // u_base = (fl*16) / 64 + (r_mod8 * 128 / 64) for the first row's u64.
    let u_base = (fl * 16) / 64 + (r_mod8 * 2);
    let bit_off = ((fl * 16) % 64) as u64;

    out[u_base] |= (rows[0] as u64) << bit_off;
    out[u_base + 2] |= (rows[1] as u64) << bit_off;
    out[u_base + 4] |= (rows[2] as u64) << bit_off;
    out[u_base + 6] |= (rows[3] as u64) << bit_off;
}

// AVX2 Lt: byte-aligned unsigned less-than via `min_epu8` identity:
//   a < c  iff  (a <= c) AND NOT (a == c)
//   a <= c  iff  min(a, c) == a
//
//   le = cmpeq_epi8(min_epu8(a, c), a)
//   eq = cmpeq_epi8(a, c)
//   lt = le AND NOT eq    = andnot(eq, le)

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "bmi2"
))]
#[inline]
unsafe fn simd_lt_w8_avx2_bmi2(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(packed_chunk.len(), 256);

    unsafe {
        let c_vec = _mm256_set1_epi8(c as i8);

        for k in 0..8usize {
            let p = packed_chunk.as_ptr().add(k * 32).cast::<__m256i>();

            let mut row0 = 0u32;
            let mut row1 = 0u32;
            let mut row2 = 0u32;
            let mut row3 = 0u32;

            for half in 0..4 {
                let p_ymm = _mm256_loadu_si256(p.add(half));
                let min = _mm256_min_epu8(p_ymm, c_vec);
                let le = _mm256_cmpeq_epi8(min, p_ymm);
                let eq = _mm256_cmpeq_epi8(p_ymm, c_vec);
                let lt = _mm256_andnot_si256(eq, le);
                let mask = _mm256_movemask_epi8(lt) as u32;

                let shift = half * 8;
                row0 |= _pext_u32(mask, 0x1111_1111) << shift;
                row1 |= _pext_u32(mask, 0x2222_2222) << shift;
                row2 |= _pext_u32(mask, 0x4444_4444) << shift;
                row3 |= _pext_u32(mask, 0x8888_8888) << shift;
            }

            scatter_w8(k, out, [row0, row1, row2, row3]);
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback.
// ---------------------------------------------------------------------------

#[inline]
fn scalar_eq_w8(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    debug_assert_eq!(packed_chunk.len(), 256);
    let c_packed: u32 = u32::from(c) * 0x0101_0101;

    for k in 0..8usize {
        let mut a0 = 0u32;
        let mut a1 = 0u32;
        let mut a2 = 0u32;
        let mut a3 = 0u32;

        for lane in 0..32usize {
            let word = packed_chunk[k * 32 + lane];
            let xor = word ^ c_packed;
            let zeros = xor.wrapping_sub(0x0101_0101) & !xor & 0x8080_8080;
            a0 |= ((zeros >> 7) & 1) << lane;
            a1 |= ((zeros >> 15) & 1) << lane;
            a2 |= ((zeros >> 23) & 1) << lane;
            a3 |= ((zeros >> 31) & 1) << lane;
        }

        scatter_w8(k, out, [a0, a1, a2, a3]);
    }
}

#[inline]
fn scalar_lt_w8(packed_chunk: &[u32], c: u8, out: &mut [u64; 16]) {
    debug_assert_eq!(packed_chunk.len(), 256);
    const H: u32 = 0x8080_8080;
    const M: u32 = 0x7F7F_7F7F;
    const L: u32 = 0x0101_0101;
    let c_packed = u32::from(c) * L;
    let c_hi = c_packed & H;
    let c_lo = c_packed & M;

    for k in 0..8usize {
        let mut a0 = 0u32;
        let mut a1 = 0u32;
        let mut a2 = 0u32;
        let mut a3 = 0u32;

        for lane in 0..32usize {
            let a = packed_chunk[k * 32 + lane];
            let a_hi = a & H;
            let a_lo = a & M;
            let hi_lt = !a_hi & c_hi;
            let hi_eq = !(a_hi ^ c_hi) & H;
            let lo_le = (c_lo | H).wrapping_sub(a_lo) & H;
            let xor_lo = a_lo ^ c_lo;
            let lo_eq = xor_lo.wrapping_sub(L) & !xor_lo & H;
            let lo_lt = lo_le & !lo_eq;
            let lt = hi_lt | (hi_eq & lo_lt);

            a0 |= ((lt >> 7) & 1) << lane;
            a1 |= ((lt >> 15) & 1) << lane;
            a2 |= ((lt >> 23) & 1) << lane;
            a3 |= ((lt >> 31) & 1) << lane;
        }

        scatter_w8(k, out, [a0, a1, a2, a3]);
    }
}

#[cfg(test)]
mod tests {
    use fastlanes::BitPacking;

    use super::*;

    fn pack_u32(values: &[u32; 1024]) -> [u32; 256] {
        let mut out = [0u32; 256];
        unsafe {
            BitPacking::unchecked_pack(8, values, &mut out);
        }
        out
    }

    fn bit(chunk_bits: &[u64; 16], i: usize) -> bool {
        (chunk_bits[i / 64] >> (i % 64)) & 1 != 0
    }

    #[test]
    fn eq_w8_matches_naive() {
        let mut values = [0u32; 1024];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 31 + 7) % 256) as u32;
        }
        let packed = pack_u32(&values);

        for c in (0..=255u8).step_by(7) {
            let mut got = [0u64; 16];
            swar_eq_w8_u32(&packed, c, &mut got);
            for i in 0..1024 {
                let expected = values[i] == c as u32;
                assert_eq!(bit(&got, i), expected, "i={i}, c={c}, value={}", values[i]);
            }
        }
    }

    #[test]
    fn lt_w8_matches_naive() {
        let mut values = [0u32; 1024];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 17 + 3) % 256) as u32;
        }
        let packed = pack_u32(&values);

        for c in (0..=255u8).step_by(7) {
            let mut got = [0u64; 16];
            swar_lt_w8_u32(&packed, c, &mut got);
            for i in 0..1024 {
                let expected = values[i] < c as u32;
                assert_eq!(bit(&got, i), expected, "i={i}, c={c}, value={}", values[i]);
            }
        }
    }
}
