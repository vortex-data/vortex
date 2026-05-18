// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Broadword / SWAR fast paths for in-range constant comparison against a `BitPacked`
//! array.
//!
//! These kernels work entirely on the packed buffer — never materializing the unpacked
//! primitive — by XORing the packed chunk against a same-layout packed constant and then
//! applying Knuth's broadword tricks (Hacker's Delight §6-1 / TAOCP 4A §7.1.3) to extract
//! per-element results.
//!
//! Currently scoped to `bit_width == 8` on `u32` storage (i.e. `i32`/`u32` columns whose
//! values fit in 8 bits), which covers byte-sized integer columns and is the cleanest
//! SWAR case: each `u32` output word holds exactly four packed elements as byte slots,
//! so the SWAR masks are uniform `0x80808080` / `0x01010101`.
//!
//! # Layout exploit
//!
//! The 32 lanes' results for a fixed `(k, byte_idx)` map to 32 **consecutive** elements
//! starting at `elem_base = FL_ORDER[row/8] * 16 + (row%8) * 128`. That's exactly half of
//! one `u64` of the chunk's element-order bitmap (`elem_base / 64` selects the `u64`,
//! `elem_base % 64` is either `0` or `32` selects which half). So we can collect a full
//! `u32` of per-element bits from the 32 lane results and `OR` it into the right slot of
//! a `[u64; 16]` chunk bitmap in one step — no scattered byte writes, no intermediate
//! `Vec<bool>`, no `collect_bool` pass.

use fastlanes::FL_ORDER;

const LANES_U32: usize = 32;
const W8_PACKED_LEN_U32: usize = 256; // = 128 * W / size_of::<u32>() for W = 8
const W: usize = 8;

/// Pre-computed `(u64_idx, bit_offset)` for each `(k, byte_idx)` pair.
/// Index `k * 4 + byte_idx`.
const TARGETS: [(usize, usize); W * 4] = {
    let mut out = [(0, 0); W * 4];
    let mut i = 0;
    while i < W * 4 {
        let k = i / 4;
        let byte_idx = i % 4;
        let row = k * 4 + byte_idx;
        let elem_base = FL_ORDER[row / 8] * 16 + (row % 8) * 128;
        out[i] = (elem_base / 64, elem_base % 64);
        i += 1;
    }
    out
};

/// SWAR Eq for `bit_width = 8` on `u32` storage, writing per-element result bits
/// directly into the chunk-local `[u64; 16]` bitmap in element order.
///
/// Algorithm: XOR each `u32` packed word against the byte-replicated constant, apply
/// Knuth's broadword zero-byte test `(x - 0x01010101) & !x & 0x80808080`, then for each
/// of the four byte positions extract bit 7 across all 32 lanes into a single `u32`
/// and `OR` it into the destination half of the appropriate `u64`.
pub(crate) fn swar_eq_w8_u32(packed_chunk: &[u32], c: u8, chunk_bits: &mut [u64; 16]) {
    debug_assert_eq!(packed_chunk.len(), W8_PACKED_LEN_U32);
    let c_packed = (c as u32) * 0x01010101u32;

    for k in 0..W {
        // Per-byte_idx accumulators: bit `lane` set iff lane's row-`k*4+byte_idx` byte
        // matched `c`. Four independent accumulators break the dependency chain so the
        // backend can pipeline the lane loop.
        let mut acc0: u32 = 0;
        let mut acc1: u32 = 0;
        let mut acc2: u32 = 0;
        let mut acc3: u32 = 0;

        for lane in 0..LANES_U32 {
            let word = packed_chunk[k * LANES_U32 + lane];
            let xor = word ^ c_packed;
            let zeros = xor.wrapping_sub(0x01010101) & !xor & 0x80808080;

            acc0 |= ((zeros >> 7) & 1) << lane;
            acc1 |= ((zeros >> 15) & 1) << lane;
            acc2 |= ((zeros >> 23) & 1) << lane;
            acc3 |= ((zeros >> 31) & 1) << lane;
        }

        let base = k * 4;
        let (u0, b0) = TARGETS[base];
        let (u1, b1) = TARGETS[base + 1];
        let (u2, b2) = TARGETS[base + 2];
        let (u3, b3) = TARGETS[base + 3];
        chunk_bits[u0] |= (acc0 as u64) << b0;
        chunk_bits[u1] |= (acc1 as u64) << b1;
        chunk_bits[u2] |= (acc2 as u64) << b2;
        chunk_bits[u3] |= (acc3 as u64) << b3;
    }
}

/// SWAR unsigned Lt for `bit_width = 8` on `u32` storage. Same output convention as
/// `swar_eq_w8_u32` — writes per-element result bits into `chunk_bits`.
///
/// `out[i] = 1` iff packed element `i` `<` `c` as unsigned. Uses the high-bit / low-bit
/// split:
///
/// ```text
/// a < c  iff  (a_hi < c_hi)
///         OR  (a_hi == c_hi AND a_lo < c_lo)
/// ```
///
/// `a_hi < c_hi` per byte is `!a_hi & c_hi`. `a_hi == c_hi` is `!(a_hi ^ c_hi)`. The
/// `a_lo <= c_lo` test uses Knuth's guard-bit subtraction: bit 7 of
/// `(c_lo | H) - a_lo` is `1` iff the subtraction did not borrow into the guard bit,
/// i.e. `a_lo <= c_lo`. `a_lo == c_lo` is the broadword zero test on `a_lo ^ c_lo`. So
/// `a_lo < c_lo = (a_lo <= c_lo) AND NOT (a_lo == c_lo)`.
pub(crate) fn swar_lt_w8_u32(packed_chunk: &[u32], c: u8, chunk_bits: &mut [u64; 16]) {
    debug_assert_eq!(packed_chunk.len(), W8_PACKED_LEN_U32);
    const H: u32 = 0x80808080;
    const M: u32 = 0x7F7F7F7F;
    let c_packed = (c as u32) * 0x01010101u32;
    let c_hi = c_packed & H;
    let c_lo = c_packed & M;

    for k in 0..W {
        let mut acc0: u32 = 0;
        let mut acc1: u32 = 0;
        let mut acc2: u32 = 0;
        let mut acc3: u32 = 0;

        for lane in 0..LANES_U32 {
            let a = packed_chunk[k * LANES_U32 + lane];
            let a_hi = a & H;
            let a_lo = a & M;

            let hi_lt = !a_hi & c_hi;
            let hi_eq = !(a_hi ^ c_hi) & H;
            let lo_le = (c_lo | H).wrapping_sub(a_lo) & H;
            let xor_lo = a_lo ^ c_lo;
            let lo_eq = xor_lo.wrapping_sub(0x01010101) & !xor_lo & H;
            let lo_lt = lo_le & !lo_eq;
            let lt = hi_lt | (hi_eq & lo_lt);

            acc0 |= ((lt >> 7) & 1) << lane;
            acc1 |= ((lt >> 15) & 1) << lane;
            acc2 |= ((lt >> 23) & 1) << lane;
            acc3 |= ((lt >> 31) & 1) << lane;
        }

        let base = k * 4;
        let (u0, b0) = TARGETS[base];
        let (u1, b1) = TARGETS[base + 1];
        let (u2, b2) = TARGETS[base + 2];
        let (u3, b3) = TARGETS[base + 3];
        chunk_bits[u0] |= (acc0 as u64) << b0;
        chunk_bits[u1] |= (acc1 as u64) << b1;
        chunk_bits[u2] |= (acc2 as u64) << b2;
        chunk_bits[u3] |= (acc3 as u64) << b3;
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

    fn bit_in_chunk_bits(chunk_bits: &[u64; 16], i: usize) -> bool {
        (chunk_bits[i / 64] >> (i % 64)) & 1 != 0
    }

    #[test]
    fn eq_w8_matches_naive() {
        let mut values = [0u32; 1024];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 31 + 7) % 256) as u32;
        }
        let packed = pack_u32(&values);

        for &c in &[0u8, 1, 7, 42, 127, 128, 200, 255] {
            let mut got = [0u64; 16];
            swar_eq_w8_u32(&packed, c, &mut got);
            for i in 0..1024 {
                let expected = values[i] == c as u32;
                assert_eq!(
                    bit_in_chunk_bits(&got, i),
                    expected,
                    "eq mismatch at i={i}, c={c}, value={}",
                    values[i]
                );
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

        for &c in &[0u8, 1, 7, 42, 127, 128, 200, 255] {
            let mut got = [0u64; 16];
            swar_lt_w8_u32(&packed, c, &mut got);
            for i in 0..1024 {
                let expected = values[i] < c as u32;
                assert_eq!(
                    bit_in_chunk_bits(&got, i),
                    expected,
                    "lt mismatch at i={i}, c={c}, value={}",
                    values[i]
                );
            }
        }
    }
}
