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

use fastlanes::FL_ORDER;

const LANES_U32: usize = 32;
const W8_PACKED_LEN_U32: usize = 256; // = 128 * W / size_of::<u32>() for W = 8
const LANE_ROWS: usize = 32; // = 8 * size_of::<u32>()

/// FastLanes (row, lane) → element index inside a 1024-chunk. Matches the formula used by
/// `fastlanes::pack!` / `unpack!`.
#[inline(always)]
fn fl_idx(row: usize, lane: usize) -> usize {
    FL_ORDER[row / 8] * 16 + (row % 8) * 128 + lane
}

/// SWAR Eq for `bit_width = 8` on `u32` storage.
///
/// Algorithm: XOR each `u32` word against the byte-replicated constant, then apply
/// Knuth's broadword zero-byte test (`(x - 0x01010101) & !x & 0x80808080` — bit 7 of
/// byte `b` is set iff byte `b` was zero). Each lane's eight output words contain 32
/// element bytes in row order; we scatter the per-element result bits into the chunk's
/// element-order slot via `fl_idx`.
///
/// Writes a `0u8` / `1u8` per element of the 1024-chunk into `out`.
pub(crate) fn swar_eq_w8_u32(packed_chunk: &[u32], c: u8, out: &mut [u8; 1024]) {
    debug_assert_eq!(packed_chunk.len(), W8_PACKED_LEN_U32);
    let c_packed = (c as u32) * 0x01010101u32;

    for lane in 0..LANES_U32 {
        for k in 0..(LANE_ROWS / 4) {
            let word = packed_chunk[k * LANES_U32 + lane];
            let xor = word ^ c_packed;
            // Knuth broadword zero-byte test.
            let zeros = xor.wrapping_sub(0x01010101) & !xor & 0x80808080;

            // Each byte of `zeros` carries the result bit at position 7.
            for byte_idx in 0..4 {
                let row = k * 4 + byte_idx;
                out[fl_idx(row, lane)] = ((zeros >> (byte_idx * 8 + 7)) & 1) as u8;
            }
        }
    }
}

/// SWAR unsigned Lt for `bit_width = 8` on `u32` storage.
///
/// `out[i] = 1` iff packed element `i` `<` `c` as unsigned.
///
/// We use the standard high-bit / low-bit split:
///
/// ```text
/// a < c  iff  (a_hi < c_hi)
///         OR  (a_hi == c_hi AND a_lo < c_lo)
/// ```
///
/// `a_hi < c_hi` per byte is `!a_hi & c_hi`. `a_hi == c_hi` per byte is `!(a_hi ^ c_hi)`.
/// `a_lo <= c_lo` per byte uses the Knuth "subtract with guard bit" trick: bit 7 of
/// `(c_lo | H) - a_lo` is `1` iff `a_lo <= c_lo` (no borrow into the guard bit). `a_lo
/// == c_lo` is byte-zero-detect on `a_lo ^ c_lo`, so `a_lo < c_lo = a_lo <= c_lo AND
/// NOT a_lo == c_lo`.
pub(crate) fn swar_lt_w8_u32(packed_chunk: &[u32], c: u8, out: &mut [u8; 1024]) {
    debug_assert_eq!(packed_chunk.len(), W8_PACKED_LEN_U32);
    const H: u32 = 0x80808080;
    const M: u32 = 0x7F7F7F7F;
    let c_packed = (c as u32) * 0x01010101u32;
    let c_hi = c_packed & H;
    let c_lo = c_packed & M;

    for lane in 0..LANES_U32 {
        for k in 0..(LANE_ROWS / 4) {
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

            for byte_idx in 0..4 {
                let row = k * 4 + byte_idx;
                out[fl_idx(row, lane)] = ((lt >> (byte_idx * 8 + 7)) & 1) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastlanes::BitPacking;

    fn pack_u32(values: &[u32; 1024]) -> [u32; 256] {
        let mut out = [0u32; 256];
        unsafe {
            BitPacking::unchecked_pack(8, values, &mut out);
        }
        out
    }

    #[test]
    fn eq_w8_matches_naive() {
        // Exercise a few constants against a random-ish bit-packed chunk.
        let mut values = [0u32; 1024];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 31 + 7) % 256) as u32;
        }
        let packed = pack_u32(&values);

        for &c in &[0u8, 1, 7, 42, 127, 128, 200, 255] {
            let mut got = [0u8; 1024];
            swar_eq_w8_u32(&packed, c, &mut got);
            for i in 0..1024 {
                let expected = if values[i] == c as u32 { 1 } else { 0 };
                assert_eq!(
                    got[i], expected,
                    "eq mismatch at i={i}, c={c}, value={}, got={}",
                    values[i], got[i]
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
            let mut got = [0u8; 1024];
            swar_lt_w8_u32(&packed, c, &mut got);
            for i in 0..1024 {
                let expected = if values[i] < c as u32 { 1 } else { 0 };
                assert_eq!(
                    got[i], expected,
                    "lt mismatch at i={i}, c={c}, value={}, got={}",
                    values[i], got[i]
                );
            }
        }
    }
}
