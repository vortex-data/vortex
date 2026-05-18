// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::many_single_char_names)]

//! Broadword / SWAR fast paths for in-range constant comparison against a `BitPacked`
//! array.
//!
//! These kernels work entirely on the packed buffer — never materialising the unpacked
//! primitive — by XORing the packed chunk against a same-layout packed constant and then
//! applying Knuth's broadword tricks (Hacker's Delight §6-1 / TAOCP 4A §7.1.3) to extract
//! per-element results.
//!
//! Supports any **power-of-two `bit_width` ≤ 16** on `u32` storage. For each such width
//! `W`, every `u32` packed word holds `32 / W` slot results in a uniform pattern, so the
//! standard SWAR masks `L` (one at the low bit of each slot) and `H` (one at the high
//! bit) cover the kernel uniformly:
//!
//! ```text
//! W   slots/word   L              H
//! 1   32           0xFFFFFFFF     0xFFFFFFFF   (special-cased; just !xor)
//! 2   16           0x55555555     0xAAAAAAAA
//! 4    8           0x11111111     0x88888888
//! 8    4           0x01010101     0x80808080
//! 16   2           0x00010001     0x80008000
//! ```
//!
//! `bit_width ∈ {3, 5, 6, 7, 9, ..., 31}` (non-power-of-two) requires straddler handling
//! across the FastLanes word boundary and is not yet implemented — those fall through to
//! the canonical path.
//!
//! # Layout exploit
//!
//! For any supported `W`, the 32 lanes' results for a fixed `(k, slot)` map to 32
//! **consecutive** elements starting at `elem_base = FL_ORDER[row/8] * 16 + (row%8) *
//! 128`, where `row = k * (32 / W) + slot`. That's exactly one 32-bit half of one `u64`
//! of the chunk's element-order bitmap (`elem_base / 64` selects the `u64`, `elem_base
//! % 64` is `0` or `32`). So we collect a full `u32` of per-element bits from the 32
//! lane results and OR it into the right slot of a `[u64; 16]` chunk bitmap in one step.

use fastlanes::FL_ORDER;

const LANES_U32: usize = 32;

/// `(L, H)` mask pair for the SWAR Eq broadword zero-test on `u32` storage at width `W`.
#[inline(always)]
const fn lh_masks(w: usize) -> (u32, u32) {
    match w {
        1 => (0xFFFFFFFF, 0xFFFFFFFF),
        2 => (0x55555555, 0xAAAAAAAA),
        4 => (0x11111111, 0x88888888),
        8 => (0x01010101, 0x80808080),
        16 => (0x00010001, 0x80008000),
        _ => unreachable!(),
    }
}

/// `(H, M)` mask pair for the SWAR Lt high/low split on `u32` storage at width `W`.
/// `H` has 1 at the high bit of each slot; `M` is `!H` restricted to the slot's
/// non-high bits.
#[inline(always)]
const fn hm_masks(w: usize) -> (u32, u32) {
    match w {
        1 => (0xFFFFFFFF, 0x00000000),
        2 => (0xAAAAAAAA, 0x55555555),
        4 => (0x88888888, 0x77777777),
        8 => (0x80808080, 0x7F7F7F7F),
        16 => (0x80008000, 0x7FFF7FFF),
        _ => unreachable!(),
    }
}

/// Returns `true` if `w` is a supported power-of-two bit width on `u32` storage.
#[inline(always)]
pub(crate) const fn is_supported_pow2_w(w: u8) -> bool {
    matches!(w, 1 | 2 | 4 | 8 | 16)
}

/// SWAR Eq: writes per-element result bits directly into the chunk-local `[u64; 16]`
/// bitmap in element order. Supported `W ∈ {1, 2, 4, 8, 16}`.
pub(crate) fn swar_eq_pow2_u32<const W: usize>(
    packed_chunk: &[u32],
    c: u32,
    chunk_bits: &mut [u64; 16],
) {
    debug_assert_eq!(packed_chunk.len(), 32 * W);
    let (l_mask, h_mask) = lh_masks(W);
    let slots_per_word: usize = 32 / W;

    // `c & ((1 << W) - 1)) * L` lays the W low bits of `c` into every slot.
    let c_low: u32 = if W == 32 { c } else { c & ((1u32 << W) - 1) };
    let c_packed: u32 = if W == 1 {
        if c_low != 0 { 0xFFFFFFFFu32 } else { 0 }
    } else {
        c_low.wrapping_mul(l_mask)
    };

    for k in 0..W {
        // Per-slot accumulators (only `slots_per_word` entries used).
        let mut accs = [0u32; 32];
        for lane in 0..LANES_U32 {
            let word = packed_chunk[k * LANES_U32 + lane];
            let xor = word ^ c_packed;
            let zeros: u32 = if W == 1 {
                // For 1-bit slots Knuth's `(d - L) & !d & H` degenerates; bit `s` of
                // `!xor` is already the slot-`s` equality bit.
                !xor
            } else {
                // Knuth broadword zero-slot test: bit `s*W + W-1` of `zeros` is set iff
                // slot `s` of `xor` was zero.
                xor.wrapping_sub(l_mask) & !xor & h_mask
            };

            // Extract slot-`s` result bit into bit `lane` of `accs[s]`.
            let mut s = 0;
            while s < slots_per_word {
                let bit_pos = s * W + (W - 1);
                accs[s] |= ((zeros >> bit_pos) & 1) << lane;
                s += 1;
            }
        }
        // Distribute the per-slot accumulators into `chunk_bits`. Each `accs[s]` covers
        // 32 consecutive element indices starting at `elem_base`.
        let mut s = 0;
        while s < slots_per_word {
            let row = k * slots_per_word + s;
            let elem_base = FL_ORDER[row / 8] * 16 + (row % 8) * 128;
            chunk_bits[elem_base / 64] |= (accs[s] as u64) << (elem_base % 64);
            s += 1;
        }
    }
}

/// SWAR unsigned Lt: writes per-element `< c` result bits into `chunk_bits`.
/// Supported `W ∈ {1, 2, 4, 8, 16}`.
///
/// High/low split per slot:
///
/// ```text
/// a < c  iff  (a_hi < c_hi)  OR  (a_hi == c_hi AND a_lo < c_lo)
///
/// a_hi < c_hi   = !a_hi & c_hi                      (per slot)
/// a_hi == c_hi  = !(a_hi ^ c_hi) & H                (per slot)
/// a_lo <= c_lo  = ((c_lo | H) - a_lo) & H            (Knuth guard-bit subtraction)
/// a_lo == c_lo  = broadword-zero on (a_lo ^ c_lo)    (Knuth zero on lower W-1 bits)
/// a_lo < c_lo   = a_lo <= c_lo AND NOT a_lo == c_lo
/// ```
///
/// For `W = 1` the slot has no lower bits, so the formula collapses to `lt = !a & c`.
pub(crate) fn swar_lt_pow2_u32<const W: usize>(
    packed_chunk: &[u32],
    c: u32,
    chunk_bits: &mut [u64; 16],
) {
    debug_assert_eq!(packed_chunk.len(), 32 * W);
    let (h_mask, m_mask) = hm_masks(W);
    let (l_mask, _) = lh_masks(W);
    let slots_per_word: usize = 32 / W;

    let c_low: u32 = if W == 32 { c } else { c & ((1u32 << W) - 1) };
    let c_packed: u32 = if W == 1 {
        if c_low != 0 { 0xFFFFFFFFu32 } else { 0 }
    } else {
        c_low.wrapping_mul(l_mask)
    };
    let c_high = c_packed & h_mask;
    let c_low_bits = c_packed & m_mask;

    for k in 0..W {
        let mut accs = [0u32; 32];
        for lane in 0..LANES_U32 {
            let a = packed_chunk[k * LANES_U32 + lane];
            let lt: u32 = if W == 1 {
                !a & c_packed
            } else {
                let a_hi = a & h_mask;
                let a_lo = a & m_mask;
                let hi_lt = !a_hi & c_high;
                let hi_eq = !(a_hi ^ c_high) & h_mask;
                let lo_le = (c_low_bits | h_mask).wrapping_sub(a_lo) & h_mask;
                let xor_lo = a_lo ^ c_low_bits;
                let lo_eq = xor_lo.wrapping_sub(l_mask) & !xor_lo & h_mask;
                let lo_lt = lo_le & !lo_eq;
                hi_lt | (hi_eq & lo_lt)
            };
            let mut s = 0;
            while s < slots_per_word {
                let bit_pos = s * W + (W - 1);
                accs[s] |= ((lt >> bit_pos) & 1) << lane;
                s += 1;
            }
        }
        let mut s = 0;
        while s < slots_per_word {
            let row = k * slots_per_word + s;
            let elem_base = FL_ORDER[row / 8] * 16 + (row % 8) * 128;
            chunk_bits[elem_base / 64] |= (accs[s] as u64) << (elem_base % 64);
            s += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use fastlanes::BitPacking;

    use super::*;

    fn pack_u32(values: &[u32; 1024], bit_width: usize) -> Vec<u32> {
        let mut out = vec![0u32; 32 * bit_width];
        unsafe {
            BitPacking::unchecked_pack(bit_width, values, &mut out);
        }
        out
    }

    fn bit(chunk_bits: &[u64; 16], i: usize) -> bool {
        (chunk_bits[i / 64] >> (i % 64)) & 1 != 0
    }

    macro_rules! eq_test {
        ($name:ident, $W:expr) => {
            #[test]
            fn $name() {
                let mask = if $W == 32 { u32::MAX } else { (1u32 << $W) - 1 };
                let mut values = [0u32; 1024];
                for (i, v) in values.iter_mut().enumerate() {
                    *v = ((i as u32).wrapping_mul(31).wrapping_add(7)) & mask;
                }
                let packed = pack_u32(&values, $W);

                let test_constants: &[u32] = &[
                    0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 127, 200, 255, 1024, 32768, 65535,
                ];
                for &c in test_constants {
                    if c > mask {
                        continue;
                    }
                    let mut got = [0u64; 16];
                    swar_eq_pow2_u32::<$W>(&packed, c, &mut got);
                    for i in 0..1024 {
                        let expected = values[i] == c;
                        assert_eq!(
                            bit(&got, i),
                            expected,
                            "W={}, i={}, c={}, value={}",
                            $W,
                            i,
                            c,
                            values[i]
                        );
                    }
                }
            }
        };
    }

    macro_rules! lt_test {
        ($name:ident, $W:expr) => {
            #[test]
            fn $name() {
                let mask = if $W == 32 { u32::MAX } else { (1u32 << $W) - 1 };
                let mut values = [0u32; 1024];
                for (i, v) in values.iter_mut().enumerate() {
                    *v = ((i as u32).wrapping_mul(17).wrapping_add(3)) & mask;
                }
                let packed = pack_u32(&values, $W);

                let test_constants: &[u32] = &[
                    0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 127, 200, 255, 1024, 32768, 65535,
                ];
                for &c in test_constants {
                    if c > mask {
                        continue;
                    }
                    let mut got = [0u64; 16];
                    swar_lt_pow2_u32::<$W>(&packed, c, &mut got);
                    for i in 0..1024 {
                        let expected = values[i] < c;
                        assert_eq!(
                            bit(&got, i),
                            expected,
                            "W={}, i={}, c={}, value={}",
                            $W,
                            i,
                            c,
                            values[i]
                        );
                    }
                }
            }
        };
    }

    eq_test!(eq_w1, 1);
    eq_test!(eq_w2, 2);
    eq_test!(eq_w4, 4);
    eq_test!(eq_w8, 8);
    eq_test!(eq_w16, 16);

    lt_test!(lt_w1, 1);
    lt_test!(lt_w2, 2);
    lt_test!(lt_w4, 4);
    lt_test!(lt_w8, 8);
    lt_test!(lt_w16, 16);
}
