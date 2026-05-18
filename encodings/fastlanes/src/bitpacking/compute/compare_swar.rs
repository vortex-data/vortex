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

/// Returns `true` if `w` is a non-power-of-two width handled by the generic
/// rotation-table kernel.
#[inline(always)]
pub(crate) const fn is_supported_nonpow2_w(w: u8) -> bool {
    matches!(w, 3 | 5 | 6 | 7 | 9 | 10 | 11 | 12 | 13 | 14 | 15)
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

// ------------------------------------------------------------------------------------
//
// Non-power-of-two widths: Knuth broadword with rotation tables.
//
// For W ∤ 32, some W-bit elements straddle the 32-bit word boundary inside a lane.
// The kernel processes each output word k ∈ 0..W:
//
//  * In-word slots — elements whose W bits are fully inside word k. Their starts within
//    the word repeat with period `W / gcd(W, 32)`, so the per-word `L` (low-bit-per-slot)
//    and `H` (high-bit-per-slot) masks rotate through that period. Per word, one Knuth
//    broadword zero-test `(xor - L) & !xor & H` gives all per-element results at once.
//
//  * Straddler — at most one element starts in word k but ends in word k+1. Its W bits
//    are extracted with a scalar two-word stitch `(packed[k] >> shift) | (packed[k+1] <<
//    (32-shift))`, masked to W bits, then compared scalar to `c`.
//
// `c_packed[k]` is the bit pattern of word k for a constant input — bit `i` of
// `c_packed[k]` equals bit `(k*32 + i) mod W` of `c`. We compute it once per (c, w, k).
//
// Output layout exploit (same as the power-of-two kernel): all 32 lanes' result for a
// fixed element row `r` map to 32 consecutive element indices starting at
// `FL_ORDER[r/8]*16 + (r%8)*128`, which is one 32-bit half of one `u64` of the chunk
// bitmap.

/// Per-word layout for the generic kernel. Up to 11 in-word slots for W=3 (smallest
/// non-pow2 W); we size to 16 to comfortably cover all supported widths.
const MAX_SLOTS_PER_WORD: usize = 16;

#[derive(Clone, Copy)]
struct WordLayout {
    /// Number of in-word slots in this word.
    n_in_word: u8,
    /// `(bit_start_in_word, element_row)` for each in-word slot.
    in_word: [(u8, u8); MAX_SLOTS_PER_WORD],
    /// Knuth broadword L mask: 1 at the low bit of each in-word slot.
    l_mask: u32,
    /// Knuth broadword H mask: 1 at the high bit of each in-word slot.
    h_mask: u32,
    /// Bit pattern of constant `c` packed into this word position.
    c_packed: u32,
    /// Optional straddler whose low bits live at `(low_shift, ..)` of this word and
    /// whose high bits continue into word `k + 1`. `element_row` is the straddler's row.
    straddler: Option<(u8, u8)>,
}

#[inline]
fn build_layouts(c: u32, w: usize) -> [WordLayout; 31] {
    let mut layouts = [WordLayout {
        n_in_word: 0,
        in_word: [(0, 0); MAX_SLOTS_PER_WORD],
        l_mask: 0,
        h_mask: 0,
        c_packed: 0,
        straddler: None,
    }; 31];
    let w_mask: u32 = if w == 32 { u32::MAX } else { (1u32 << w) - 1 };
    let c_low = c & w_mask;

    for k in 0..w {
        let mut layout = WordLayout {
            n_in_word: 0,
            in_word: [(0, 0); MAX_SLOTS_PER_WORD],
            l_mask: 0,
            h_mask: 0,
            c_packed: {
                let mut p = 0u32;
                for i in 0..32usize {
                    let bit_in_c = (k * 32 + i) % w;
                    let bit = (c_low >> bit_in_c) & 1;
                    p |= bit << i;
                }
                p
            },
            straddler: None,
        };

        // First element starting at or after stream bit `k * 32`.
        let r0 = (k * 32).div_ceil(w);
        let mut r = r0;
        while r < 32 {
            let bit_start = r * w;
            // bit_start is in word k iff k*32 <= bit_start < (k+1)*32.
            if bit_start >= (k + 1) * 32 {
                break;
            }
            let bit_start_in_word = bit_start - k * 32;
            if bit_start_in_word + w <= 32 {
                let n = layout.n_in_word as usize;
                layout.in_word[n] = (bit_start_in_word as u8, r as u8);
                layout.n_in_word += 1;
                layout.l_mask |= 1u32 << bit_start_in_word;
                layout.h_mask |= 1u32 << (bit_start_in_word + w - 1);
                r += 1;
            } else {
                // Straddler: low part in word k, high part in word k+1.
                layout.straddler = Some((bit_start_in_word as u8, r as u8));
                break;
            }
        }

        layouts[k] = layout;
    }

    layouts
}

/// Per-W precomputed Knuth-broadword tables. Built once per `(c, W)` and reused across
/// all chunks of the array.
pub(crate) struct GenericLayout {
    /// Bit width (W).
    w: usize,
    /// Low-W-bits mask `(1 << W) - 1`.
    w_mask: u32,
    /// Low W bits of the constant.
    c_low: u32,
    /// `(L, H)` Knuth masks per word, padded to 16 entries.
    layouts: [WordLayout; 16],
    /// For each output word index `k ∈ 0..W`, the Lt-specific `(M, full_slot_mask,
    /// c_high, c_low_bits)` precomputed.
    lt_tables: [LtTables; 16],
}

#[derive(Clone, Copy, Default)]
struct LtTables {
    m_mask: u32,
    full_slot_mask: u32,
    c_high: u32,
    c_low_bits: u32,
}

pub(crate) fn build_generic_layout(c: u32, w: usize) -> GenericLayout {
    debug_assert!((3..=15).contains(&w));
    let w_mask: u32 = (1u32 << w) - 1;
    let c_low = c & w_mask;
    let layouts = {
        let out = build_layouts(c, w);
        // We sized `build_layouts`'s output to 31 entries to handle widths up to 30; pack
        // down to 16 (we only support `W ≤ 16`).
        let mut packed = [WordLayout {
            n_in_word: 0,
            in_word: [(0, 0); MAX_SLOTS_PER_WORD],
            l_mask: 0,
            h_mask: 0,
            c_packed: 0,
            straddler: None,
        }; 16];
        packed[..w].copy_from_slice(&out[..w]);
        packed
    };

    let mut lt_tables = [LtTables::default(); 16];
    for k in 0..w {
        let layout = &layouts[k];
        let n_in_word = layout.n_in_word as usize;
        let mut m_mask: u32 = 0;
        let mut full_slot_mask: u32 = 0;
        for i in 0..n_in_word {
            let s = u32::from(layout.in_word[i].0);
            m_mask |= ((1u32 << (w - 1)) - 1) << s;
            full_slot_mask |= w_mask << s;
        }
        let c_packed_masked = layout.c_packed & full_slot_mask;
        lt_tables[k] = LtTables {
            m_mask,
            full_slot_mask,
            c_high: c_packed_masked & layout.h_mask,
            c_low_bits: c_packed_masked & m_mask,
        };
    }

    GenericLayout {
        w,
        w_mask,
        c_low,
        layouts,
        lt_tables,
    }
}

/// Generic SWAR Eq for `bit_width ∈ {3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15}` on `u32`
/// storage. Layouts must be pre-built via [`build_generic_layout`].
pub(crate) fn swar_eq_generic_u32(
    packed_chunk: &[u32],
    layout_data: &GenericLayout,
    chunk_bits: &mut [u64; 16],
) {
    let w = layout_data.w;
    debug_assert_eq!(packed_chunk.len(), 32 * w);
    let w_mask = layout_data.w_mask;
    let c_low = layout_data.c_low;
    let mut accs = [0u32; 32];

    for k in 0..w {
        let layout = &layout_data.layouts[k];
        let n_in_word = layout.n_in_word as usize;

        for lane in 0..LANES_U32 {
            let a = packed_chunk[k * LANES_U32 + lane];
            let xor = a ^ layout.c_packed;
            // Knuth broadword zero-test over in-word slots.
            let zeros = xor.wrapping_sub(layout.l_mask) & !xor & layout.h_mask;

            for i in 0..n_in_word {
                let (slot_start, r) = layout.in_word[i];
                let high_bit_pos = u32::from(slot_start) + (w as u32) - 1;
                accs[r as usize] |= ((zeros >> high_bit_pos) & 1) << lane;
            }

            if let Some((low_shift, r)) = layout.straddler {
                let low_shift = u32::from(low_shift);
                let low_bits = 32 - low_shift;
                let lo = a >> low_shift;
                let hi = packed_chunk[(k + 1) * LANES_U32 + lane] << low_bits;
                let extracted = (lo | hi) & w_mask;
                accs[r as usize] |= u32::from(extracted == c_low) << lane;
            }
        }
    }

    for r in 0..32usize {
        let elem_base = FL_ORDER[r / 8] * 16 + (r % 8) * 128;
        chunk_bits[elem_base / 64] |= (accs[r] as u64) << (elem_base % 64);
    }
}

/// Generic SWAR Lt for non-power-of-two widths on `u32` storage. Layouts must be
/// pre-built via [`build_generic_layout`].
pub(crate) fn swar_lt_generic_u32(
    packed_chunk: &[u32],
    layout_data: &GenericLayout,
    chunk_bits: &mut [u64; 16],
) {
    let w = layout_data.w;
    debug_assert_eq!(packed_chunk.len(), 32 * w);
    let w_mask = layout_data.w_mask;
    let c_low = layout_data.c_low;
    let mut accs = [0u32; 32];

    for k in 0..w {
        let layout = &layout_data.layouts[k];
        let lt_t = &layout_data.lt_tables[k];
        let n_in_word = layout.n_in_word as usize;
        let h_mask = layout.h_mask;

        for lane in 0..LANES_U32 {
            let a_word = packed_chunk[k * LANES_U32 + lane];
            let a = a_word & lt_t.full_slot_mask;
            let a_hi = a & h_mask;
            let a_lo = a & lt_t.m_mask;

            let hi_lt = !a_hi & lt_t.c_high;
            let hi_eq = !(a_hi ^ lt_t.c_high) & h_mask;
            let lo_le = (lt_t.c_low_bits | h_mask).wrapping_sub(a_lo) & h_mask;
            let xor_lo = a_lo ^ lt_t.c_low_bits;
            let lo_eq = xor_lo.wrapping_sub(layout.l_mask) & !xor_lo & h_mask;
            let lo_lt = lo_le & !lo_eq;
            let lt = hi_lt | (hi_eq & lo_lt);

            for i in 0..n_in_word {
                let (slot_start, r) = layout.in_word[i];
                let high_bit_pos = u32::from(slot_start) + (w as u32) - 1;
                accs[r as usize] |= ((lt >> high_bit_pos) & 1) << lane;
            }

            if let Some((low_shift, r)) = layout.straddler {
                let low_shift = u32::from(low_shift);
                let low_bits = 32 - low_shift;
                let lo = a_word >> low_shift;
                let hi = packed_chunk[(k + 1) * LANES_U32 + lane] << low_bits;
                let extracted = (lo | hi) & w_mask;
                accs[r as usize] |= u32::from(extracted < c_low) << lane;
            }
        }
    }

    for r in 0..32usize {
        let elem_base = FL_ORDER[r / 8] * 16 + (r % 8) * 128;
        chunk_bits[elem_base / 64] |= (accs[r] as u64) << (elem_base % 64);
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

    // Generic non-power-of-two kernel tests.
    macro_rules! generic_eq_test {
        ($name:ident, $W:expr) => {
            #[test]
            fn $name() {
                let mask = (1u32 << $W) - 1;
                let mut values = [0u32; 1024];
                for (i, v) in values.iter_mut().enumerate() {
                    *v = ((i as u32).wrapping_mul(31).wrapping_add(7)) & mask;
                }
                let packed = pack_u32(&values, $W);

                let test_constants: &[u32] = &[
                    0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 127, 200, 255, 1024, 8192, 32768,
                ];
                for &c in test_constants {
                    if c > mask {
                        continue;
                    }
                    let layout = build_generic_layout(c, $W);
                    let mut got = [0u64; 16];
                    swar_eq_generic_u32(&packed, &layout, &mut got);
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

    macro_rules! generic_lt_test {
        ($name:ident, $W:expr) => {
            #[test]
            fn $name() {
                let mask = (1u32 << $W) - 1;
                let mut values = [0u32; 1024];
                for (i, v) in values.iter_mut().enumerate() {
                    *v = ((i as u32).wrapping_mul(17).wrapping_add(3)) & mask;
                }
                let packed = pack_u32(&values, $W);

                let test_constants: &[u32] = &[
                    0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 127, 200, 255, 1024, 8192, 32768,
                ];
                for &c in test_constants {
                    if c > mask {
                        continue;
                    }
                    let layout = build_generic_layout(c, $W);
                    let mut got = [0u64; 16];
                    swar_lt_generic_u32(&packed, &layout, &mut got);
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

    generic_eq_test!(eq_w3_generic, 3);
    generic_eq_test!(eq_w5_generic, 5);
    generic_eq_test!(eq_w6_generic, 6);
    generic_eq_test!(eq_w7_generic, 7);
    generic_eq_test!(eq_w9_generic, 9);
    generic_eq_test!(eq_w11_generic, 11);
    generic_eq_test!(eq_w13_generic, 13);
    generic_eq_test!(eq_w15_generic, 15);

    generic_lt_test!(lt_w3_generic, 3);
    generic_lt_test!(lt_w5_generic, 5);
    generic_lt_test!(lt_w6_generic, 6);
    generic_lt_test!(lt_w7_generic, 7);
    generic_lt_test!(lt_w9_generic, 9);
    generic_lt_test!(lt_w11_generic, 11);
    generic_lt_test!(lt_w13_generic, 13);
    generic_lt_test!(lt_w15_generic, 15);
}
