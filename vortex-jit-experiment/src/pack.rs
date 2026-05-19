// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Simple linear LSB-first bit-packing for `u32`.
//!
//! Not FastLanes — this is the simplest possible layout, chosen so the JIT,
//! the hand-fused, and the composed paths can all read from the *same* bytes.
//! That way the only thing we're measuring is the cost of the boundary.
//!
//! Layout for bit-width `B`:
//! - Values are packed LSB-first into a stream of `u32` words.
//! - Value `i` occupies bits `[i*B, (i+1)*B)` of the stream.
//! - A value may straddle a word boundary (we read 1 or 2 words to recover it).

use crate::CHUNK_SIZE;

/// Pack `CHUNK_SIZE` values, each fitting in `bit_width` bits, into a linear
/// LSB-first stream of `u32` words.
///
/// Panics in debug if any value has bits set above `bit_width`.
pub fn pack_chunk(values: &[u32; CHUNK_SIZE], bit_width: u32) -> Vec<u32> {
    assert!((1..=31).contains(&bit_width), "bit_width must be 1..=31");
    let total_bits = CHUNK_SIZE * bit_width as usize;
    let n_words = total_bits.div_ceil(32);
    let mut out = vec![0u32; n_words];

    for (i, &v) in values.iter().enumerate() {
        debug_assert!(
            v < (1u32 << bit_width) || bit_width == 32,
            "value {} overflows bit_width {}",
            v,
            bit_width,
        );
        let bit_off = i * bit_width as usize;
        let word_off = bit_off / 32;
        let shift = (bit_off % 32) as u32;
        out[word_off] |= v << shift;
        if shift + bit_width > 32 {
            out[word_off + 1] |= v >> (32 - shift);
        }
    }
    out
}

/// Unpack a single value at `i` from the linear stream. Useful for
/// correctness checks and as the reference implementation that the JIT
/// must match.
#[inline]
pub fn unpack_one(packed: &[u32], i: usize, bit_width: u32) -> u32 {
    let bit_off = i * bit_width as usize;
    let word_off = bit_off / 32;
    let shift = (bit_off % 32) as u32;
    let mask = if bit_width == 32 {
        u32::MAX
    } else {
        (1u32 << bit_width) - 1
    };
    let lo = packed[word_off] >> shift;
    let hi = if shift + bit_width > 32 {
        packed[word_off + 1] << (32 - shift)
    } else {
        0
    };
    (lo | hi) & mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_various_widths() {
        for bit_width in [1u32, 3, 7, 8, 11, 16, 17, 24, 31] {
            let max = if bit_width == 32 {
                u32::MAX
            } else {
                (1u32 << bit_width) - 1
            };
            let mut values = [0u32; CHUNK_SIZE];
            for (i, v) in values.iter_mut().enumerate() {
                *v = (i as u32).wrapping_mul(0x9E37_79B1) & max;
            }
            let packed = pack_chunk(&values, bit_width);
            for (i, &expected) in values.iter().enumerate() {
                assert_eq!(
                    unpack_one(&packed, i, bit_width),
                    expected,
                    "mismatch at i={i} bw={bit_width}",
                );
            }
        }
    }
}
