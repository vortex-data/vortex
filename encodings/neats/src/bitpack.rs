// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    dead_code,
    reason = "ZigZag + bit-packing are intentional bit manipulations; `write_bits`/`read_bits`/\
              `cumulative_offsets` are kept for the planned flat-buffer per-piece bit-pack \
              follow-up (today the encoder uses ChunkedArray<ZigZag<BitPacked>> instead)"
)]

//! Per-piece bit-packing for NeaTS residuals.
//!
//! This module implements the residual encoding described in the NeaTS paper: each piece's
//! residuals are ZigZag-encoded (mapping signed → unsigned without losing the sign bit) and then
//! bit-packed with a per-piece bit-width tight to that piece's max-abs residual.
//!
//! The packed representation lives in a flat byte buffer; per-piece widths and the cumulative
//! bit offsets are stored as separate small slots. Random access to value `i` of piece `p` is
//! `unpack_from(data, base_offset[p] + (i - piece_start[p]) * width[p], width[p])` —
//! a single bit-shift unpack with no decode of neighbouring pieces.

/// Map a signed `i64` to an unsigned `u64` via ZigZag so small absolute values map to small
/// unsigned values. `0 → 0`, `-1 → 1`, `1 → 2`, `-2 → 3`, ...
#[inline]
pub fn zigzag_encode(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// Inverse of [`zigzag_encode`].
#[inline]
pub fn zigzag_decode(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

/// Number of bits needed to bit-pack ZigZag values whose source range is `[-max_abs, max_abs]`.
/// The maximum ZigZag-encoded value is `2 * max_abs`. Returns 0 for `max_abs == 0`.
#[inline]
pub fn bits_for_max_abs(max_abs: u64) -> u8 {
    if max_abs == 0 {
        return 0;
    }
    let zz_max = max_abs.saturating_mul(2);
    (64 - zz_max.leading_zeros()) as u8
}

/// Write `bits` low-order bits of `value` into `dest` starting at the `bit_offset`th bit
/// (LSB-first within each byte). Caller guarantees `dest` is large enough.
#[inline]
pub fn write_bits(dest: &mut [u8], bit_offset: usize, mut value: u64, bits: u8) {
    if bits == 0 {
        return;
    }
    debug_assert!(bits <= 64);
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    value &= mask;
    let mut byte_idx = bit_offset / 8;
    let in_byte = (bit_offset % 8) as u8;
    let mut remaining = bits;
    let mut shifted = value << in_byte;
    let total_bits_to_write = u32::from(remaining) + u32::from(in_byte);
    let mut bytes_to_touch = total_bits_to_write.div_ceil(8) as usize;

    // First byte: OR in the low (8 - in_byte) bits.
    if remaining > 0 {
        let take = 8 - in_byte;
        let take = take.min(remaining);
        let chunk = (shifted & 0xFF) as u8;
        dest[byte_idx] |= chunk & (0xFFu8 << in_byte);
        let _ = take;
        shifted >>= 8;
        bytes_to_touch -= 1;
        byte_idx += 1;
        // After the first byte we have `remaining - take` bits left in `shifted` to write.
        if remaining > (8 - in_byte) {
            remaining -= 8 - in_byte;
        } else {
            remaining = 0;
        }
    }

    // Subsequent bytes: write 8 bits at a time.
    while remaining >= 8 {
        dest[byte_idx] = (shifted & 0xFF) as u8;
        shifted >>= 8;
        byte_idx += 1;
        remaining -= 8;
        bytes_to_touch = bytes_to_touch.saturating_sub(1);
    }

    // Final partial byte.
    if remaining > 0 {
        let mask = (1u8 << remaining) - 1;
        dest[byte_idx] |= (shifted & u64::from(mask)) as u8 & mask;
    }
}

/// Read `bits` bits from `src` starting at `bit_offset` (LSB-first within each byte) and return
/// them as an unsigned `u64`.
#[inline]
pub fn read_bits(src: &[u8], bit_offset: usize, bits: u8) -> u64 {
    if bits == 0 {
        return 0;
    }
    debug_assert!(bits <= 64);
    let byte_idx = bit_offset / 8;
    let in_byte = (bit_offset % 8) as u8;
    let total_bits = u32::from(bits) + u32::from(in_byte);
    let bytes_to_read = total_bits.div_ceil(8) as usize;
    let mut buf = 0u64;
    for i in 0..bytes_to_read {
        if byte_idx + i >= src.len() {
            break;
        }
        buf |= u64::from(src[byte_idx + i]) << (i * 8);
    }
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    (buf >> in_byte) & mask
}

/// Compute the cumulative bit offset for each piece given per-piece `(piece_len, bit_width)`.
/// Returns a vector of length `P + 1` where the last element is the total number of bits.
pub fn cumulative_offsets(piece_lens: &[u32], widths: &[u8]) -> Vec<u64> {
    debug_assert_eq!(piece_lens.len(), widths.len());
    let mut offsets = Vec::with_capacity(piece_lens.len() + 1);
    let mut total = 0u64;
    offsets.push(total);
    for (l, w) in piece_lens.iter().zip(widths.iter()) {
        total += u64::from(*l) * u64::from(*w);
        offsets.push(total);
    }
    offsets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_roundtrip() {
        for v in [
            0_i64,
            -1,
            1,
            -2,
            2,
            i64::MIN + 1,
            i64::MAX,
            1234567,
            -987654,
        ] {
            let zz = zigzag_encode(v);
            assert_eq!(zigzag_decode(zz), v, "roundtrip failed for {v}");
        }
    }

    #[test]
    fn bits_for_max_abs_known_values() {
        assert_eq!(bits_for_max_abs(0), 0);
        assert_eq!(bits_for_max_abs(1), 2); // 2*1 = 2 → 2 bits (10)
        assert_eq!(bits_for_max_abs(3), 3); // 2*3 = 6 → 3 bits (110)
        assert_eq!(bits_for_max_abs(127), 8); // 2*127 = 254 → 8 bits
    }

    #[test]
    fn pack_unpack_roundtrip_simple() {
        // Pack three 5-bit values: 17, 8, 31.
        let mut buf = vec![0u8; 8];
        write_bits(&mut buf, 0, 17, 5);
        write_bits(&mut buf, 5, 8, 5);
        write_bits(&mut buf, 10, 31, 5);
        assert_eq!(read_bits(&buf, 0, 5), 17);
        assert_eq!(read_bits(&buf, 5, 5), 8);
        assert_eq!(read_bits(&buf, 10, 5), 31);
    }

    #[test]
    fn pack_unpack_roundtrip_zigzag() {
        let values: Vec<i64> = vec![0, -1, 1, -2, 2, -127, 127, -64, 64, 0];
        let max_abs = values.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
        let bits = bits_for_max_abs(max_abs);
        let total_bits = (values.len() as u64) * u64::from(bits);
        let total_bytes = total_bits.div_ceil(8) as usize;
        let mut buf = vec![0u8; total_bytes];
        for (i, &v) in values.iter().enumerate() {
            let bit_offset = i * usize::from(bits);
            write_bits(&mut buf, bit_offset, zigzag_encode(v), bits);
        }
        for (i, &v) in values.iter().enumerate() {
            let bit_offset = i * usize::from(bits);
            let zz = read_bits(&buf, bit_offset, bits);
            assert_eq!(zigzag_decode(zz), v, "roundtrip failed at i={i}");
        }
    }

    #[test]
    fn cumulative_offsets_basic() {
        let lens = [10_u32, 20, 30];
        let widths = [4_u8, 8, 2];
        let off = cumulative_offsets(&lens, &widths);
        assert_eq!(off, vec![0, 40, 200, 260]);
    }
}
