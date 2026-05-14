// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Pure-Rust reader for the LSB-first bit-packed token stream produced by
// `BitWriter`. The implementation is identical to `vortex-onpair-sys`'s
// helper of the same name; we keep a local copy so this crate doesn't depend
// on the C++ FFI crate.

/// Read `bits` (1..=16) bits from `packed` starting at LSB-first bit position
/// `bit_pos`. Matches OnPair's `BitWriter` layout exactly.
#[inline]
pub fn read_bits_lsb(packed: &[u64], bit_pos: usize, bits: u32) -> u16 {
    debug_assert!((1..=16).contains(&bits));
    let word_idx = bit_pos / 64;
    let bit_off = (bit_pos % 64) as u32;
    let mask: u64 = (1u64 << bits) - 1;
    let low = packed[word_idx] >> bit_off;
    let combined = if bit_off + bits <= 64 {
        low & mask
    } else {
        let high = packed[word_idx + 1] << (64 - bit_off);
        (low | high) & mask
    };
    combined as u16
}

/// Decompress an LSB-first bit-packed token stream into a flat `Vec<u16>`,
/// one element per token. Each `u16` only uses its low `bits` bits.
pub fn unpack_codes_to_u16(packed: &[u64], total_tokens: usize, bits: u32) -> Vec<u16> {
    assert!((9..=16).contains(&bits), "bits must be in [9, 16]");
    let mut out = Vec::with_capacity(total_tokens);
    for t in 0..total_tokens {
        out.push(read_bits_lsb(packed, t * bits as usize, bits));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpack_roundtrips_simple_pattern() {
        // Three 12-bit tokens packed LSB-first into one u64.
        let bits = 12u32;
        let a = 0xABC_u64;
        let b = 0xDEF_u64;
        let c = 0x123_u64;
        // word0 layout: a in bits 0..12, b in 12..24, c in 24..36.
        let word = a | (b << 12) | (c << 24);
        let packed = vec![word, 0];
        assert_eq!(read_bits_lsb(&packed, 0, bits), 0xABC);
        assert_eq!(read_bits_lsb(&packed, 12, bits), 0xDEF);
        assert_eq!(read_bits_lsb(&packed, 24, bits), 0x123);

        let unpacked = unpack_codes_to_u16(&packed, 3, bits);
        assert_eq!(unpacked, vec![0xABC, 0xDEF, 0x123]);
    }
}
