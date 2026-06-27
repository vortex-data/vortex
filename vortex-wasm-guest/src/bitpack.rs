// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Minimal LSB-first bit packing for unsigned values.
//!
//! This is a small, dependency-free packer used by example encodings (e.g. Frame-of-Reference +
//! bit packing). Value `i` occupies bits `[i * bit_width, (i + 1) * bit_width)` of the output,
//! least-significant-bit first. It is not the fastlanes layout Vortex uses natively — it is kept
//! deliberately simple so a kernel can unpack it in a few lines.

/// The number of bits needed to represent every value up to and including `max` (0 for `max == 0`).
pub fn bit_width(max: u32) -> u8 {
    if max == 0 {
        0
    } else {
        (32 - max.leading_zeros()) as u8
    }
}

/// Number of bytes a packed buffer of `len` values at `bit_width` bits each occupies.
pub fn packed_len(len: usize, bit_width: u8) -> usize {
    (len * bit_width as usize).div_ceil(8)
}

/// Pack `values` into an LSB-first bitstream of `bit_width` bits per value.
///
/// Bits of each value above `bit_width` are ignored; callers must size `bit_width` to fit.
pub fn pack(values: &[u32], bit_width: u8) -> Vec<u8> {
    if bit_width == 0 {
        return Vec::new();
    }
    let mut out = vec![0u8; packed_len(values.len(), bit_width)];
    let mut bit = 0usize;
    for &v in values {
        for b in 0..bit_width as usize {
            if (v >> b) & 1 == 1 {
                out[bit / 8] |= 1 << (bit % 8);
            }
            bit += 1;
        }
    }
    out
}

/// Unpack `len` values of `bit_width` bits each from an LSB-first bitstream.
///
/// Returns all-zero values when `bit_width == 0`. Out-of-range reads are clamped to zero bits so a
/// malformed `packed`/`len`/`bit_width` cannot panic.
pub fn unpack(packed: &[u8], len: usize, bit_width: u8) -> Vec<u32> {
    let mut out = Vec::with_capacity(len);
    if bit_width == 0 {
        out.resize(len, 0);
        return out;
    }
    let mut bit = 0usize;
    for _ in 0..len {
        let mut v = 0u32;
        for b in 0..bit_width as usize {
            let byte = bit / 8;
            let set = packed.get(byte).is_some_and(|x| (x >> (bit % 8)) & 1 == 1);
            if set {
                v |= 1 << b;
            }
            bit += 1;
        }
        out.push(v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let values: Vec<u32> = (0..100).map(|i| (i * 7) % 256).collect();
        let bw = bit_width(*values.iter().max().unwrap());
        assert_eq!(bw, 8);
        let packed = pack(&values, bw);
        assert_eq!(packed.len(), packed_len(values.len(), bw));
        assert_eq!(unpack(&packed, values.len(), bw), values);
    }

    #[test]
    fn zero_width() {
        let values = vec![0u32; 10];
        let bw = bit_width(0);
        assert_eq!(bw, 0);
        assert!(pack(&values, bw).is_empty());
        assert_eq!(unpack(&[], 10, bw), values);
    }

    #[test]
    fn narrow_width() {
        let values = vec![0u32, 1, 2, 3, 1, 0, 3, 2];
        let bw = bit_width(3);
        assert_eq!(bw, 2);
        let packed = pack(&values, bw);
        assert_eq!(packed.len(), 2); // 8 * 2 bits = 16 bits = 2 bytes
        assert_eq!(unpack(&packed, values.len(), bw), values);
    }
}
