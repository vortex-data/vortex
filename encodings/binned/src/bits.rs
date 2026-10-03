// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Little-endian bit packing.

/// Bytes of zero padding kept after every readable stream, so a read of up to 64 bits can always
/// load whole `u64`s without a bounds check, including the two loads of a wide read.
pub(crate) const PADDING: usize = 16;

/// Largest bit width [`read_bits`] handles in one load.
pub(crate) const MAX_FAST_READ_BITS: u8 = 56;

#[derive(Default)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    n_bits: u32,
}

impl BitWriter {
    /// Appends the low `n_bits` (at most 64) bits of `value`.
    #[inline]
    pub(crate) fn write(&mut self, value: u64, n_bits: u32) {
        if n_bits == 0 {
            return;
        }
        let value = if n_bits == 64 {
            value
        } else {
            value & ((1 << n_bits) - 1)
        };
        self.acc |= value << self.n_bits;
        let total = self.n_bits + n_bits;
        if total >= 64 {
            self.bytes.extend_from_slice(&self.acc.to_le_bytes());
            self.acc = if self.n_bits == 0 {
                0
            } else {
                value >> (64 - self.n_bits)
            };
            self.n_bits = total - 64;
        } else {
            self.n_bits = total;
        }
    }

    /// Bits written so far.
    pub(crate) fn bit_len(&self) -> usize {
        self.bytes.len() * 8 + self.n_bits as usize
    }

    /// Flushes to a byte boundary and returns the bytes written.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        let n_bytes = self.n_bits.div_ceil(8) as usize;
        self.bytes
            .extend_from_slice(&self.acc.to_le_bytes()[..n_bytes]);
        self.bytes
    }
}

/// Loads the 8 bytes at `byte_idx`.
///
/// # Safety
///
/// `byte_idx + 8 <= data.len()`.
#[inline]
pub(crate) unsafe fn load_u64(data: &[u8], byte_idx: usize) -> u64 {
    debug_assert!(byte_idx + 8 <= data.len());
    // SAFETY: the caller guarantees 8 readable bytes at `byte_idx`.
    u64::from_le(unsafe { data.as_ptr().add(byte_idx).cast::<u64>().read_unaligned() })
}

/// Reads `n_bits <= 56` bits starting at bit `bit_idx`.
///
/// # Safety
///
/// `bit_idx / 8 + 8 <= data.len()`.
#[inline]
pub(crate) unsafe fn read_bits(data: &[u8], bit_idx: usize, n_bits: u8) -> u64 {
    debug_assert!(n_bits <= MAX_FAST_READ_BITS);
    // SAFETY: forwarded from the caller.
    let word = unsafe { load_u64(data, bit_idx >> 3) } >> (bit_idx & 7);
    word & ((1u64 << n_bits) - 1)
}

/// Reads `n_bits <= 64` bits starting at bit `bit_idx`.
///
/// # Safety
///
/// `bit_idx / 8 + 12 <= data.len()`.
#[inline]
pub(crate) unsafe fn read_bits_wide(data: &[u8], bit_idx: usize, n_bits: u8) -> u64 {
    if n_bits <= MAX_FAST_READ_BITS {
        // SAFETY: forwarded from the caller.
        return unsafe { read_bits(data, bit_idx, n_bits) };
    }
    // SAFETY: forwarded from the caller; the second read starts 4 bytes later.
    unsafe {
        let lo = read_bits(data, bit_idx, 32);
        let hi = read_bits(data, bit_idx + 32, n_bits - 32);
        lo | (hi << 32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_mixed_widths() {
        let items: Vec<(u64, u32)> = (0..2000u64)
            .map(|i| {
                let w = u32::try_from(i % 65).unwrap_or(0);
                let v = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                (if w == 64 { v } else { v & ((1u64 << w) - 1) }, w)
            })
            .collect();
        let mut w = BitWriter::default();
        for &(v, n) in &items {
            w.write(v, n);
        }
        let mut bytes = w.finish();
        bytes.extend_from_slice(&[0; PADDING + 4]);
        let mut pos = 0usize;
        for &(v, n) in &items {
            // SAFETY: padded above.
            let got = unsafe { read_bits_wide(&bytes, pos, u8::try_from(n).unwrap_or(0)) };
            assert_eq!(got, v);
            pos += n as usize;
        }
    }
}
