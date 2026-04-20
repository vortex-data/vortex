// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-FileCopyrightText: Copyright the Apache Arrow contributors

//! Iterators over packed bitmaps in 64-bit chunks.
//!
//! These types were originally ported from `arrow-buffer` so that `vortex-buffer` can
//! avoid depending on Arrow.

use std::fmt::Debug;

/// Iterates over an arbitrarily aligned byte buffer.
///
/// Yields an iterator of aligned `u64`, along with the leading and trailing `u64`
/// necessary to align the buffer to a 8-byte boundary.
///
/// Unlike [`BitChunks`], this exposes both a leading and a trailing `u64`, so that
/// inner iteration can read directly from an aligned `&[u64]`.
#[derive(Debug)]
pub struct UnalignedBitChunk<'a> {
    lead_padding: usize,
    trailing_padding: usize,

    prefix: Option<u64>,
    chunks: &'a [u64],
    suffix: Option<u64>,
}

impl<'a> UnalignedBitChunk<'a> {
    /// Create a new [`UnalignedBitChunk`] over `buffer` covering `len` bits starting at
    /// bit `offset`.
    pub fn new(buffer: &'a [u8], offset: usize, len: usize) -> Self {
        if len == 0 {
            return Self {
                lead_padding: 0,
                trailing_padding: 0,
                prefix: None,
                chunks: &[],
                suffix: None,
            };
        }

        let byte_offset = offset / 8;
        let offset_padding = offset % 8;

        let bytes_len = (len + offset_padding).div_ceil(8);
        let buffer = &buffer[byte_offset..byte_offset + bytes_len];

        let prefix_mask = compute_prefix_mask(offset_padding);

        // If less than 8 bytes, read into prefix
        if buffer.len() <= 8 {
            let (suffix_mask, trailing_padding) = compute_suffix_mask(len, offset_padding);
            let prefix = read_u64(buffer) & suffix_mask & prefix_mask;

            return Self {
                lead_padding: offset_padding,
                trailing_padding,
                prefix: Some(prefix),
                chunks: &[],
                suffix: None,
            };
        }

        // If less than 16 bytes, read into prefix and suffix
        if buffer.len() <= 16 {
            let (suffix_mask, trailing_padding) = compute_suffix_mask(len, offset_padding);
            let prefix = read_u64(&buffer[..8]) & prefix_mask;
            let suffix = read_u64(&buffer[8..]) & suffix_mask;

            return Self {
                lead_padding: offset_padding,
                trailing_padding,
                prefix: Some(prefix),
                chunks: &[],
                suffix: Some(suffix),
            };
        }

        // Read into prefix and suffix as needed
        // SAFETY: `align_to` is safe for any `T` where `T: Sized`.
        let (prefix, mut chunks, suffix) = unsafe { buffer.align_to::<u64>() };
        assert!(
            prefix.len() < 8 && suffix.len() < 8,
            "align_to did not return largest possible aligned slice"
        );

        let (alignment_padding, prefix) = match (offset_padding, prefix.is_empty()) {
            (0, true) => (0, None),
            (_, true) => {
                let prefix = chunks[0] & prefix_mask;
                chunks = &chunks[1..];
                (0, Some(prefix))
            }
            (_, false) => {
                let alignment_padding = (8 - prefix.len()) * 8;

                let prefix = (read_u64(prefix) & prefix_mask) << alignment_padding;
                (alignment_padding, Some(prefix))
            }
        };

        let lead_padding = offset_padding + alignment_padding;
        let (suffix_mask, trailing_padding) = compute_suffix_mask(len, lead_padding);

        let suffix = match (trailing_padding, suffix.is_empty()) {
            (0, _) => None,
            (_, true) => {
                let suffix = chunks[chunks.len() - 1] & suffix_mask;
                chunks = &chunks[..chunks.len() - 1];
                Some(suffix)
            }
            (_, false) => Some(read_u64(suffix) & suffix_mask),
        };

        Self {
            lead_padding,
            trailing_padding,
            prefix,
            chunks,
            suffix,
        }
    }

    /// The number of leading padding bits included in the prefix chunk.
    pub fn lead_padding(&self) -> usize {
        self.lead_padding
    }

    /// The number of trailing padding bits included in the suffix chunk.
    pub fn trailing_padding(&self) -> usize {
        self.trailing_padding
    }

    /// Returns the prefix chunk, if any.
    pub fn prefix(&self) -> Option<u64> {
        self.prefix
    }

    /// Returns the suffix chunk, if any.
    pub fn suffix(&self) -> Option<u64> {
        self.suffix
    }

    /// Returns the 8-byte aligned 64-bit chunks between prefix and suffix.
    pub fn chunks(&self) -> &'a [u64] {
        self.chunks
    }

    /// Returns an iterator over `prefix`, `chunks`, and `suffix` as `u64` values.
    pub fn iter(&self) -> UnalignedBitChunkIterator<'a> {
        self.prefix
            .into_iter()
            .chain(self.chunks.iter().cloned())
            .chain(self.suffix)
    }

    /// Counts the number of set bits across prefix, chunks, and suffix.
    pub fn count_ones(&self) -> usize {
        self.iter().map(|x| x.count_ones() as usize).sum()
    }
}

/// Iterator type for an [`UnalignedBitChunk`].
pub type UnalignedBitChunkIterator<'a> = std::iter::Chain<
    std::iter::Chain<std::option::IntoIter<u64>, std::iter::Cloned<std::slice::Iter<'a, u64>>>,
    std::option::IntoIter<u64>,
>;

#[inline]
fn read_u64(input: &[u8]) -> u64 {
    let len = input.len().min(8);
    let mut buf = [0_u8; 8];
    buf[..len].copy_from_slice(&input[..len]);
    u64::from_le_bytes(buf)
}

#[inline]
fn compute_prefix_mask(lead_padding: usize) -> u64 {
    !((1u64 << lead_padding) - 1)
}

#[inline]
fn compute_suffix_mask(len: usize, lead_padding: usize) -> (u64, usize) {
    let trailing_bits = (len + lead_padding) % 64;

    if trailing_bits == 0 {
        return (u64::MAX, 0);
    }

    let trailing_padding = 64 - trailing_bits;
    let suffix_mask = (1u64 << trailing_bits) - 1;
    (suffix_mask, trailing_padding)
}

/// Iterates over an arbitrarily aligned byte buffer 64 bits at a time.
///
/// [`Self::iter`] yields an iterator of `u64`, along with a trailing `u64` of
/// remainder bits that can be obtained via [`Self::remainder_bits`]. The first
/// byte in the buffer will be the least significant byte in each output `u64`.
#[derive(Debug)]
pub struct BitChunks<'a> {
    buffer: &'a [u8],
    /// Offset inside a byte, always between 0 and 7 (inclusive).
    bit_offset: usize,
    /// Number of complete u64 chunks.
    chunk_len: usize,
    /// Number of remaining bits, always between 0 and 63 (inclusive).
    remainder_len: usize,
}

impl<'a> BitChunks<'a> {
    /// Create a new [`BitChunks`] over `buffer` covering `len` bits starting at `offset`.
    pub fn new(buffer: &'a [u8], offset: usize, len: usize) -> Self {
        assert!(
            (offset + len).div_ceil(8) <= buffer.len(),
            "offset + len out of bounds"
        );

        let byte_offset = offset / 8;
        let bit_offset = offset % 8;

        let chunk_len = len / 64;
        let remainder_len = len % 64;

        BitChunks::<'a> {
            buffer: &buffer[byte_offset..],
            bit_offset,
            chunk_len,
            remainder_len,
        }
    }
}

/// Iterator over chunks of 64 bits represented as a `u64`.
#[derive(Debug)]
pub struct BitChunkIterator<'a> {
    buffer: &'a [u8],
    bit_offset: usize,
    chunk_len: usize,
    index: usize,
}

impl<'a> BitChunks<'a> {
    /// Returns the number of remaining bits, always between 0 and 63 (inclusive).
    #[inline]
    pub const fn remainder_len(&self) -> usize {
        self.remainder_len
    }

    /// Returns the number of complete `u64` chunks.
    #[inline]
    pub const fn chunk_len(&self) -> usize {
        self.chunk_len
    }

    /// Returns the bitmask of remaining bits not covered by a complete `u64` chunk.
    #[inline]
    pub fn remainder_bits(&self) -> u64 {
        let bit_len = self.remainder_len;
        if bit_len == 0 {
            0
        } else {
            let bit_offset = self.bit_offset;
            // Number of bytes to read (can be up to 9 if the offset spans a byte).
            let byte_len = (bit_len + bit_offset).div_ceil(8);
            // SAFETY: the constructor asserts that `offset + len` fits in `buffer`,
            // and we advance the pointer by `chunk_len * 8` bytes which is within
            // the range covered by that assertion.
            let base = unsafe {
                self.buffer
                    .as_ptr()
                    .add(self.chunk_len * size_of::<u64>())
            };

            let mut bits = unsafe { std::ptr::read(base) } as u64 >> bit_offset;
            for i in 1..byte_len {
                let byte = unsafe { std::ptr::read(base.add(i)) };
                bits |= (byte as u64) << (i * 8 - bit_offset);
            }

            bits & ((1u64 << bit_len) - 1)
        }
    }

    /// Returns the total number of `u64`s needed to represent all bits including remainder.
    #[inline]
    pub fn num_u64s(&self) -> usize {
        if self.remainder_len == 0 {
            self.chunk_len
        } else {
            self.chunk_len + 1
        }
    }

    /// Returns the total number of bytes needed to represent all bits including remainder.
    #[inline]
    pub fn num_bytes(&self) -> usize {
        (self.chunk_len * 64 + self.remainder_len).div_ceil(8)
    }

    /// Returns an iterator over chunks of 64 bits as `u64`.
    #[inline]
    pub const fn iter(&self) -> BitChunkIterator<'a> {
        BitChunkIterator::<'a> {
            buffer: self.buffer,
            bit_offset: self.bit_offset,
            chunk_len: self.chunk_len,
            index: 0,
        }
    }

    /// Returns an iterator over chunks of 64 bits, appending the remainder as a final
    /// zero-padded `u64`.
    #[inline]
    pub fn iter_padded(&self) -> impl Iterator<Item = u64> + 'a {
        self.iter().chain(std::iter::once(self.remainder_bits()))
    }
}

impl<'a> IntoIterator for BitChunks<'a> {
    type Item = u64;
    type IntoIter = BitChunkIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Iterator for BitChunkIterator<'_> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<u64> {
        let index = self.index;
        if index >= self.chunk_len {
            return None;
        }

        // SAFETY: we read `chunk_len` u64 words via unaligned reads. The pointer comes
        // from a byte slice, so alignment is handled via `read_unaligned`.
        #[allow(clippy::cast_ptr_alignment)]
        let raw_data = self.buffer.as_ptr() as *const u64;

        // Bit-packed buffers are stored with least-significant-byte-first, so on a
        // big-endian machine the bytes need to be swapped before further processing.
        let current = unsafe { std::ptr::read_unaligned(raw_data.add(index)).to_le() };

        let bit_offset = self.bit_offset;

        let combined = if bit_offset == 0 {
            current
        } else {
            // We read one extra byte to shift in the high bits for the output word.
            let next =
                unsafe { std::ptr::read_unaligned(raw_data.add(index + 1) as *const u8) as u64 };

            (current >> bit_offset) | (next << (64 - bit_offset))
        };

        self.index = index + 1;

        Some(combined)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.chunk_len - self.index,
            Some(self.chunk_len - self.index),
        )
    }
}

impl ExactSizeIterator for BitChunkIterator<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.chunk_len - self.index
    }
}

#[cfg(test)]
mod tests {
    use super::UnalignedBitChunk;
    use crate::Buffer;
    use crate::ByteBuffer;

    fn byte_buffer(bytes: &[u8]) -> ByteBuffer {
        Buffer::copy_from(bytes)
    }

    #[test]
    fn test_iter_aligned() {
        let input: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7];
        let buffer = byte_buffer(input);

        let bitchunks = super::BitChunks::new(buffer.as_slice(), 0, 64);
        let result = bitchunks.into_iter().collect::<Vec<_>>();

        assert_eq!(vec![0x0706050403020100], result);
    }

    #[test]
    fn test_iter_unaligned() {
        let input: &[u8] = &[
            0b00000000, 0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000,
            0b01000000, 0b11111111,
        ];
        let buffer = byte_buffer(input);

        let bitchunks = super::BitChunks::new(buffer.as_slice(), 4, 64);

        assert_eq!(0, bitchunks.remainder_len());
        assert_eq!(0, bitchunks.remainder_bits());

        let result = bitchunks.into_iter().collect::<Vec<_>>();

        assert_eq!(
            vec![0b1111010000000010000000010000000010000000010000000010000000010000],
            result
        );
    }

    #[test]
    fn test_iter_unaligned_remainder_1_byte() {
        let input: &[u8] = &[
            0b00000000, 0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000,
            0b01000000, 0b11111111,
        ];
        let buffer = byte_buffer(input);

        let bitchunks = super::BitChunks::new(buffer.as_slice(), 4, 66);

        assert_eq!(2, bitchunks.remainder_len());
        assert_eq!(0b00000011, bitchunks.remainder_bits());

        let result = bitchunks.into_iter().collect::<Vec<_>>();

        assert_eq!(
            vec![0b1111010000000010000000010000000010000000010000000010000000010000],
            result
        );
    }

    #[test]
    fn test_iter_unaligned_remainder_bits_across_bytes() {
        let input: &[u8] = &[0b00111111, 0b11111100];
        let buffer = byte_buffer(input);

        let bitchunks = super::BitChunks::new(buffer.as_slice(), 6, 7);

        assert_eq!(7, bitchunks.remainder_len());
        assert_eq!(0b1110000, bitchunks.remainder_bits());
    }

    #[test]
    #[should_panic(expected = "offset + len out of bounds")]
    fn test_out_of_bound_panics() {
        let input = vec![0xFF_u8; 16];
        let buffer = byte_buffer(&input);
        let _ = super::BitChunks::new(buffer.as_slice(), 0, (input.len() + 1) * 8);
    }

    #[test]
    fn test_unaligned_bit_chunk_basic() {
        let buffer = byte_buffer(&[0xFF; 5]);
        let unaligned = UnalignedBitChunk::new(buffer.as_slice(), 0, 40);

        assert!(unaligned.chunks().is_empty());
        assert_eq!(unaligned.lead_padding(), 0);
        assert_eq!(unaligned.trailing_padding(), 24);
        assert_eq!(
            unaligned.prefix(),
            Some(0b0000000000000000000000001111111111111111111111111111111111111111)
        );
        assert_eq!(unaligned.suffix(), None);
        assert_eq!(unaligned.count_ones(), 40);
    }
}
