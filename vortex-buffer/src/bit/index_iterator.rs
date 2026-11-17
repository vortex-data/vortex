// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lazy set-bit iteration with bulk skipping of empty bitmap words.

use std::iter::FusedIterator;

use arrow_buffer::ToByteSlice;
use vortex_error::VortexExpect;

use crate::bit::UnalignedBitChunk;

/// Iterates the logical positions of set bits in a bitmap, in ascending order.
///
/// The iterator borrows its bitmap and skips empty word ranges without materializing indices.
/// Word preparation uses Arrow's bitmap chunking implementation and currently supports
/// little-endian targets.
#[derive(Debug)]
pub struct BitIndexIterator<'a> {
    current_chunk: u64,
    // The logical index of bit zero in `current_chunk`. Only the first word can have a
    // negative base: its masked leading padding precedes the logical bitmap's start.
    chunk_offset: i64,
    // Native-endian words after `current_chunk`, excluding the masked suffix. Testing for
    // zero is byte-order independent; a word is converted from little endian when loaded.
    chunks: &'a [u64],
    suffix: Option<u64>,
    len: usize,
}

impl<'a> BitIndexIterator<'a> {
    /// Creates an iterator over `len` bits in `buffer`, starting at bit `offset`.
    ///
    /// A nonempty range requires `buffer` to cover every bit in `offset..offset + len`.
    ///
    /// # Panics
    ///
    /// Panics if a nonempty range extends beyond `buffer`.
    #[inline]
    pub fn new(buffer: &'a [u8], offset: usize, len: usize) -> Self {
        let unaligned = UnalignedBitChunk::new(buffer, offset, len);
        let mut chunks = unaligned.chunks();
        let mut suffix = unaligned.suffix().filter(|&word| word != 0);
        let current_chunk = if let Some(prefix) = unaligned.prefix() {
            prefix
        } else if let Some((&first, rest)) = chunks.split_first() {
            chunks = rest;
            u64::from_le(first)
        } else {
            suffix.take().unwrap_or(0)
        };

        Self {
            current_chunk,
            chunk_offset: -(unaligned.lead_padding() as i64),
            chunks,
            suffix,
            len,
        }
    }

    /// Discards remaining indices below `index`, without consuming any later index.
    ///
    /// Seeking backward does not restore consumed indices. Seeking to or beyond the logical
    /// bitmap length exhausts the iterator.
    #[inline]
    pub fn seek_to(&mut self, index: usize) {
        if index >= self.len {
            self.current_chunk = 0;
            self.chunks = &[];
            self.suffix = None;
            return;
        }

        // The leading padding can put this intermediate just beyond usize::MAX on 32-bit
        // targets, even though the resulting word index still fits usize.
        let relative = index as i128 - i128::from(self.chunk_offset);
        if relative < 0 {
            return;
        }

        let skipped = usize::try_from(relative / 64).vortex_expect("bitmap word index fits usize");
        if skipped != 0 {
            if skipped <= self.chunks.len() {
                self.current_chunk = u64::from_le(self.chunks[skipped - 1]);
                self.chunks = &self.chunks[skipped..];
            } else {
                self.current_chunk = self.suffix.take().unwrap_or(0);
                self.chunks = &[];
            }
            self.chunk_offset += skipped as i64 * 64;
        }
        self.current_chunk &= u64::MAX << (relative % 64);
    }

    #[inline]
    fn advance_to_nonzero_word(&mut self) -> Option<()> {
        if self.chunks.is_empty() && self.suffix.is_none() {
            return None;
        }

        let skipped = first_nonzero_word(self.chunks);

        if let Some(&word) = self.chunks.get(skipped) {
            self.current_chunk = u64::from_le(word);
            self.chunks = &self.chunks[skipped + 1..];
            self.chunk_offset += (skipped as i64 + 1) * 64;
            return Some(());
        }

        let skipped = self.chunks.len();
        self.chunks = &[];
        let suffix = self.suffix.take()?;
        self.chunk_offset += (skipped as i64 + 1) * 64;
        self.current_chunk = suffix;
        Some(())
    }
}

impl Iterator for BitIndexIterator<'_> {
    type Item = usize;

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_chunk == 0 {
            self.advance_to_nonzero_word()?;
        }

        let bit = self.current_chunk.trailing_zeros();
        self.current_chunk &= self.current_chunk - 1;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "masked padding and tail bits keep the index within the usize bitmap length"
        )]
        let index = (self.chunk_offset + i64::from(bit)) as usize;
        Some(index)
    }
}

impl FusedIterator for BitIndexIterator<'_> {}

// Byte-sized position candidates let a SIMD minimum both skip empty blocks and locate
// the first occupied word, without a second search through the winning block.
#[inline]
fn first_nonzero_word(words: &[u64]) -> usize {
    // Avoid a block reduction when the next unread word is already occupied.
    let Some((&word, words)) = words.split_first() else {
        return 0;
    };
    if word != 0 {
        return 0;
    }

    let (blocks, tail) = words.as_chunks::<8>();
    for (index, block) in blocks.iter().enumerate() {
        let bytes = block.as_slice().to_byte_slice();
        let mut first = 64_u8;
        for byte in 0_u8..64 {
            first = first.min(if bytes[usize::from(byte)] != 0 {
                byte
            } else {
                64
            });
        }
        if first < 64 {
            return 1 + index * 8 + usize::from(first) / 8;
        }
    }
    1 + blocks.len() * 8
        + tail
            .iter()
            .position(|&word| word != 0)
            .unwrap_or(tail.len())
}

#[cfg(test)]
mod tests {
    use crate::BitBuffer;
    use crate::bit::BitIndexIterator;

    #[test]
    fn iterates_offset_bitmap_without_padding() {
        let bytes = [0b1101_0011, 0b1110_1001];
        let actual: Vec<_> = BitIndexIterator::new(&bytes, 3, 9).collect();
        assert_eq!(actual, [1, 3, 4, 5, 8]);
    }

    #[test]
    fn skips_empty_words_and_preserves_tail() {
        let bits = BitBuffer::from_indices(1_027, [0, 641, 1_026]);
        assert_eq!(bits.set_indices().collect::<Vec<_>>(), [0, 641, 1_026]);
    }

    #[test]
    fn handles_empty_and_all_set_bitmaps() {
        assert_eq!(BitIndexIterator::new(&[], 0, 0).next(), None);
        assert_eq!(BitIndexIterator::new(&[0, 0], 0, 9).next(), None);
        assert_eq!(
            BitIndexIterator::new(&[0xff], 2, 4).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
    }

    #[test]
    fn seeks_forward_without_restoring_consumed_bits() {
        let bits = BitBuffer::from_indices(300, [0, 5, 63, 64, 130, 299]);
        let mut indices = bits.set_indices();
        assert_eq!(indices.next(), Some(0));
        indices.seek_to(5);
        assert_eq!(indices.next(), Some(5));
        indices.seek_to(1);
        assert_eq!(indices.next(), Some(63));
        indices.seek_to(65);
        assert_eq!(indices.next(), Some(130));
        indices.seek_to(299);
        assert_eq!(indices.next(), Some(299));
        assert_eq!(indices.next(), None);
        indices.seek_to(0);
        assert_eq!(indices.next(), None);
    }

    #[test]
    fn seeks_offset_bitmaps_and_clamps_past_end() {
        let bytes = [0xff; 16];
        let mut indices = BitIndexIterator::new(&bytes, 5, 100);
        indices.seek_to(63);
        assert_eq!(indices.next(), Some(63));
        indices.seek_to(99);
        assert_eq!(indices.next(), Some(99));
        indices.seek_to(usize::MAX);
        assert_eq!(indices.next(), None);
    }
}
