// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-FileCopyrightText: Copyright the Apache Arrow contributors

//! Iterators over packed bitmaps that yield individual booleans, indices, or runs.
//!
//! These types were originally ported from `arrow-buffer` so that `vortex-buffer` can
//! avoid depending on Arrow.

use crate::bit::chunk_iterator::UnalignedBitChunk;
use crate::bit::chunk_iterator::UnalignedBitChunkIterator;
use crate::bit::get_bit_unchecked;

/// Iterator over the bits within a packed bitmap.
///
/// For efficient iteration over only the set bits, see [`BitIndexIterator`] and
/// [`BitSliceIterator`].
#[derive(Clone)]
pub struct BitIterator<'a> {
    buffer: &'a [u8],
    current_offset: usize,
    end_offset: usize,
}

impl<'a> BitIterator<'a> {
    /// Create a new [`BitIterator`] over `buffer` covering `len` bits starting at bit `offset`.
    ///
    /// # Panic
    ///
    /// Panics if `buffer` is too short for the provided `offset` and `len`.
    pub fn new(buffer: &'a [u8], offset: usize, len: usize) -> Self {
        let end_offset = offset.checked_add(len).unwrap();
        let required_len = end_offset.div_ceil(8);
        assert!(
            buffer.len() >= required_len,
            "BitIterator buffer too small, expected {required_len} got {}",
            buffer.len()
        );

        Self {
            buffer,
            current_offset: offset,
            end_offset,
        }
    }
}

impl Iterator for BitIterator<'_> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_offset == self.end_offset {
            return None;
        }
        // SAFETY: the constructor ensures all offsets are within bounds.
        let v = unsafe { get_bit_unchecked(self.buffer.as_ptr(), self.current_offset) };
        self.current_offset += 1;
        Some(v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_bits = self.end_offset - self.current_offset;
        (remaining_bits, Some(remaining_bits))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        // When `n == 0` we want the immediate next value, so we advance by `n` (not `n - 1`).
        match self.current_offset.checked_add(n) {
            Some(new_offset) if new_offset < self.end_offset => {
                self.current_offset = new_offset;
            }
            _ => {
                self.current_offset = self.end_offset;
                return None;
            }
        }

        self.next()
    }

    fn last(mut self) -> Option<Self::Item> {
        if self.current_offset == self.end_offset {
            return None;
        }

        self.current_offset = self.end_offset - 1;
        self.next()
    }

    fn max(self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        if self.current_offset == self.end_offset {
            return None;
        }

        // `true` is greater than `false`, so the max is `true` iff any set bit remains.
        let mut bit_index_iter = BitIndexIterator::new(
            self.buffer,
            self.current_offset,
            self.end_offset - self.current_offset,
        );

        if bit_index_iter.next().is_some() {
            return Some(true);
        }

        Some(false)
    }
}

impl ExactSizeIterator for BitIterator<'_> {}

impl DoubleEndedIterator for BitIterator<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.current_offset == self.end_offset {
            return None;
        }
        self.end_offset -= 1;
        // SAFETY: the constructor ensures all offsets are within bounds.
        let v = unsafe { get_bit_unchecked(self.buffer.as_ptr(), self.end_offset) };
        Some(v)
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        match self.end_offset.checked_sub(n) {
            Some(new_offset) if self.current_offset < new_offset => {
                self.end_offset = new_offset;
            }
            _ => {
                self.current_offset = self.end_offset;
                return None;
            }
        }

        self.next_back()
    }
}

/// Iterator of contiguous ranges of set bits within a packed bitmap.
///
/// Yields `(start, end)` tuples, where `start` is inclusive and `end` is exclusive.
#[derive(Debug)]
pub struct BitSliceIterator<'a> {
    iter: UnalignedBitChunkIterator<'a>,
    len: usize,
    current_offset: i64,
    current_chunk: u64,
}

impl<'a> BitSliceIterator<'a> {
    /// Create a new [`BitSliceIterator`] over `buffer` covering `len` bits starting at bit `offset`.
    pub fn new(buffer: &'a [u8], offset: usize, len: usize) -> Self {
        let chunk = UnalignedBitChunk::new(buffer, offset, len);
        let mut iter = chunk.iter();

        let current_offset = -(chunk.lead_padding() as i64);
        let current_chunk = iter.next().unwrap_or(0);

        Self {
            iter,
            len,
            current_offset,
            current_chunk,
        }
    }

    /// Returns `Some((chunk_offset, bit_offset))` for the next chunk with at least one set bit,
    /// or `None` if no such chunk remains.
    fn advance_to_set_bit(&mut self) -> Option<(i64, u32)> {
        loop {
            if self.current_chunk != 0 {
                let bit_pos = self.current_chunk.trailing_zeros();
                return Some((self.current_offset, bit_pos));
            }

            self.current_chunk = self.iter.next()?;
            self.current_offset += 64;
        }
    }
}

impl Iterator for BitSliceIterator<'_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }

        let (start_chunk, start_bit) = self.advance_to_set_bit()?;

        // Mark bits up to `start` as already consumed so the end-search skips them.
        self.current_chunk |= (1u64 << start_bit) - 1;

        loop {
            if self.current_chunk != u64::MAX {
                let end_bit = self.current_chunk.trailing_ones();

                self.current_chunk &= !((1u64 << end_bit) - 1);

                return Some((
                    (start_chunk + start_bit as i64) as usize,
                    (self.current_offset + end_bit as i64) as usize,
                ));
            }

            match self.iter.next() {
                Some(next) => {
                    self.current_chunk = next;
                    self.current_offset += 64;
                }
                None => {
                    return Some((
                        (start_chunk + start_bit as i64) as usize,
                        std::mem::replace(&mut self.len, 0),
                    ));
                }
            }
        }
    }
}

/// Iterator of indices whose corresponding bit in the packed bitmap is set.
///
/// This provides the best performance on most masks, apart from those with long runs
/// of set bits, where [`BitSliceIterator`] is faster.
#[derive(Debug)]
pub struct BitIndexIterator<'a> {
    current_chunk: u64,
    chunk_offset: i64,
    iter: UnalignedBitChunkIterator<'a>,
}

impl<'a> BitIndexIterator<'a> {
    /// Create a new [`BitIndexIterator`] over `buffer` covering `len` bits starting at bit `offset`.
    pub fn new(buffer: &'a [u8], offset: usize, len: usize) -> Self {
        let chunks = UnalignedBitChunk::new(buffer, offset, len);
        let mut iter = chunks.iter();

        let current_chunk = iter.next().unwrap_or(0);
        let chunk_offset = -(chunks.lead_padding() as i64);

        Self {
            current_chunk,
            chunk_offset,
            iter,
        }
    }
}

impl Iterator for BitIndexIterator<'_> {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_chunk != 0 {
                let bit_pos = self.current_chunk.trailing_zeros();
                self.current_chunk &= self.current_chunk - 1;
                return Some((self.chunk_offset + bit_pos as i64) as usize);
            }

            self.current_chunk = self.iter.next()?;
            self.chunk_offset += 64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BitIndexIterator;
    use super::BitIterator;

    #[test]
    fn test_bit_iterator_size_hint() {
        let mut b = BitIterator::new(&[0b00000011], 0, 2);
        assert_eq!(b.size_hint(), (2, Some(2)));

        b.next();
        assert_eq!(b.size_hint(), (1, Some(1)));

        b.next();
        assert_eq!(b.size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_bit_iterator() {
        let mask = &[0b00010010, 0b00100011, 0b00000101, 0b00010001, 0b10010011];
        let actual: Vec<_> = BitIterator::new(mask, 0, 5).collect();
        assert_eq!(actual, &[false, true, false, false, true]);

        let actual: Vec<_> = BitIterator::new(mask, 4, 5).collect();
        assert_eq!(actual, &[true, false, false, false, true]);

        let actual: Vec<_> = BitIterator::new(mask, 12, 14).collect();
        assert_eq!(
            actual,
            &[
                false, true, false, false, true, false, true, false, false, false, false, false,
                true, false
            ]
        );

        assert_eq!(BitIterator::new(mask, 0, 0).count(), 0);
        assert_eq!(BitIterator::new(mask, 40, 0).count(), 0);
    }

    #[test]
    #[should_panic(expected = "BitIterator buffer too small, expected 3 got 2")]
    fn test_bit_iterator_bounds() {
        let mask = &[223, 23];
        BitIterator::new(mask, 17, 0);
    }

    #[test]
    fn test_bit_iterator_last_max() {
        let mask = &[0b00010010, 0b00100011];
        let it = BitIterator::new(mask, 0, 16);
        assert_eq!(it.clone().last(), Some(false));
        assert_eq!(it.max(), Some(true));

        let empty = BitIterator::new(mask, 0, 0);
        assert_eq!(empty.clone().last(), None);
        assert_eq!(empty.max(), None);
    }

    #[test]
    fn test_bit_iterator_double_ended() {
        let mask = &[0b00010010];
        let mut it = BitIterator::new(mask, 0, 8);
        assert_eq!(it.next_back(), Some(false));
        assert_eq!(it.next_back(), Some(false));
        assert_eq!(it.next_back(), Some(false));
        assert_eq!(it.next(), Some(false));
        assert_eq!(it.next_back(), Some(true));
    }

    #[test]
    fn test_bit_iterator_nth() {
        let mask = &[0b00010010];
        let mut it = BitIterator::new(mask, 0, 8);
        assert_eq!(it.nth(1), Some(true));
        assert_eq!(it.nth(2), Some(true));
        assert_eq!(it.nth(100), None);
    }

    #[test]
    fn test_bit_index_iterator() {
        let mask = &[0b00010010];
        let indices: Vec<_> = BitIndexIterator::new(mask, 0, 8).collect();
        assert_eq!(indices, vec![1, 4]);

        let mask = &[0xFF; 16];
        let indices: Vec<_> = BitIndexIterator::new(mask, 0, 128).collect();
        assert_eq!(indices, (0..128).collect::<Vec<_>>());

        let mask = &[0x00; 16];
        let indices: Vec<_> = BitIndexIterator::new(mask, 0, 128).collect();
        assert!(indices.is_empty());
    }

    #[test]
    fn test_bit_slice_iterator() {
        let mask = &[0b11110011, 0b00000001];
        let slices: Vec<_> = super::BitSliceIterator::new(mask, 0, 16).collect();
        // Bits 4..8 (in byte 0) and bit 8 (bit 0 of byte 1) are adjacent so they merge.
        assert_eq!(slices, vec![(0, 2), (4, 9)]);

        // Non-adjacent runs stay separate.
        let mask = &[0b11110011, 0b00000010];
        let slices: Vec<_> = super::BitSliceIterator::new(mask, 0, 16).collect();
        assert_eq!(slices, vec![(0, 2), (4, 8), (9, 10)]);
    }
}
