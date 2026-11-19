// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::BitBuffer;
use crate::bit::BitChunkIterator;
use crate::bit::BitChunks;

/// Counts the set bits of a [`BitBuffer`] before positions visited in non-decreasing order.
///
/// Created by [`BitBuffer::rank_cursor`]. All ranks together cost one pass over the buffer's
/// words, where counting each range separately with [`BitBuffer::count_range`] costs a call per
/// range, which dominates when there are many short ranges.
pub struct BitRankCursor<'a> {
    words: BitChunkIterator<'a>,
    /// The final partial word, taken after `words`.
    remainder: Option<u64>,
    /// The word holding the bits from `word_start`.
    word: u64,
    word_start: usize,
    /// The set bits before `word_start`.
    rank_at_word_start: usize,
    len: usize,
}

impl<'a> BitRankCursor<'a> {
    fn new(chunks: BitChunks<'a>, len: usize) -> Self {
        let remainder = (chunks.remainder_len() > 0).then(|| chunks.remainder_bits());
        let mut cursor = Self {
            words: chunks.iter(),
            remainder,
            word: 0,
            word_start: 0,
            rank_at_word_start: 0,
            len,
        };
        cursor.word = cursor.next_word();
        cursor
    }

    #[inline]
    fn next_word(&mut self) -> u64 {
        self.words
            .next()
            .or_else(|| self.remainder.take())
            .unwrap_or(0)
    }

    /// The number of set bits before `position`.
    ///
    /// # Panics
    ///
    /// Panics if `position` exceeds the length of the buffer or is less than the position of a
    /// previous call.
    #[inline]
    pub fn rank(&mut self, position: usize) -> usize {
        assert!(
            position <= self.len,
            "position {position} exceeds length {}",
            self.len
        );
        assert!(
            position >= self.word_start,
            "position {position} precedes a previous position"
        );
        while position >= self.word_start + 64 {
            self.rank_at_word_start += self.word.count_ones() as usize;
            self.word = self.next_word();
            self.word_start += 64;
        }
        let bits = position - self.word_start;
        let below = if bits == 0 {
            0
        } else {
            self.word & (u64::MAX >> (64 - bits))
        };
        self.rank_at_word_start + below.count_ones() as usize
    }
}

impl BitBuffer {
    /// Returns a cursor that counts the set bits before non-decreasing positions.
    pub fn rank_cursor(&self) -> BitRankCursor<'_> {
        BitRankCursor::new(self.chunks(), self.len())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::BitBuffer;

    #[rstest]
    fn rank_matches_count_range(#[values(0, 3, 64, 70)] offset: usize) {
        let len = 1000;
        let bits = BitBuffer::collect_bool(len + offset, |i| (i * 7 + i / 13) % 5 < 2)
            .slice(offset..offset + len);
        let mut cursor = bits.rank_cursor();
        for position in (0..=len).step_by(3).chain([len]) {
            assert_eq!(cursor.rank(position), bits.count_range(0, position));
        }
    }
}
