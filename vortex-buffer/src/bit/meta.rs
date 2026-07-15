// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Bound;
use std::ops::RangeBounds;

use vortex_error::VortexExpect;

/// In-memory metadata describing a packed bitset: a normalized bit `offset` (always `< 8`) and a
/// logical bit `len`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BitBufferMeta {
    offset: usize,
    len: usize,
}

impl BitBufferMeta {
    /// Create metadata for a bitset starting at bit `offset` with `len` bits.
    ///
    /// Panics if `offset >= 8`. Use [`from_raw_offset`](Self::from_raw_offset) to normalize a
    /// larger offset.
    pub fn new(offset: usize, len: usize) -> Self {
        assert!(offset < 8, "BitBufferMeta offset must be < 8, got {offset}");
        Self { offset, len }
    }

    /// Normalize a raw bit `offset` into a whole-byte offset plus metadata whose `offset` is
    /// `< 8`.
    ///
    /// Returns `(byte_offset, meta)` so the caller can slice its backing buffer by `byte_offset`
    /// and store the remaining sub-byte offset in `meta`.
    pub fn from_raw_offset(offset: usize, len: usize) -> (usize, Self) {
        (
            offset / 8,
            Self {
                offset: offset % 8,
                len,
            },
        )
    }

    /// Return the leading byte offset and normalized metadata for a logical slice.
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds or its end precedes its start.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> (usize, Self) {
        let start = match range.start_bound() {
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start
                .checked_add(1)
                .vortex_expect("excluded slice start must not overflow"),
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&end) => end
                .checked_add(1)
                .vortex_expect("included slice end must not overflow"),
            Bound::Excluded(&end) => end,
            Bound::Unbounded => self.len,
        };

        assert!(start <= end);
        assert!(start <= self.len);
        assert!(end <= self.len);

        Self::from_raw_offset(self.offset + start, end - start)
    }

    /// The sub-byte bit offset. Always `< 8`.
    #[inline(always)]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// The logical length of the bitset in bits.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the bitset is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of backing bytes required to hold `offset + len` bits.
    #[inline]
    pub fn byte_len(&self) -> usize {
        (self.offset + self.len).div_ceil(8)
    }
}

#[cfg(test)]
mod tests {
    use super::BitBufferMeta;

    #[test]
    fn slice_normalizes_bit_offset() {
        let meta = BitBufferMeta::new(3, 18);

        let (byte_offset, sliced) = meta.slice(7..16);

        assert_eq!(byte_offset, 1);
        assert_eq!(sliced, BitBufferMeta::new(2, 9));
        assert_eq!(sliced.byte_len(), 2);
    }

    #[test]
    fn slice_supports_inclusive_and_unbounded_ranges() {
        let meta = BitBufferMeta::new(5, 20);

        assert_eq!(meta.slice(..), (0, meta));
        assert_eq!(meta.slice(3..=10), (1, BitBufferMeta::new(0, 8)));
        assert_eq!(meta.slice(20..), (3, BitBufferMeta::new(1, 0)));
    }
}
