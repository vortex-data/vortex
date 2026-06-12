// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
