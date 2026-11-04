// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::BitBuffer;

/// Constant-time rank queries over a [`BitBuffer`].
///
/// The rank at position `n` is the number of set bits before `n`. Each entry pairs a 64-bit word
/// of the bitmap with the number of set bits in all preceding words, so a rank query is one load
/// and one population count. Use it in place of [`BitBuffer::count_range`] when a caller makes
/// many rank queries over the same bitmap.
///
/// The query methods are always inlined, so a caller compiled with POPCNT, for example in a
/// [`fearless_simd`] dispatch body, counts with one instruction.
pub struct BitRankDirectory {
    /// `(word, set bits before word)` over the aligned words of the bitmap, followed by one zero
    /// word, so that `rank(len)` is in bounds.
    entries: Vec<(u64, usize)>,

    /// Bit position of logical bit 0 within the first word. The padding bits are zero.
    lead_padding: usize,

    /// Number of bits in the bitmap.
    len: usize,
}

impl BitRankDirectory {
    /// Builds the directory for `bits`.
    #[expect(
        clippy::inline_always,
        reason = "must inline into a target-feature body to use its POPCNT"
    )]
    #[inline(always)]
    pub fn new(bits: &BitBuffer) -> Self {
        let chunks = bits.unaligned_chunks();
        let mut entries = Vec::with_capacity(chunks.chunks().len() + 3);
        let mut preceding = 0usize;
        let mut push = |word: u64| {
            entries.push((word, preceding));
            preceding += word.count_ones() as usize;
        };

        chunks.prefix().into_iter().for_each(&mut push);
        chunks.chunks().iter().copied().for_each(&mut push);
        chunks.suffix().into_iter().for_each(&mut push);
        push(0);

        Self {
            entries,
            lead_padding: chunks.lead_padding(),
            len: bits.len(),
        }
    }

    /// Returns the number of bits in the bitmap.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the bitmap has no bits.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of set bits before position `n`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is greater than the length of the bitmap.
    #[expect(
        clippy::inline_always,
        reason = "must inline into a target-feature body to use its POPCNT"
    )]
    #[inline(always)]
    pub fn rank(&self, n: usize) -> usize {
        assert!(n <= self.len, "rank position {n} exceeds len {}", self.len);

        // SAFETY: the assertion above checks the precondition.
        unsafe { self.rank_unchecked(n) }
    }

    /// Returns the number of set bits before position `n`, without bounds checks.
    ///
    /// # Safety
    ///
    /// `n` must be at most the length of the bitmap.
    #[expect(
        clippy::inline_always,
        reason = "must inline into a target-feature body to use its POPCNT"
    )]
    #[inline(always)]
    pub unsafe fn rank_unchecked(&self, n: usize) -> usize {
        debug_assert!(n <= self.len, "rank position {n} exceeds len {}", self.len);

        let position = n + self.lead_padding;
        let entry_idx = position / 64;

        // SAFETY: the entries cover every word through `lead_padding + len`, plus one zero word,
        // so the caller's `n <= len` keeps `entry_idx` in bounds.
        let (word, preceding) = unsafe { *self.entries.get_unchecked(entry_idx) };
        preceding + (word & ((1u64 << (position % 64)) - 1)).count_ones() as usize
    }
}

impl BitBuffer {
    /// Builds a [`BitRankDirectory`] for constant-time rank queries over this buffer.
    #[inline]
    pub fn rank_directory(&self) -> BitRankDirectory {
        BitRankDirectory::new(self)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::BitBuffer;

    #[rstest]
    #[case(0, 0)]
    #[case(0, 200)]
    #[case(5, 200)]
    #[case(3, 64)]
    #[case(64, 129)]
    #[case(70, 7)]
    fn rank_matches_count_range(#[case] offset: usize, #[case] len: usize) {
        let backing = BitBuffer::from_iter((0..offset + len).map(|i| (i * 7 + i / 5) % 3 == 0));
        let bits = BitBuffer::new_with_offset(backing.inner().clone(), len, offset);
        let directory = bits.rank_directory();

        assert_eq!(directory.len(), len);
        for n in 0..=len {
            assert_eq!(directory.rank(n), bits.count_range(0, n), "rank at {n}");
        }
    }

    #[test]
    #[should_panic(expected = "rank position 11 exceeds len 10")]
    fn rank_rejects_out_of_bounds() {
        let _ = BitBuffer::new_set(10).rank_directory().rank(11);
    }
}
