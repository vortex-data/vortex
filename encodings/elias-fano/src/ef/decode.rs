// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bulk decode, one run at a time, so a caller can supply low bits a block at a time rather than
//! materialising the whole sequence.

use super::Malformed;
use super::Ones;
use super::element_of;
use super::high_of;
use super::lower_mask;

/// Walks the set bits of an upper window, pairing each with a low part.
pub struct Decoder<'a> {
    ones: Ones<'a>,
    /// Absolute bit position the window starts at, which its set-bit indices are relative to.
    start: usize,
    first_rank: u64,
    len: usize,
    lower_width: u8,
    lower_mask: u64,
    /// How many elements have been written so far.
    rank: usize,
}

impl<'a> Decoder<'a> {
    /// Open a decoder over `words`, the window's bits as whole `u64`s at a zero bit offset.
    ///
    /// Checks what the per-element loop relies on once rather than `len` times: `len` set bits in
    /// the window, and `start > first_rank` so every position exceeds its own rank.
    pub fn new(
        words: &'a [u64],
        start: usize,
        first_rank: u64,
        len: usize,
        lower_width: u8,
    ) -> Result<Self, Malformed> {
        let found = words.iter().map(|word| word.count_ones() as usize).sum();
        if found != len {
            return Err(Malformed::SetBitCount {
                expected: len,
                found,
            });
        }
        if start as u64 <= first_rank {
            return Err(Malformed::PositionAtOrBelowRank {
                rank: first_rank,
                position: start as u64,
            });
        }
        Ok(Self {
            ones: Ones::new(words),
            start,
            first_rank,
            len,
            lower_width,
            lower_mask: lower_mask(lower_width),
            rank: 0,
        })
    }

    /// How many elements are still to be written. A caller feeding runs stops at zero.
    pub fn remaining(&self) -> usize {
        self.len - self.rank
    }

    /// Fold one run of consecutive low parts, calling `out(index, element)` for each element.
    ///
    /// `lows` of `None` is the `lower_width == 0` layout, where the run covers whatever is left.
    ///
    /// Cannot fail: an uninvertible position leaves the count short for [`Self::finish`]. Inlined
    /// because out of line, a zero-width layout has no unpacking to hide the call behind and
    /// decodes at half the speed.
    #[inline]
    pub fn segment(&mut self, lows: Option<&[u64]>, mut out: impl FnMut(usize, u64)) {
        let remaining = self.len - self.rank;
        let take = lows.map_or(remaining, |lows| lows.len().min(remaining));

        for offset in 0..take {
            let index = self.rank + offset;
            let Some(position) = self.ones.next() else {
                self.rank = index;
                return;
            };
            let Some(high) = high_of(
                (self.start + position) as u64,
                self.first_rank + index as u64,
            ) else {
                self.rank = index;
                return;
            };
            // A supplier may hand back bits above `lower_width`, which would bleed into the high
            // part.
            let low = lows.map_or(0, |lows| lows[offset] & self.lower_mask);
            out(index, element_of(high, low, self.lower_width));
        }

        self.rank += take;
    }

    /// Close the decode, refusing a layout that produced fewer elements than it claimed: either a
    /// position that could not be inverted, or runs of low bits that ran short.
    pub fn finish(self) -> Result<(), Malformed> {
        if self.rank != self.len {
            return Err(Malformed::SetBitCount {
                expected: self.len,
                found: self.rank,
            });
        }
        Ok(())
    }
}
