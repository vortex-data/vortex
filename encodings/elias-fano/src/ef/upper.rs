// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The upper bit array: one set bit per element at [`position_of`], one unset bit per bucket
//! boundary, with [`high_of`] inverting it.

use super::Bits;
use super::LOG_SAMPLING0;
use super::LOG_SAMPLING1;
use super::select_range;
use super::select_zero_range;

/// The bit position claimed by the element of rank `rank`.
///
/// The `+ rank` keeps positions distinct when elements share a high part; the `+ 1` is the sentinel
/// that aligns the unset bits with the high parts.
#[inline]
pub fn position_of(element: u64, rank: u64, lower_width: u8) -> u64 {
    (element >> lower_width) + rank + 1
}

/// The high part of the element of rank `rank` sitting at `position`, inverting [`position_of`].
///
/// `None` when the position is at or below its own rank. No array this crate builds can produce
/// that, but the upper buffer's contents are never validated, so a corrupt file can.
#[inline]
pub fn high_of(position: u64, rank: u64) -> Option<u64> {
    position.checked_sub(rank + 1)
}

/// The element with high part `high` and low part `low`.
#[inline]
pub fn element_of(high: u64, low: u64, lower_width: u8) -> u64 {
    (high << lower_width) | low
}

/// A window of one sample table, stored as raw little-endian `u64`s.
///
/// Holds the entries from absolute index `first` on, for a window of the upper array with
/// `base_rank` bits of the counted kind before it. A whole table has both at 0. Deserialized
/// buffers carry no alignment guarantee, hence byte-at-a-time reads rather than a cast.
#[derive(Clone, Copy, Debug)]
pub struct Samples<'a> {
    bytes: &'a [u8],
    first: u64,
    base_rank: u64,
}

impl<'a> Samples<'a> {
    /// A whole table, as the encoder writes it.
    #[inline]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            first: 0,
            base_rank: 0,
        }
    }

    /// The entries a window holds when `base_rank` bits of the counted kind precede it, sampled
    /// once per `1 << log_sampling`.
    #[inline]
    pub fn windowed(bytes: &'a [u8], base_rank: u64, log_sampling: usize) -> Self {
        Self {
            bytes,
            first: base_rank >> log_sampling,
            base_rank,
        }
    }

    /// The number of entries held.
    #[inline]
    pub fn len(&self) -> usize {
        self.bytes.len() / size_of::<u64>()
    }

    /// Whether no entry is held.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The absolute index of the first entry held.
    #[inline]
    pub fn first(&self) -> u64 {
        self.first
    }

    /// The bits of the counted kind before the window.
    #[inline]
    pub fn base_rank(&self) -> u64 {
        self.base_rank
    }

    /// The entry at `local` index, or `None` past the last one.
    #[inline]
    pub fn get(&self, local: usize) -> Option<u64> {
        let at = local.checked_mul(size_of::<u64>())?;
        let chunk = self.bytes.get(at..)?.first_chunk::<8>()?;
        Some(u64::from_le_bytes(*chunk))
    }
}

/// The bit position of the `target`-th set bit of `bits`, or of its `target`-th unset bit when
/// `zeros`, found through the sample table that brackets it.
///
/// `samples` holds one position per `1 << log_sampling` bits of the kind being counted, from rank
/// 1 upward; rank 0 is never stored, since the search starts at the window's first bit anyway. The
/// sample bounds the scan to fewer than `1 << log_sampling` bits of the kind being counted, but not
/// the other kind in between: a gap in the values makes a long run of unset bits, duplicates of set
/// bits.
///
/// Returns the absolute position, or `None` if `target` lies outside the window. `inline(always)`
/// because both callers pass `log_sampling` and `zeros` as constants, which measurably do not fold
/// otherwise.
#[inline(always)]
#[allow(clippy::inline_always)]
pub fn sampled_select(
    bits: Bits<'_>,
    samples: Samples<'_>,
    log_sampling: usize,
    target: u64,
    zeros: bool,
) -> Option<usize> {
    let sample = target >> log_sampling;
    let (start, skipped) = match sample.checked_sub(samples.first + 1) {
        Some(local) => (
            samples.get(usize::try_from(local).ok()?)?,
            sample << log_sampling,
        ),
        // The block's sample lies before the window, so scan from the window's start.
        None => (bits.base() as u64, samples.base_rank),
    };
    let nth = usize::try_from(target.checked_sub(skipped)?).ok()?;
    let start = usize::try_from(start).ok()?;
    if start < bits.base() || start > bits.end() {
        return None;
    }
    let offset = if zeros {
        select_zero_range(bits, start, bits.end(), nth)
    } else {
        select_range(bits, start, bits.end(), nth)
    }?;
    Some(start + offset)
}

/// A walk over the set bits of a window, a `u64` word at a time.
///
/// Costs a `trailing_zeros` and a clear-lowest-bit per element. Takes whole words, so an unaligned
/// window goes through [`window_words`](super::window_words) first.
pub struct Ones<'a> {
    words: &'a [u64],
    /// Index of the word `current` was taken from.
    word: usize,
    /// The bits of that word not yet returned.
    current: u64,
}

impl<'a> Ones<'a> {
    /// Walk the set bits of `words`, in order.
    pub fn new(words: &'a [u64]) -> Self {
        Self {
            words,
            word: 0,
            current: words.first().copied().unwrap_or(0),
        }
    }
}

impl Iterator for Ones<'_> {
    type Item = usize;

    /// The next set bit's index within the window, or `None` once the words run out.
    #[inline]
    fn next(&mut self) -> Option<usize> {
        while self.current == 0 {
            self.word += 1;
            self.current = *self.words.get(self.word)?;
        }
        let bit = self.current.trailing_zeros() as usize;
        self.current &= self.current - 1;
        Some(self.word * u64::BITS as usize + bit)
    }
}

/// Builds the upper array and both sample tables together, in one pass over the elements.
///
/// A zero-sample is the position of a sampled *unset* bit, and the unset runs are only known as the
/// set bits bounding them are written, so the tables cannot be built in a later pass.
pub struct UpperBuilder {
    bits: Vec<u8>,
    len: usize,
    samples0: Vec<u64>,
    samples1: Vec<u64>,
    /// The next unset-bit rank owed a sample. Sample 0 is never stored, for either table: the
    /// sentinel puts the 0th unset bit at position 0 and the 0th set bit is the array's first, both
    /// of which a reader can assume.
    next_zero_sample: u64,
}

impl UpperBuilder {
    /// An all-unset array of `upper_len` bits, with both sample tables empty.
    pub fn new(upper_len: usize) -> Self {
        Self {
            bits: vec![0u8; upper_len.div_ceil(8)],
            len: upper_len,
            samples0: Vec::new(),
            samples1: Vec::new(),
            next_zero_sample: 1 << LOG_SAMPLING0,
        }
    }

    /// Record the element of rank `rank` as a set bit at `position`.
    ///
    /// Must be called with strictly increasing `rank` and `position`.
    pub fn push(&mut self, rank: u64, position: u64) {
        self.sample_zeros_below(position, rank);

        debug_assert!(position < self.len as u64, "position out of bounds");
        self.bits[(position / 8) as usize] |= 1 << (position % 8);

        if rank > 0 && rank.is_multiple_of(1 << LOG_SAMPLING1) {
            self.samples1.push(position);
        }
    }

    /// Emit a zero-sample for every sampled unset rank below `position`.
    ///
    /// An unset bit in this run has exactly `ones` set bits before it, so its rank is
    /// `position - ones` — inverted here to get the position back.
    fn sample_zeros_below(&mut self, position: u64, ones: u64) {
        while self.next_zero_sample + ones < position {
            self.samples0.push(self.next_zero_sample + ones);
            self.next_zero_sample += 1 << LOG_SAMPLING0;
        }
    }

    /// Close the array, returning the upper bytes, the zero-samples and the one-samples.
    pub fn finish(mut self, n: u64, upper_len: u64) -> (Vec<u8>, Vec<u64>, Vec<u64>) {
        // The trailing unset bits past the last element, which `push` never reached: the bucket
        // boundaries above the maximum element's high part, plus the guard zero.
        self.sample_zeros_below(upper_len, n);

        (self.bits, self.samples0, self.samples1)
    }
}
