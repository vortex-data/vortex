// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Random access over an encoded sequence, in expected `O(1)`; the [module docs](super) give the
//! worst case.
//!
//! Reading element `i` is one sampled `select1` for its high part and one read of its low part.

use super::Bits;
use super::LOG_SAMPLING0;
use super::LOG_SAMPLING1;
use super::Malformed;
use super::ReadError;
use super::Samples;
use super::element_of;
use super::high_of;
use super::lower_mask;
use super::sampled_select;

/// The low bits of each element, supplied by the caller.
///
/// Asked for one rank at a time, as a read decides which it needs, so a caller keeps its own
/// storage rather than flattening it first. A supplier need not mask what it returns: the reader
/// keeps only `lower_width` bits.
pub trait LowBits {
    /// What supplying a low part can fail with. Use [`Infallible`](core::convert::Infallible) for a
    /// source that cannot, such as a slice already in memory.
    type Error;

    /// The low part of the element at absolute rank `rank`.
    fn get(&mut self, rank: u64) -> Result<u64, Self::Error>;
}

/// The part of an encoded sequence a stored array holds: its upper bytes, both sample tables, and
/// where it starts.
///
/// A whole sequence has `upper_base` and `first_rank` at 0. A slice keeps the bits from its first
/// element's set bit on, so `upper_base` is that bit and `first_rank` the set bits before it.
#[derive(Clone, Copy, Debug)]
pub struct Window<'a> {
    /// The upper array's bytes, from byte `upper_base / 8` of the whole array.
    pub upper: &'a [u8],
    /// The absolute position of the window's first bit.
    pub upper_base: u64,
    /// The rank of the window's first element: the number of set bits before `upper_base`.
    pub first_rank: u64,
    /// The number of elements in the window.
    pub len: usize,
    /// The zero-samples the window holds, as little-endian `u64`s.
    pub samples0: &'a [u8],
    /// The one-samples the window holds, as little-endian `u64`s.
    pub samples1: &'a [u8],
}

impl<'a> Window<'a> {
    /// The number of unset bits before the window.
    #[inline]
    pub fn zeros_before(&self) -> u64 {
        self.upper_base.saturating_sub(self.first_rank)
    }

    /// The window's bits in absolute positions, cut off at `upper_len`.
    pub fn bits(&self, upper_len: u64) -> Bits<'a> {
        let Ok(base) = usize::try_from(self.upper_base) else {
            return Bits::new(&[], 0, 0);
        };
        let available = self.upper.len().saturating_mul(8);
        let offset = (base % 8).min(available);
        let rest = usize::try_from(upper_len.saturating_sub(self.upper_base)).unwrap_or(usize::MAX);
        let len = (available - offset).min(rest).min(usize::MAX - base);
        Bits::with_base(self.upper, offset, base, len)
    }

    /// The zero-sample table, from absolute index `zeros_before >> LOG_SAMPLING0`.
    #[inline]
    pub fn samples0(&self) -> Samples<'a> {
        Samples::windowed(self.samples0, self.zeros_before(), LOG_SAMPLING0)
    }

    /// The one-sample table, from absolute index `first_rank >> LOG_SAMPLING1`.
    #[inline]
    pub fn samples1(&self) -> Samples<'a> {
        Samples::windowed(self.samples1, self.first_rank, LOG_SAMPLING1)
    }
}

/// The borrowed parts of an encoded sequence a read needs.
///
/// Every field is a slice the caller already holds, so describing a layout copies nothing.
/// `first_rank` and `len` narrow it to one window of a longer encoded sequence.
#[derive(Clone, Copy, Debug)]
pub struct Layout<'a> {
    upper: Bits<'a>,
    samples1: Samples<'a>,
    lower_width: u8,
    first_rank: u64,
    len: usize,
}

impl<'a> Layout<'a> {
    /// Describe a window of an encoded sequence.
    ///
    /// `upper` and `samples1` may cover only part of the encoded sequence, in absolute positions.
    /// `first_rank` is where this window starts within the encoded sequence and `len` how many
    /// elements it covers. `lower_width` must be at most 63, as
    /// [`validate_layout`](super::validate_layout) guarantees for a stored layout.
    pub fn new(
        upper: Bits<'a>,
        samples1: Samples<'a>,
        lower_width: u8,
        first_rank: u64,
        len: usize,
    ) -> Self {
        Self {
            upper,
            samples1,
            lower_width,
            first_rank,
            len,
        }
    }

    /// How many elements this layout covers.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this layout covers no elements at all.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The bit position of the set bit belonging to absolute rank `rank`, as a sampled `select1`.
    ///
    /// `samples1` bounds the window at `1 << LOG_SAMPLING1` ones; the unset bits between them are
    /// unbounded, so a gap in the values widens it.
    fn position_of_rank(&self, rank: u64) -> Result<usize, Malformed> {
        position_of_rank(self.upper, self.samples1, rank)
    }

    /// The low bits at `rank`, masked.
    ///
    /// A supplier may hand back bits above `lower_width`, which would bleed into the high part. At
    /// width zero nothing is stored, so the supplier is not asked at all.
    fn low_at<L: LowBits>(&self, rank: u64, low: &mut L) -> Result<u64, ReadError<L::Error>> {
        if self.lower_width == 0 {
            return Ok(0);
        }
        low.get(rank)
            .map(|bits| bits & lower_mask(self.lower_width))
            .map_err(ReadError::LowBits)
    }

    /// The element seated at absolute `rank`, given the bit position of its set bit.
    fn element_at_position<L: LowBits>(
        &self,
        position: usize,
        rank: u64,
        low: &mut L,
    ) -> Result<u64, ReadError<L::Error>> {
        let high = high_of(position as u64, rank).ok_or(Malformed::PositionAtOrBelowRank {
            rank,
            position: position as u64,
        })?;
        Ok(element_of(high, self.low_at(rank, low)?, self.lower_width))
    }
}

/// The bit position of the set bit belonging to absolute rank `rank`, taking the upper array and
/// its sample table directly, for a caller that has no [`Layout`] to hand.
///
/// A corrupt sample table can pass [`validate_layout`](super::validate_layout) yet return
/// positions out of rank order, so check a window's bounds before
/// [`window_words`](super::window_words), which panics on a reversed one.
pub fn position_of_rank(
    upper: Bits<'_>,
    samples1: Samples<'_>,
    rank: u64,
) -> Result<usize, Malformed> {
    sampled_select(upper, samples1, LOG_SAMPLING1, rank, false)
        .ok_or(Malformed::NoElementOfRank { rank })
}

/// The element at logical `index`, in one sampled `select1`.
///
/// # Panics
///
/// Panics if `index` is at or beyond the layout's length.
pub fn element_at<L: LowBits>(
    layout: Layout<'_>,
    index: usize,
    low: &mut L,
) -> Result<u64, ReadError<L::Error>> {
    assert!(index < layout.len, "index {index} is out of bounds");
    let rank = layout.first_rank + index as u64;
    let position = layout.position_of_rank(rank)?;
    layout.element_at_position(position, rank, low)
}
