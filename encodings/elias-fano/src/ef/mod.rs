// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A self-contained Elias-Fano codec, depending on nothing beyond `std`.
//!
//! [`encode()`] builds the layout from a non-decreasing sequence of elements, [`element_at`] reads
//! one back, [`Decoder`] turns a whole window back into elements, and [`validate_layout`] checks
//! that stored geometry describes the universe it claims.
//!
//! Each element splits into an `l`-bit low part and a high part `element >> l`, set as one bit at
//! position `high + rank + 1` of a bit array. The `+ rank` keeps positions distinct when elements
//! share a high part, so reading element `i` is a `select1` and the inverse is
//! `high = position - rank - 1`. The `+ 1` sentinel aligns the unset bits with the high parts,
//! giving `rank1(select0(h)) == select0(h) - h`, so one `select0` counts the elements below a high
//! part with no rank directory stored. [`LOG_SAMPLING1`] and [`LOG_SAMPLING0`] bound the scans over
//! that array.
//!
//! The low parts live outside these buffers, supplied through [`LowBits`]: they compress better
//! under a dedicated encoding than anything here would manage.
//!
//! Samples are written native-endian and read back with `from_le_bytes`, so a root building this
//! for a big-endian target wants `#![cfg(target_endian = "little")]`.

use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;

mod decode;
mod encode;
mod params;
mod read;
mod select;
mod upper;
mod validate;

#[cfg(test)]
mod tests;

pub use decode::Decoder;
pub use encode::Encoded;
pub use encode::encode;
pub use params::LOG_SAMPLING0;
pub use params::LOG_SAMPLING1;
pub use params::MAX_LOWER_WIDTH;
pub use params::lower_mask;
pub use params::lower_width;
pub use params::num_samples0;
pub use params::num_samples1;
pub use params::num_zeros;
pub use params::upper_len;
pub use read::Layout;
pub use read::LowBits;
pub use read::element_at;
pub use read::position_of_rank;
pub use select::Bits;
pub use select::select_range;
pub use select::select_zero_range;
pub use select::window_words;
pub use upper::Ones;
pub use upper::UpperBuilder;
pub use upper::element_of;
pub use upper::high_of;
pub use upper::position_of;
pub use upper::read_sample;
pub use upper::sampled_select;
pub use validate::validate_layout;

/// A sequence whose layout cannot be represented, i.e. bad arguments to the encoder.
///
/// Both variants need a universe of very nearly `2^64`, so neither is reachable from a sequence
/// held in memory. The same geometry is re-derived from untrusted metadata, where they are.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The upper array's length overflows a `u64`.
    UpperLenOverflow {
        /// Number of elements.
        n: usize,
        /// The universe's width, `max - reference`.
        span: u64,
        /// Low bits per element.
        lower_width: u8,
    },
    /// The upper array is longer than `usize` can address.
    UpperLenTooLarge {
        /// The length in bits that does not fit.
        upper_len: u64,
    },
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::UpperLenOverflow {
                n,
                span,
                lower_width,
            } => write!(
                f,
                "upper array overflows: n {n}, span {span}, lower_width {lower_width}"
            ),
            Self::UpperLenTooLarge { upper_len } => {
                write!(f, "upper array of {upper_len} bits does not fit in memory")
            }
        }
    }
}

impl core::error::Error for Error {}

/// Which of the two sample tables a fault was found in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Table {
    /// The table sampling unset bits, one per `1 << LOG_SAMPLING0`.
    Zero,
    /// The table sampling set bits, one per `1 << LOG_SAMPLING1`.
    One,
}

impl Display for Table {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(match self {
            Self::Zero => "zero",
            Self::One => "one",
        })
    }
}

/// Buffers that do not describe a valid Elias-Fano sequence.
///
/// [`Error`] is about arguments the encoder cannot serve; this is about a layout handed to a
/// reader. Nothing here builds one, but the upper array's contents are never checked against the
/// elements, so a corrupt file can.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Malformed {
    /// The stored geometry describes a layout that cannot exist at all.
    Unrepresentable(Error),
    /// The upper array runs out before the element of this rank.
    NoElementOfRank {
        /// The rank that has no set bit.
        rank: u64,
    },
    /// A set bit sits at or below its own rank, so [`high_of`] cannot invert it.
    PositionAtOrBelowRank {
        /// The element's rank.
        rank: u64,
        /// The bit it was found at.
        position: u64,
    },
    /// A window holds a different number of set bits than it has elements.
    SetBitCount {
        /// How many the layout calls for.
        expected: usize,
        /// How many were found.
        found: usize,
    },
    /// The stored low-bits width disagrees with the one the universe implies.
    LowerWidth {
        /// The width the universe implies.
        expected: u8,
        /// The width stored.
        found: u8,
    },
    /// The stored upper length disagrees with the one the universe implies.
    UpperLen {
        /// The length the universe implies.
        expected: u64,
        /// The length stored.
        found: u64,
    },
    /// The samples buffer holds a different number of entries than the layout calls for.
    SampleCount {
        /// How many the layout calls for.
        expected: u64,
        /// How many are stored.
        found: u64,
    },
    /// A sample points outside the range its own rank allows.
    SampleOutOfRange {
        /// Which table it is in.
        table: Table,
        /// Its index within that table.
        index: usize,
        /// The position it points to.
        sample: u64,
        /// The lowest position its rank permits.
        min: u64,
        /// One past the highest.
        max: u64,
    },
    /// Samples within one table are not strictly increasing, which a sampled search relies on.
    SamplesNotIncreasing {
        /// Which table.
        table: Table,
        /// The index at which the order breaks.
        index: usize,
    },
}

impl Display for Malformed {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Unrepresentable(error) => write!(f, "{error}"),
            Self::NoElementOfRank { rank } => {
                write!(f, "upper array holds no element of rank {rank}")
            }
            Self::PositionAtOrBelowRank { rank, position } => write!(
                f,
                "upper array is malformed: the element of rank {rank} sits at bit {position}, at \
                 or below its own rank"
            ),
            Self::SetBitCount { expected, found } => write!(
                f,
                "upper array is malformed: expected exactly {expected} set bits above their own \
                 ranks, found {found}"
            ),
            Self::LowerWidth { expected, found } => write!(
                f,
                "lower_width {found} does not match the {expected} its universe implies"
            ),
            Self::UpperLen { expected, found } => write!(
                f,
                "upper_len {found} does not match the {expected} its universe implies"
            ),
            Self::SampleCount { expected, found } => {
                write!(f, "holds {found} samples, expected {expected}")
            }
            Self::SampleOutOfRange {
                table,
                index,
                sample,
                min,
                max,
            } => write!(
                f,
                "{table}-sample {index} points to bit {sample}, outside the {min}..{max} its rank \
                 allows"
            ),
            Self::SamplesNotIncreasing { table, index } => {
                write!(
                    f,
                    "{table}-samples are not strictly increasing at index {index}"
                )
            }
        }
    }
}

impl core::error::Error for Malformed {}

impl From<Error> for Malformed {
    fn from(error: Error) -> Self {
        Self::Unrepresentable(error)
    }
}

/// A read that either met a malformed layout or could not obtain its low bits.
///
/// Generic over the supplier's error, so a host's own type survives the round trip rather than
/// being flattened into a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadError<E> {
    /// The layout does not describe a valid sequence.
    Malformed(Malformed),
    /// The low-bits source could not supply a value.
    LowBits(E),
}

impl<E> From<Malformed> for ReadError<E> {
    fn from(error: Malformed) -> Self {
        Self::Malformed(error)
    }
}

impl<E: Display> Display for ReadError<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Malformed(error) => write!(f, "{error}"),
            Self::LowBits(error) => write!(f, "low bits unavailable: {error}"),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for ReadError<E> {}
