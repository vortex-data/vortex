// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A self-contained Elias-Fano codec, depending on nothing beyond `std`.
//!
//! [`encode()`] builds the layout from a non-decreasing sequence of elements, [`element_at`] reads
//! one back, [`Decoder`] turns a whole window back into elements, and [`validate_layout`] checks
//! that stored geometry describes the universe it claims. A [`Window`] is the part of a sequence an
//! array holds; [`slice_bounds`] finds what a slice of one keeps, in absolute positions, so nothing
//! is rewritten.
//!
//! Each element splits into an `l`-bit low part and a high part `element >> l`, set as one bit at
//! position `high + rank + 1` of a bit array. The `+ rank` keeps positions distinct when elements
//! share a high part, so reading element `i` is a `select1` and the inverse is
//! `high = position - rank - 1`. The `+ 1` sentinel aligns the unset bits with the high parts,
//! giving `rank1(select0(h)) == select0(h) - h`, so one `select0` counts the elements below a high
//! part with no rank directory stored. Two sample tables, one position per 256 set bits and one per
//! 512 unset bits, bound the scans over that array.
//!
//! Each table bounds only the bits it counts. A `select1` scan also walks the unset bits between
//! its set bits, and a gap of `g` in the values puts about `g >> l` of them in one place;
//! duplicates do the same to `select0`. Random access is `O(1)` in expectation, but one large gap
//! can stretch nearby scans to the whole upper array.
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
mod slice;
mod upper;
mod validate;

#[cfg(test)]
mod tests;

pub use decode::Decoder;
pub use encode::Encoded;
pub use encode::encode;
pub(crate) use params::LOG_SAMPLING0;
pub(crate) use params::LOG_SAMPLING1;
pub use params::MAX_LOWER_WIDTH;
pub(crate) use params::lower_mask;
pub(crate) use params::lower_width;
pub(crate) use params::num_samples0;
pub(crate) use params::num_samples1;
pub(crate) use params::num_zeros;
pub(crate) use params::upper_len;
pub use read::Layout;
pub use read::LowBits;
pub use read::Window;
pub use read::element_at;
pub use read::position_of_rank;
pub use select::Bits;
pub(crate) use select::select_range;
pub(crate) use select::select_zero_range;
pub use select::window_words;
pub use slice::SliceBounds;
pub use slice::slice_bounds;
pub(crate) use upper::Ones;
pub use upper::Samples;
pub(crate) use upper::UpperBuilder;
pub(crate) use upper::element_of;
pub(crate) use upper::high_of;
pub(crate) use upper::position_of;
pub(crate) use upper::sampled_select;
pub use validate::validate_layout;

/// Bad arguments to the encoder: a sequence it cannot encode or whose layout it cannot represent.
///
/// The two layout variants need an upper array longer than `u64` or `usize` can count: roughly
/// `2^62` elements, or `2^30` on a 32-bit target. In practice only untrusted metadata claims that
/// many.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The sequence is empty.
    Empty,
    /// An element is below its predecessor.
    Decreasing {
        /// The element's index.
        index: usize,
    },
    /// An element exceeds the span the layout is sized for.
    AboveSpan {
        /// The element's index.
        index: usize,
        /// The element.
        element: u64,
        /// The span the encoder was given.
        span: u64,
    },
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
            Self::Empty => f.write_str("an empty sequence has no layout"),
            Self::Decreasing { index } => {
                write!(f, "the element at index {index} is below its predecessor")
            }
            Self::AboveSpan {
                index,
                element,
                span,
            } => write!(
                f,
                "the element {element} at index {index} exceeds the span {span}"
            ),
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
    /// A set bit sits at or below its own rank, so its high part cannot be recovered.
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
    /// The upper array's buffer is not `upper_len` bits rounded up to whole bytes.
    UpperBytes {
        /// How many bytes the layout calls for.
        expected: u64,
        /// How many are stored.
        found: u64,
    },
    /// A whole array's sample table holds a different number of entries than the layout calls for.
    SampleCount {
        /// Which table.
        table: Table,
        /// How many the layout calls for.
        expected: u64,
        /// How many are stored.
        found: u64,
    },
    /// A sample points outside the range its own rank allows.
    SampleOutOfRange {
        /// Which table it is in.
        table: Table,
        /// Its absolute index within that table.
        index: u64,
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
        /// The absolute index at which the order breaks.
        index: u64,
    },
    /// A window's ranks run past the encoded sequence.
    RankRange {
        /// The window's first rank.
        first_rank: u64,
        /// Its element count.
        len: usize,
        /// The encoded element count.
        n: usize,
    },
    /// A window's first bit cannot have its first rank's set bits before it.
    UpperBase {
        /// The window's first bit.
        upper_base: u64,
        /// The window's first rank.
        first_rank: u64,
    },
    /// A window's upper bytes run past the whole array, or hold fewer bits than its elements.
    UpperWindow {
        /// The window's first bit.
        upper_base: u64,
        /// The bytes it holds.
        bytes: u64,
        /// The whole array's length in bits.
        upper_len: u64,
    },
    /// A window's sample table ends past the whole table, or before the samples its ranks need.
    SampleWindow {
        /// Which table.
        table: Table,
        /// The absolute index one past the window's last sample.
        end: u64,
        /// The lowest end its ranks allow.
        min: u64,
        /// The highest end the whole table allows.
        max: u64,
    },
    /// The set bits of two ranks are closer together than the ranks between them.
    PositionsOutOfOrder {
        /// The lower rank.
        first: u64,
        /// The higher rank.
        last: u64,
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
            Self::UpperBytes { expected, found } => {
                write!(f, "upper array is {found} bytes, expected {expected}")
            }
            Self::SampleCount {
                table,
                expected,
                found,
            } => write!(f, "holds {found} {table}-samples, expected {expected}"),
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
            Self::RankRange { first_rank, len, n } => write!(
                f,
                "a window of {len} elements from rank {first_rank} runs past the {n} encoded"
            ),
            Self::UpperBase {
                upper_base,
                first_rank,
            } => write!(
                f,
                "a window at bit {upper_base} cannot start at rank {first_rank}"
            ),
            Self::UpperWindow {
                upper_base,
                bytes,
                upper_len,
            } => write!(
                f,
                "{bytes} upper bytes at bit {upper_base} do not fit the {upper_len}-bit array or \
                 hold the window"
            ),
            Self::SampleWindow {
                table,
                end,
                min,
                max,
            } => write!(
                f,
                "{table}-samples end at index {end}, outside the {min}..={max} the window allows"
            ),
            Self::PositionsOutOfOrder { first, last } => write!(
                f,
                "upper array is malformed: ranks {first} and {last} sit too close together"
            ),
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
