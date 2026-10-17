// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The bytes and samples a slice of a window keeps, found with two sampled selects.

use std::ops::Range;

use super::LOG_SAMPLING0;
use super::LOG_SAMPLING1;
use super::Malformed;
use super::Samples;
use super::Table;
use super::Window;
use super::high_of;
use super::position_of_rank;

/// What a slice keeps of a window, as ranges into the window's own buffers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceBounds {
    /// The absolute position of the slice's first bit: the set bit of its first rank.
    pub upper_base: u64,
    /// The bytes of the window's upper buffer the slice keeps.
    pub upper: Range<usize>,
    /// The entries of the window's zero-sample table the slice keeps.
    pub samples0: Range<usize>,
    /// The entries of the window's one-sample table the slice keeps.
    pub samples1: Range<usize>,
}

/// The bounds of the slice holding elements `range` of `window`, in an encoded sequence of `n`
/// elements and `upper_len` bits.
///
/// The slice keeps the bits from its first element's set bit to its last's, and the samples between
/// them. Every `select1` of its ranks reads the same bits as before, and so does every `select0` of
/// a high part above its first element's and up to its last's.
///
/// # Panics
///
/// Panics if `range` is empty or ends past `window.len`.
pub fn slice_bounds(
    window: &Window<'_>,
    upper_len: u64,
    n: usize,
    range: Range<usize>,
) -> Result<SliceBounds, Malformed> {
    assert!(
        range.start < range.end && range.end <= window.len,
        "slice {range:?} is out of bounds for a window of {}",
        window.len
    );
    let bits = window.bits(upper_len);
    let (samples0, samples1) = (window.samples0(), window.samples1());
    let first = window.first_rank + range.start as u64;
    let last = window.first_rank + (range.end - 1) as u64;

    let p0 = position_of_rank(bits, samples1, first)?;
    let p1 = position_of_rank(bits, samples1, last)?;
    let (start, end) = (p0 as u64, p1 as u64);
    if end < start + (last - first) {
        return Err(Malformed::PositionsOutOfOrder { first, last });
    }
    let zeros = start.checked_sub(first).filter(|&zeros| zeros > 0).ok_or(
        Malformed::PositionAtOrBelowRank {
            rank: first,
            position: start,
        },
    )?;
    let inconsistent = Malformed::UpperBase {
        upper_base: start,
        first_rank: first,
    };
    // Corrupt input can claim more zeros before the window than the layout has.
    if zeros > upper_len.saturating_sub(n as u64) {
        return Err(inconsistent);
    }
    let h1 = high_of(end, last).ok_or(Malformed::PositionAtOrBelowRank {
        rank: last,
        position: end,
    })?;

    let kept0 = kept(
        samples0,
        zeros >> LOG_SAMPLING0,
        h1 >> LOG_SAMPLING0,
        inconsistent,
    )?;
    let kept1 = kept(
        samples1,
        first >> LOG_SAMPLING1,
        last >> LOG_SAMPLING1,
        inconsistent,
    )?;
    check_ends(samples0, &kept0, Table::Zero, start + 1, end)?;
    check_ends(samples1, &kept1, Table::One, start + 1, end + 1)?;

    let origin = bits.base() / 8;
    Ok(SliceBounds {
        upper_base: start,
        upper: p0 / 8 - origin..(p1 + 1).div_ceil(8) - origin,
        samples0: kept0,
        samples1: kept1,
    })
}

/// The local entries of `samples` at absolute indices `lo..hi`, clamped to what it holds.
fn kept(
    samples: Samples<'_>,
    lo: u64,
    hi: u64,
    inconsistent: Malformed,
) -> Result<Range<usize>, Malformed> {
    let held = samples.len();
    let start = lo.checked_sub(samples.first()).ok_or(inconsistent)?;
    // An empty range can start past the end of the table.
    let start = usize::try_from(start).map_or(held, |start| start.min(held));
    let end = usize::try_from(hi.saturating_sub(samples.first()))
        .map_or(held, |end| end.min(held))
        .max(start);
    Ok(start..end)
}

/// Refuse a kept table whose first or last sample lies outside `min..max`.
///
/// The input's tables were validated as strictly increasing, so the ends bound every sample kept.
fn check_ends(
    samples: Samples<'_>,
    kept: &Range<usize>,
    table: Table,
    min: u64,
    max: u64,
) -> Result<(), Malformed> {
    let Some(last) = kept.end.checked_sub(1).filter(|&last| last >= kept.start) else {
        return Ok(());
    };
    for local in [kept.start, last] {
        let sample = samples.get(local).unwrap_or(u64::MAX);
        if !(min..max).contains(&sample) {
            return Err(Malformed::SampleOutOfRange {
                table,
                index: samples.first() + local as u64,
                sample,
                min,
                max,
            });
        }
    }
    Ok(())
}
