// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Splits a scan from its plans rather than from the layout reader.
//!
//! A scan splits twice. The filter runs over filter splits, cut where the chunks of the columns the
//! filter reads start, or of those the projection reads when the scan has no filter, and capped in
//! size so every thread has several splits to run. Under each filter split, the projection runs
//! over one or more projection splits, cut where the chunks of the columns the projection reads
//! start, so each projection split reads at most one chunk of every column.

use std::iter;
use std::ops::Range;

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::plan::Concat;
use crate::plan::Eval;
use crate::plan::Filter;
use crate::plan::Pack;
use crate::plan::PlanRef;
use crate::plan::Take;

/// The most rows a filter split has, which amortises a split's fixed cost over a large scan.
const MAX_SPLIT_ROWS: u64 = 1 << 16;

/// The fewest rows the cap on a filter split goes down to for a small scan.
const MIN_SPLIT_ROWS: u64 = 1 << 13;

/// How many filter splits a scan aims to give each thread, so threads that finish early find more
/// work.
const SPLITS_PER_THREAD: u64 = 4;

/// The most rows a filter split of a scan over `rows` rows has when `threads` threads run it.
pub(super) fn max_split_rows(rows: u64, threads: usize) -> u64 {
    let splits = SPLITS_PER_THREAD * u64::try_from(threads.max(1)).unwrap_or(u64::MAX);
    (rows / splits).clamp(MIN_SPLIT_ROWS, MAX_SPLIT_ROWS)
}

/// The row positions where the chunks of `plans` start, over their shared row domain, sorted and
/// without duplicates.
///
/// This follows plans whose children cover the same rows, or known parts of them, and stops at any
/// other plan: a take's values or a list's elements cover other rows.
pub(super) fn chunk_starts<'a>(
    plans: impl IntoIterator<Item = &'a PlanRef>,
) -> VortexResult<Vec<u64>> {
    let mut starts = Vec::new();
    for plan in plans {
        collect_starts(plan, 0, &mut starts)?;
    }
    starts.sort_unstable();
    starts.dedup();
    Ok(starts)
}

fn collect_starts(plan: &PlanRef, offset: u64, starts: &mut Vec<u64>) -> VortexResult<()> {
    if let Some(concat) = plan.as_opt::<Concat>() {
        for (index, &child_offset) in concat.row_offsets().iter().enumerate() {
            starts.push(offset + child_offset);
            collect_starts(
                &concat.child_required(index)?,
                offset + child_offset,
                starts,
            )?;
        }
    } else if let Some(take) = plan.as_opt::<Take>() {
        collect_starts(&take.codes()?, offset, starts)?;
    } else if plan.is::<Pack>() || plan.is::<Filter>() || plan.is::<Eval>() {
        for child in plan.children().iter() {
            collect_starts(&child?, offset, starts)?;
        }
    }
    Ok(())
}

/// The filter splits of `rows`, as every cut including `rows.start` and `rows.end`.
///
/// Rows are cut at `starts`, dropping a cut closer than a quarter of `max_rows` to the previous
/// one, so misaligned chunks of different columns do not leave slivers. Any split longer than
/// `max_rows` is then divided into equal parts no longer than it.
pub(super) fn filter_split_boundaries(starts: &[u64], rows: Range<u64>, max_rows: u64) -> Vec<u64> {
    if rows.is_empty() {
        return Vec::new();
    }
    let min_rows = max_rows / 4;
    let mut cuts = vec![rows.start];
    for &start in starts {
        let previous = cuts[cuts.len() - 1];
        if rows.start < start && start < rows.end && start - previous >= min_rows {
            cuts.push(start);
        }
    }
    // The last split may be a sliver; fold it into the one before.
    if cuts.len() > 1 && rows.end - cuts[cuts.len() - 1] < min_rows {
        cuts.pop();
    }
    cuts.push(rows.end);

    let mut boundaries = vec![rows.start];
    for (start, end) in cuts.into_iter().tuple_windows() {
        let parts = (end - start).div_ceil(max_rows.max(1));
        boundaries.extend((1..parts).map(|part| start + (end - start) * part / parts));
        boundaries.push(end);
    }
    boundaries
}

/// The projection splits of `rows`: cut at every one of `starts` inside it.
pub(crate) fn projection_splits(starts: &[u64], rows: Range<u64>) -> Vec<Range<u64>> {
    let first = starts.partition_point(|&start| start <= rows.start);
    let last = starts.partition_point(|&start| start < rows.end);
    iter::once(rows.start)
        .chain(starts[first..last].iter().copied())
        .chain(iter::once(rows.end))
        .tuple_windows()
        .map(|(start, end)| start..end)
        .filter(|range| !range.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::divided_when_no_starts(&[], 0..150_000, 65_536, vec![0, 50_000, 100_000, 150_000])]
    #[case::at_starts(&[40_000, 80_000], 0..120_000, 65_536, vec![0, 40_000, 80_000, 120_000])]
    #[case::drops_slivers(&[40_000, 41_000, 80_000], 0..120_000, 65_536, vec![0, 40_000, 80_000, 120_000])]
    #[case::folds_short_tail(&[40_000, 115_000], 0..120_000, 131_072, vec![0, 40_000, 120_000])]
    #[case::divides_long_splits(&[40_000], 0..120_000, 65_536, vec![0, 40_000, 80_000, 120_000])]
    #[case::only_inside_rows(&[10_000, 50_000, 200_000], 20_000..100_000, 65_536, vec![20_000, 50_000, 100_000])]
    #[case::empty(&[10], 5..5, 65_536, vec![])]
    fn filter_splits(
        #[case] starts: &[u64],
        #[case] rows: Range<u64>,
        #[case] max_rows: u64,
        #[case] expected: Vec<u64>,
    ) {
        assert_eq!(filter_split_boundaries(starts, rows, max_rows), expected);
    }

    #[rstest]
    #[case::small_scan_floors(150_000, 14, 8_192)]
    #[case::medium_scan(1_500_000, 14, 26_785)]
    #[case::large_scan_caps(6_000_000, 14, 65_536)]
    #[case::no_threads(100_000, 0, 25_000)]
    fn split_rows(#[case] rows: u64, #[case] threads: usize, #[case] expected: u64) {
        assert_eq!(max_split_rows(rows, threads), expected);
    }

    #[rstest]
    #[case::whole(&[], 10..20, vec![10..20])]
    #[case::cut(&[0, 12, 15, 30], 10..20, vec![10..12, 12..15, 15..20])]
    #[case::start_on_boundary(&[10, 15], 10..20, vec![10..15, 15..20])]
    fn projection(
        #[case] starts: &[u64],
        #[case] rows: Range<u64>,
        #[case] expected: Vec<Range<u64>>,
    ) {
        assert_eq!(projection_splits(starts, rows), expected);
    }
}
