// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;

use vortex_scan::selection::Selection;

use crate::scan::IDEAL_SPLIT_SIZE;

/// The maximum number of rows in a single range. This is somewhat arbitrarily chosen.
const MAX_RANGE_SIZE: u64 = IDEAL_SPLIT_SIZE / 25;

/// The minimum gap between ranges. This is somewhat arbitrarily chosen.
const MIN_GAP_BETWEEN_RANGES: u64 = IDEAL_SPLIT_SIZE / 2;

/// The way in which we compute splits for a file.
pub enum Splits {
    /// Natural splits computed by the layout reader (e.g., computing splits across different-sized
    /// column chunks).
    Natural(NaturalSplits),

    /// Exact split ranges.
    ///
    /// This is an optimization for when we know the exact rows we need to get from a file (which is
    /// common if we just want to select a few (sparse) indices).
    Ranges(Vec<Range<u64>>),
}

/// The natural splits of a scan: the boundaries that the layout registers, with the spans between
/// them divided on demand.
///
/// Only the boundaries are stored, and their number follows the size of the layout tree. The
/// number of splits follows the row count instead, and a file can declare any row count in a few
/// bytes: a struct with no fields holds its rows in a layout with no children. The splits are
/// therefore iterated, never collected.
#[derive(Clone, Debug)]
pub struct NaturalSplits {
    /// The boundaries that the layout registers, sorted and deduplicated.
    boundaries: Arc<[u64]>,
    /// How the span between two adjacent boundaries is divided.
    subdivision: Subdivision,
}

/// How the span between two adjacent boundaries is divided into splits.
///
/// Division only adds split points strictly inside a span, so the splits still cover every row
/// between the first and the last boundary exactly once.
#[derive(Clone, Copy, Debug)]
pub enum Subdivision {
    /// The span is one split.
    Whole,
    /// A span wider than this many rows is divided into evenly sized splits of at most this many
    /// rows.
    Even(NonZeroU64),
    /// The span is divided every this many rows. Its last split holds the remainder.
    Every(NonZeroU64),
}

impl Subdivision {
    /// The distance between the split points of a span of `span` rows.
    fn step(self, span: u64) -> u64 {
        match self {
            Self::Whole => span,
            Self::Even(max_rows) if span <= max_rows.get() => span,
            // Use the fewest splits that each fit in `max_rows`, then size them evenly.
            Self::Even(max_rows) => span.div_ceil(span.div_ceil(max_rows.get())),
            Self::Every(rows) => rows.get(),
        }
    }
}

impl NaturalSplits {
    /// Create splits from sorted, deduplicated `boundaries`.
    pub fn new(boundaries: Arc<[u64]>, subdivision: Subdivision) -> Self {
        debug_assert!(
            boundaries.windows(2).all(|w| w[0] < w[1]),
            "split boundaries must be strictly increasing"
        );
        Self {
            boundaries,
            subdivision,
        }
    }

    /// Every split point in ascending order: each boundary, and the points that divide the span
    /// after it.
    pub fn points(&self) -> impl Iterator<Item = u64> + use<> {
        SplitPoints::after(self.clone(), None)
    }

    /// The splits within `row_range` in ascending order, or every split if it is `None`.
    ///
    /// A split that crosses an end of `row_range` is cut there.
    pub fn ranges(&self, row_range: Option<Range<u64>>) -> SplitRanges {
        let row_range =
            row_range.unwrap_or_else(|| match (self.boundaries.first(), self.boundaries.last()) {
                (Some(&first), Some(&last)) => first..last,
                _ => 0..0,
            });
        SplitRanges {
            points: SplitPoints::after(self.clone(), Some(row_range.start)),
            start: row_range.start,
            end: row_range.end,
        }
    }
}

/// The split points of [`NaturalSplits`] in ascending order.
struct SplitPoints {
    splits: NaturalSplits,
    /// The next point to yield.
    next: Option<u64>,
    /// The index of the first boundary at or after `next`.
    upper: usize,
    /// The distance between the split points of the span that ends at `upper`.
    step: u64,
}

impl SplitPoints {
    /// The split points after `row`, or all of them if `row` is `None`.
    fn after(splits: NaturalSplits, row: Option<u64>) -> Self {
        let boundaries = &splits.boundaries;
        let upper = row.map_or(0, |row| boundaries.partition_point(|&b| b <= row));
        let mut next = boundaries.get(upper).copied();
        let mut step = 0;

        // `row` lies in the span from `lo` to `hi`. Compute the first point of that span after
        // `row`, because stepping to it from `lo` takes time in proportion to the row count.
        if let (Some(row), Some(hi), Some(lower)) = (row, next, upper.checked_sub(1)) {
            let lo = boundaries[lower];
            step = splits.subdivision.step(hi - lo);
            let steps = (row - lo) / step + 1;

            // A product or sum that overflows is past the end of the span.
            next = Some(
                steps
                    .checked_mul(step)
                    .and_then(|offset| lo.checked_add(offset))
                    .map_or(hi, |point| point.min(hi)),
            );
        }

        Self {
            splits,
            next,
            upper,
            step,
        }
    }
}

impl Iterator for SplitPoints {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        let point = self.next?;
        let boundaries = &self.splits.boundaries;

        // A boundary starts the next span, which has its own step.
        if boundaries.get(self.upper) == Some(&point) {
            self.upper += 1;
            self.step = boundaries
                .get(self.upper)
                .map_or(0, |&hi| self.splits.subdivision.step(hi - point));
        }

        // A sum that overflows is past the end of the span.
        self.next = boundaries
            .get(self.upper)
            .map(|&hi| point.checked_add(self.step).map_or(hi, |next| next.min(hi)));

        Some(point)
    }
}

/// The row ranges of the splits of [`NaturalSplits`] within a row range, in ascending order.
pub struct SplitRanges {
    /// The split points after `start`.
    points: SplitPoints,
    /// The start of the next range.
    start: u64,
    /// The end of the last range.
    end: u64,
}

impl Iterator for SplitRanges {
    type Item = Range<u64>;

    fn next(&mut self) -> Option<Range<u64>> {
        if self.start >= self.end {
            return None;
        }

        let end = self
            .points
            .next()
            .map_or(self.end, |point| point.min(self.end));
        let range = self.start..end;
        self.start = end;

        Some(range)
    }
}

/// Attempts to compute split ranges from the given selection.
pub fn attempt_split_ranges(
    selection: &Selection,
    row_range: Option<&Range<u64>>,
) -> Option<Vec<Range<u64>>> {
    if row_range.is_some_and(|row_range| row_range.is_empty()) {
        return Some(Vec::new());
    }

    let row_range = row_range.cloned().unwrap_or(0..u64::MAX);

    match selection {
        Selection::IncludeByIndex(buffer) => {
            let indices = buffer.as_slice();
            let start = indices.partition_point(|&index| index < row_range.start);
            let end = indices.partition_point(|&index| index < row_range.end);
            debug_assert!(indices[start..end].is_sorted());
            sparse_ranges(indices[start..end].iter().copied())
        }
        Selection::IncludeRoaring(roaring) => {
            let mut iter = roaring.iter();
            iter.advance_to(row_range.start);
            sparse_ranges(iter.take_while(|&index| index < row_range.end))
        }
        _ => None,
    }
}

/// Builds ranges covering the sorted, unique `indices`, or returns `None` if they are too dense
/// for exact ranges to beat the natural splits.
fn sparse_ranges(mut indices: impl Iterator<Item = u64>) -> Option<Vec<Range<u64>>> {
    let Some(first) = indices.next() else {
        return Some(Vec::new());
    };

    // We need to create ranges that will represent splits that cover our indices.
    // We want to make sure that we do not create too many splits. We also want to make sure our
    // splits do not cover too much as they would overlap column chunk boundaries.

    let mut ranges = Vec::new();
    let mut curr_start = first;
    let mut curr_end = first + 1; // Ranges are exclusive at the end.

    // Build the ranges by iterating over the indices and attempting to extend the current range.
    for idx in indices {
        // Check what the new range size would be if we extend the current range.
        let new_range_size = (idx + 1) - curr_start;
        let gap = (idx + 1) - curr_end;

        if new_range_size >= MAX_RANGE_SIZE {
            // If we need to start a new range, check that it is far enough away.
            if gap >= MIN_GAP_BETWEEN_RANGES {
                // Finalize the current range and start a new one.
                ranges.push(curr_start..curr_end);
                curr_start = idx;
                curr_end = idx + 1;
            } else {
                return None;
            }
        } else {
            // Extend the current range to include this index.
            curr_end = idx + 1;
        }
    }

    // Add the last range.
    ranges.push(curr_start..curr_end);

    Some(ranges)
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;

    use super::*;

    fn even(max_rows: u64) -> Subdivision {
        Subdivision::Even(NonZeroU64::new(max_rows).vortex_expect("max_rows must be non-zero"))
    }

    fn every(rows: u64) -> Subdivision {
        Subdivision::Every(NonZeroU64::new(rows).vortex_expect("rows must be non-zero"))
    }

    /// Collects every split point by walking each span one step at a time.
    fn reference_points(boundaries: &[u64], subdivision: Subdivision) -> Vec<u64> {
        let mut points = Vec::new();
        for span in boundaries.windows(2) {
            let step = usize::try_from(subdivision.step(span[1] - span[0]))
                .vortex_expect("test steps fit in usize");
            points.extend((span[0]..span[1]).step_by(step));
        }
        points.extend(boundaries.last());
        points
    }

    /// Cuts `row_range` at every split point strictly inside it.
    fn reference_ranges(points: &[u64], row_range: Option<Range<u64>>) -> Vec<Range<u64>> {
        let Some(row_range) = row_range else {
            return points.windows(2).map(|w| w[0]..w[1]).collect();
        };
        if row_range.is_empty() {
            return Vec::new();
        }

        let mut cuts = vec![row_range.start];
        cuts.extend(
            points
                .iter()
                .filter(|&&p| row_range.start < p && p < row_range.end),
        );
        cuts.push(row_range.end);
        cuts.windows(2).map(|w| w[0]..w[1]).collect()
    }

    #[rstest]
    #[case::empty(           vec![],            Subdivision::Whole, vec![])]
    #[case::single(          vec![7],           even(2),            vec![7])]
    #[case::whole(           vec![0, 5, 100],   Subdivision::Whole, vec![0, 5, 100])]
    #[case::even_fits(       vec![0, 100],      even(100),          vec![0, 100])]
    #[case::even_remainder(  vec![0, 100],      even(30),           vec![0, 25, 50, 75, 100])]
    #[case::even_wide_spans( vec![0, 5, 15, 16], even(4),           vec![0, 3, 5, 9, 13, 15, 16])]
    #[case::every_remainder( vec![0, 100],      every(30),          vec![0, 30, 60, 90, 100])]
    #[case::every_exact(     vec![10, 40],      every(10),          vec![10, 20, 30, 40])]
    fn points_divide_each_span(
        #[case] boundaries: Vec<u64>,
        #[case] subdivision: Subdivision,
        #[case] expected: Vec<u64>,
    ) {
        assert_eq!(reference_points(&boundaries, subdivision), expected);

        let splits = NaturalSplits::new(boundaries.into(), subdivision);
        assert_eq!(splits.points().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn ranges_match_reference() {
        let mut rng = StdRng::seed_from_u64(12020);
        for _ in 0..1000 {
            let mut boundaries = vec![rng.random_range(0..100u64)];
            for _ in 0..rng.random_range(0..8) {
                boundaries.push(boundaries[boundaries.len() - 1] + rng.random_range(1..200));
            }
            let subdivision = match rng.random_range(0..3) {
                0 => Subdivision::Whole,
                1 => even(rng.random_range(1..60)),
                _ => every(rng.random_range(1..60)),
            };
            // Row ranges start and end before, inside, and after the boundaries.
            let row_range = rng.random_bool(0.9).then(|| {
                let end = boundaries[boundaries.len() - 1] + 50;
                rng.random_range(0..end)..rng.random_range(0..end)
            });

            let points = reference_points(&boundaries, subdivision);
            let splits = NaturalSplits::new(boundaries.clone().into(), subdivision);
            assert_eq!(
                splits.ranges(row_range.clone()).collect::<Vec<_>>(),
                reference_ranges(&points, row_range.clone()),
                "boundaries {boundaries:?}, {subdivision:?}, row range {row_range:?}"
            );
        }
    }

    /// The splits of a huge span are reached by arithmetic, not by stepping through the span.
    #[test]
    fn ranges_of_a_huge_span_are_lazy() {
        let splits = NaturalSplits::new([0, u64::MAX].into(), even(100_000));
        assert_eq!(
            splits.ranges(None).take(2).collect::<Vec<_>>(),
            [0..100_000, 100_000..200_000]
        );

        // The last split is cut at the boundary: the next point would overflow.
        let tail = u64::MAX - 250_000..u64::MAX;
        assert_eq!(
            splits.ranges(Some(tail.clone())).collect::<Vec<_>>(),
            [
                tail.start..18_446_744_073_709_400_000,
                18_446_744_073_709_400_000..18_446_744_073_709_500_000,
                18_446_744_073_709_500_000..u64::MAX,
            ]
        );
    }

    fn include(indices: impl IntoIterator<Item = u64>) -> Selection {
        Selection::IncludeByIndex(
            StrictSortedBuffer::try_new(Buffer::from_iter(indices))
                .vortex_expect("test indices must be strictly sorted"),
        )
    }

    #[test]
    fn split_ranges_intersect_row_range() {
        let ranges = attempt_split_ranges(&include([1, 3, 5, 7, 9]), Some(&(3..9)))
            .vortex_expect("sparse split planning should apply");

        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 3..8);
    }

    #[test]
    fn roaring_split_ranges_match_index_split_ranges() {
        let indices = [1u64, 3, 5, MAX_RANGE_SIZE * 2 + MIN_GAP_BETWEEN_RANGES];
        let roaring = Selection::IncludeRoaring(indices.into_iter().collect());

        for row_range in [None, Some(3..9), Some(6..9), Some(0..u64::MAX)] {
            assert_eq!(
                attempt_split_ranges(&roaring, row_range.as_ref()),
                attempt_split_ranges(&include(indices), row_range.as_ref()),
                "row range {row_range:?}"
            );
        }
    }

    #[test]
    fn dense_roaring_selection_uses_natural_splits() {
        let roaring = Selection::IncludeRoaring((0..MAX_RANGE_SIZE * 2).collect());
        assert_eq!(attempt_split_ranges(&roaring, None), None);
    }

    #[test]
    fn split_ranges_empty_intersection() {
        assert_eq!(
            attempt_split_ranges(&include([1, 3, 5]), Some(&(6..9))),
            Some(Vec::new())
        );
    }

    #[test]
    fn restrictive_row_range_enables_sparse_splits() {
        let indices = (0..MAX_RANGE_SIZE).chain([MAX_RANGE_SIZE + MIN_GAP_BETWEEN_RANGES]);
        let ranges = attempt_split_ranges(&include(indices), Some(&(MAX_RANGE_SIZE..u64::MAX)))
            .vortex_expect("the row range should make sparse split planning applicable");

        assert_eq!(ranges.len(), 1);
        assert_eq!(
            ranges[0],
            MAX_RANGE_SIZE + MIN_GAP_BETWEEN_RANGES..MAX_RANGE_SIZE + MIN_GAP_BETWEEN_RANGES + 1
        );
    }
}
