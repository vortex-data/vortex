// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter::once;
use std::ops::Range;

use vortex_array::dtype::FieldMask;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::LayoutReader;
use crate::RowSplits;
use crate::SplitRange;
use crate::scan::IDEAL_SPLIT_SIZE;

/// Chunk-boundary spans wider than this are sub-divided into multiple row-range splits so that a
/// file with few, large chunks can be decoded across multiple cores rather than one.
///
/// Reuses [`IDEAL_SPLIT_SIZE`] as the target span per split.
const MAX_SPLIT_ROWS: u64 = IDEAL_SPLIT_SIZE;

/// Defines how the Vortex file is split into batches for reading.
///
/// Note that each split must fit into the platform's maximum usize.
#[derive(Default, Copy, Clone, Debug)]
pub enum SplitBy {
    /// Splits any time there is a chunk boundary in the file, and nowhere else. This yields
    /// splits that follow the file's chunking exactly, trading intra-file decode parallelism
    /// for fewer, larger batches.
    Layout,
    #[default]
    /// Splits like [`SplitBy::Layout`], except that spans between adjacent chunk boundaries
    /// wider than `MAX_SPLIT_ROWS` are further sub-divided so that a file with few, large
    /// chunks can still be decoded across multiple cores.
    LayoutSubSplitting,
    /// Splits every n rows.
    RowCount(usize),
    // UncompressedSize(u64),
}

impl SplitBy {
    /// Compute the splits for the given layout.
    // TODO(ngates): remove this once layout readers are stream based.
    pub fn splits(
        &self,
        layout_reader: &dyn LayoutReader,
        row_range: &Range<u64>,
        field_mask: &[FieldMask],
    ) -> VortexResult<Vec<u64>> {
        Ok(match *self {
            SplitBy::Layout | SplitBy::LayoutSubSplitting => {
                // We usually have under 100 splits so reserving upfront saves
                // us some allocations
                let mut row_splits = RowSplits::new_capacity(128);
                row_splits.push(row_range.start);
                layout_reader.register_splits(
                    field_mask,
                    &SplitRange::root(row_range.clone())?,
                    &mut row_splits,
                )?;
                let boundaries = row_splits.into_sorted_deduped();
                if matches!(self, SplitBy::LayoutSubSplitting) {
                    subdivide_large_spans(boundaries, MAX_SPLIT_ROWS)
                } else {
                    boundaries
                }
            }
            SplitBy::RowCount(n) => row_range
                .clone()
                .step_by(n)
                .chain(once(row_range.end))
                .collect(),
        })
    }
}

/// Sub-divide any gap between adjacent split boundaries that is wider than `max_span` into evenly
/// sized row-range sub-splits.
///
/// `boundaries` is the sorted, deduplicated list of split points produced by the layout (chunk
/// boundaries). Downstream consumers turn this list into half-open ranges by pairing adjacent
/// entries (`tuple_windows().map(|(s, e)| s..e)`), so the row coverage is fully determined by the
/// boundary set. This function only *inserts* points that lie strictly between two existing
/// adjacent boundaries; it never moves or removes a boundary. Splitting `[lo, hi)` at an interior
/// point `m` (with `lo < m < hi`) yields exactly `[lo, m) + [m, hi)`, so the union of ranges is
/// unchanged: the rows are still partitioned contiguously, with no gaps and no overlaps, covering
/// every row exactly once. The output remains sorted and deduplicated.
fn subdivide_large_spans(boundaries: Vec<u64>, max_span: u64) -> Vec<u64> {
    debug_assert!(boundaries.is_sorted(), "boundaries must be sorted");
    debug_assert!(max_span > 0, "max_span must be non-zero");

    // Fast path: nothing to split (also covers the empty / single-boundary cases).
    if boundaries.len() < 2 || boundaries.windows(2).all(|w| w[1] - w[0] <= max_span) {
        return boundaries;
    }

    let mut out = Vec::with_capacity(boundaries.len() * 2);
    for window in boundaries.windows(2) {
        let lo = window[0];
        let hi = window[1];
        // Always emit the lower boundary; the final `hi` is appended once after the loop.
        out.push(lo);

        let span = hi - lo;
        if span > max_span {
            // Number of sub-ranges so that each is <= max_span. `span > max_span` and
            // `max_span >= 1` guarantee `sub_count >= 2`.
            let sub_count = span.div_ceil(max_span);
            // Even sub-range size (rounded up); the last sub-range absorbs any remainder and is
            // bounded by `hi`. Inserted points `lo + k*sub_size` are strictly in `(lo, hi)`.
            let sub_size = span.div_ceil(sub_count);
            let mut point = lo + sub_size;
            while point < hi {
                out.push(point);
                // Saturating: a sum past u64::MAX can never be < `hi`, so the loop exits.
                point = point.saturating_add(sub_size);
            }
        }
    }
    // Append the final boundary (the `hi` of the last window).
    out.push(*boundaries.last().vortex_expect("len >= 2 checked above"));

    debug_assert!(out.is_sorted(), "subdivided boundaries must stay sorted");
    debug_assert!(
        out.windows(2).all(|w| w[0] < w[1]),
        "subdivided boundaries must stay strictly increasing (deduped)"
    );
    out
}

#[cfg(test)]
mod test {
    use std::any::Any;
    use std::sync::Arc;

    use futures::future::BoxFuture;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldPath;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::Expression;
    use vortex_buffer::buffer;
    use vortex_io::runtime::single::block_on;
    use vortex_mask::Mask;

    use super::*;
    use crate::LayoutReaderRef;
    use crate::LayoutStrategy;
    use crate::RowSplits;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::scan::test::SCAN_SESSION;
    use crate::scan::test::session_with_handle;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;

    fn reader() -> LayoutReaderRef {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let layout = block_on(|handle| async {
            let session = session_with_handle(handle);
            FlatLayoutStrategy::default()
                .write_stream(
                    ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    buffer![1_i32; 10]
                        .into_array()
                        .to_array_stream()
                        .sequenced(ptr),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();

        layout
            .new_reader("".into(), segments, &SCAN_SESSION, &Default::default())
            .unwrap()
    }

    #[test]
    fn test_layout_splits_flat() {
        let reader = reader();

        let splits = SplitBy::Layout
            .splits(
                reader.as_ref(),
                &(0..10),
                &[FieldMask::Exact(FieldPath::root())],
            )
            .unwrap();
        assert_eq!(splits, vec![0u64, 10]);
    }

    #[test]
    fn test_row_count_splits() {
        let reader = reader();

        let splits = SplitBy::RowCount(3)
            .splits(
                reader.as_ref(),
                &(0..10),
                &[FieldMask::Exact(FieldPath::root())],
            )
            .unwrap();
        assert_eq!(splits, vec![0u64, 3, 6, 9, 10]);
    }

    /// A reader that registers a fixed set of interior split points (which may repeat) plus the
    /// end of its row range.
    struct StubReader {
        name: Arc<str>,
        dtype: DType,
        row_count: u64,
        interior_splits: Vec<u64>,
    }

    impl StubReader {
        fn new(row_count: u64, interior_splits: Vec<u64>) -> Self {
            Self {
                name: Arc::from("stub"),
                dtype: DType::Primitive(PType::U8, Nullability::NonNullable),
                row_count,
                interior_splits,
            }
        }
    }

    impl LayoutReader for StubReader {
        fn name(&self) -> &Arc<str> {
            &self.name
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn dtype(&self) -> &DType {
            &self.dtype
        }

        fn row_count(&self) -> u64 {
            self.row_count
        }

        fn register_splits(
            &self,
            _field_mask: &[FieldMask],
            split_range: &SplitRange,
            splits: &mut RowSplits,
        ) -> VortexResult<()> {
            for split in &self.interior_splits {
                splits.push(split_range.row_offset() + split);
            }
            splits.push(split_range.root_row_range().end);
            Ok(())
        }

        fn pruning_evaluation(
            &self,
            _: &Range<u64>,
            _: &Expression,
            _: Mask,
        ) -> VortexResult<MaskFuture> {
            unimplemented!()
        }

        fn filter_evaluation(
            &self,
            _: &Range<u64>,
            _: &Expression,
            _: MaskFuture,
        ) -> VortexResult<MaskFuture> {
            unimplemented!()
        }

        fn projection_evaluation(
            &self,
            _: &Range<u64>,
            _: &Expression,
            _: MaskFuture,
        ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
            unimplemented!()
        }
    }

    #[test]
    fn test_layout_splits_dedup() -> VortexResult<()> {
        let reader = StubReader::new(10, vec![5, 5]);
        let splits = SplitBy::Layout.splits(&reader, &(0..10), &[FieldMask::All])?;
        assert_eq!(splits, vec![0u64, 5, 10]);
        Ok(())
    }

    #[test]
    fn test_layout_keeps_large_chunk_whole() -> VortexResult<()> {
        // A single chunk wider than MAX_SPLIT_ROWS: sub-divided by SplitBy::LayoutSubSplitting,
        // left whole by SplitBy::Layout.
        let row_count = MAX_SPLIT_ROWS * 2 + 1;
        let reader = StubReader::new(row_count, vec![]);

        let splits =
            SplitBy::LayoutSubSplitting.splits(&reader, &(0..row_count), &[FieldMask::All])?;
        assert!(splits.len() > 2, "expected sub-divided splits: {splits:?}");

        let splits = SplitBy::Layout.splits(&reader, &(0..row_count), &[FieldMask::All])?;
        assert_eq!(splits, vec![0, row_count]);

        Ok(())
    }

    #[test]
    fn subdivide_below_threshold_is_noop() {
        // Gaps all <= max_span: boundaries returned unchanged.
        assert_eq!(subdivide_large_spans(vec![0, 5, 10], 100), vec![0, 5, 10]);
        assert_eq!(subdivide_large_spans(vec![0, 100], 100), vec![0, 100]);
        assert_eq!(
            subdivide_large_spans(Vec::<u64>::new(), 100),
            Vec::<u64>::new()
        );
        assert_eq!(subdivide_large_spans(vec![7], 100), vec![7]);
    }

    #[test]
    fn subdivide_near_u64_max_does_not_overflow() {
        // The increment past the last interior point would overflow without saturating math.
        let hi = u64::MAX;
        let out = subdivide_large_spans(vec![hi - 3, hi], 2);
        assert_eq!(out, vec![hi - 3, hi - 1, hi]);
    }

    #[test]
    fn subdivide_splits_large_single_chunk() {
        // One large chunk [0, 1000) with max_span 100 -> 10 contiguous sub-splits.
        let out = subdivide_large_spans(vec![0, 1000], 100);
        assert_eq!(
            out,
            vec![0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        );
    }

    #[test]
    fn subdivide_only_large_gaps() {
        // Mixed: [0,50) stays whole, [50, 350) splits into 100-row pieces, [350, 360) stays whole.
        let out = subdivide_large_spans(vec![0, 50, 350, 360], 100);
        assert_eq!(out, vec![0, 50, 150, 250, 350, 360]);
    }

    /// Property: for any sorted, deduped boundary set, subdivision (a) keeps the first and last
    /// boundary, (b) stays strictly increasing, and (c) preserves exact row coverage — the union
    /// of the half-open ranges the consumer derives is identical before and after.
    #[test]
    fn subdivide_preserves_exact_coverage() {
        let cases: Vec<Vec<u64>> = vec![
            vec![0, 1000],
            vec![0, 7, 250_001],
            vec![0, 5, 10, 15, 20, 25, 30],
            vec![3, 1_000_003],
            vec![0, 99_999, 100_000, 300_000],
        ];
        for boundaries in cases {
            let out = subdivide_large_spans(boundaries.clone(), MAX_SPLIT_ROWS);
            // (a) endpoints preserved
            assert_eq!(out.first(), boundaries.first());
            assert_eq!(out.last(), boundaries.last());
            // (b) strictly increasing (sorted + deduped)
            assert!(
                out.windows(2).all(|w| w[0] < w[1]),
                "not strictly increasing: {out:?}"
            );
            // (c) exact coverage: ranges from `out` tile the same span with no gap/overlap, and
            // every original boundary is still present (so original ranges are sub-divided, never
            // merged or shifted).
            let total: u64 = out.windows(2).map(|w| w[1] - w[0]).sum();
            let expected_total = boundaries.last().unwrap() - boundaries.first().unwrap();
            assert_eq!(
                total, expected_total,
                "coverage span changed for {boundaries:?}"
            );
            for b in &boundaries {
                assert!(
                    out.contains(b),
                    "original boundary {b} dropped from {out:?}"
                );
            }
        }
    }
}
