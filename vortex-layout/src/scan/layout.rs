// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::FutureExt;
use futures::stream;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::expr::stats::Precision;
use vortex_array::scalar::Scalar;
use vortex_array::stats::StatsSet;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_metrics::MetricsRegistry;
use vortex_scan::DataSource;
use vortex_scan::Scan;
use vortex_scan::ScanPool;
use vortex_scan::ScanPoolRef;
use vortex_scan::ScanRef;
use vortex_scan::ScanRequest;
use vortex_session::VortexSession;

use crate::LayoutReaderRef;
use crate::scan::metrics::ScanMetrics;
use crate::scan::metrics::ScanRecorder;
use crate::scan::pool::FileWork;
use crate::scan::pool::PlannedSplits;
use crate::scan::pool::PoolOptions;
use crate::scan::pool::PreparedScan;
use crate::scan::pool::SplitPool;
use crate::scan::pool::default_splits_in_flight;
use crate::scan::scan_builder::ScanBuilder;

/// An implementation of a [`DataSource`] that reads data from a [`LayoutReaderRef`].
///
/// A scan's units are the layout's splits. Workers of an opened scan share them, each keeping
/// several in flight on its own thread; see [`Self::with_max_splits_in_flight`].
pub struct LayoutReaderDataSource {
    reader: LayoutReaderRef,
    session: VortexSession,
    metrics: Option<ScanMetrics>,
    max_splits_in_flight: NonZeroUsize,
}

impl LayoutReaderDataSource {
    /// Creates a new [`LayoutReaderDataSource`].
    pub fn new(reader: LayoutReaderRef, session: VortexSession) -> Self {
        Self {
            reader,
            session,
            metrics: None,
            max_splits_in_flight: default_splits_in_flight(),
        }
    }

    /// Bounds the splits in flight at once across all workers of one opened scan.
    ///
    /// Each split in flight holds its pending reads and the decoded chunks of every column it
    /// overlaps, so a scan that runs alongside others can lower this to bound its memory. Each
    /// worker may always have one split in flight. Defaults to four for each available core,
    /// which lets a single worker overlap the reads of several splits on its own thread.
    pub fn with_max_splits_in_flight(mut self, max: NonZeroUsize) -> Self {
        self.max_splits_in_flight = max;
        self
    }

    /// Registers aggregate counters once for this data source.
    pub fn with_metrics_registry(mut self, metrics: Arc<dyn MetricsRegistry>) -> Self {
        self.metrics = Some(ScanMetrics::new(metrics.as_ref()));
        self
    }

    /// Registers aggregate counters once for this data source, or clears them.
    pub fn with_some_metrics_registry(mut self, metrics: Option<Arc<dyn MetricsRegistry>>) -> Self {
        self.metrics = metrics.as_deref().map(ScanMetrics::new);
        self
    }
}

#[async_trait]
impl DataSource for LayoutReaderDataSource {
    fn dtype(&self) -> &DType {
        self.reader.dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::exact(self.reader.row_count())
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<ScanRef> {
        let total_rows = self.reader.row_count();
        let row_range = clamp_row_range(scan_request.row_range, total_rows);

        let projection = scan_request
            .projection
            .bind(self.reader.dtype())?
            .optimize_recursive()?;
        let filter = scan_request
            .filter
            .map(|expr| expr.bind(self.reader.dtype())?.optimize_recursive())
            .transpose()?;
        let dtype = projection.dtype().clone();

        // If the dtype is an empty struct, and there is no filter, we can return a special
        // length-only scan.
        if let DType::Struct(fields, Nullability::NonNullable) = &dtype
            && fields.nfields() == 0
            && filter.is_none()
        {
            let row_count = scan_request.selection.row_count_in_range(&row_range);
            let row_count = scan_request
                .limit
                .map_or(row_count, |limit| row_count.min(limit));
            return Ok(Box::new(ConstantScan { dtype, row_count }));
        }

        // If statistics reject the entire requested range, the scan is empty.
        if let Some(filter) = &filter {
            let mask = Mask::new_true(
                usize::try_from(row_range.end - row_range.start).unwrap_or(usize::MAX),
            );
            let pruning_result = self
                .reader
                .pruning_evaluation(&row_range, filter, mask)?
                .now_or_never();
            if let Some(Ok(result_mask)) = pruning_result
                && result_mask.all_false()
            {
                if let Some(recorder) = ScanRecorder::new(None, self.metrics.clone())
                    && scan_request.limit != Some(0)
                    && scan_request.selection.row_count_in_range(&row_range) != 0
                {
                    // The eager check covers the requested range. Only whole-file statistics
                    // justify a file rejection. Other rejections count as one range attempt.
                    recorder.attempt_file_scan();

                    if self.reader.can_prune_file(filter)? {
                        recorder.prune_file();
                    } else {
                        recorder.attempt_split();

                        if self.reader.split_pruning_is_statistics(filter)? {
                            recorder.prune_split();
                        } else {
                            recorder.empty_split_after_predicate();
                        }
                    }
                }

                return Ok(Box::new(ConstantScan {
                    dtype,
                    row_count: 0,
                }));
            }
        }

        let selected = scan_request.selection.row_count_in_range(&row_range);
        let estimate = scan_request
            .limit
            .map_or(selected, |limit| selected.min(limit));
        let row_count = if filter.is_some() {
            Precision::inexact(estimate)
        } else {
            Precision::exact(estimate)
        };
        let options = LayoutScanOptions {
            dtype,
            ordered: scan_request.ordered,
            limit: scan_request.limit,
            has_filter: filter.is_some(),
            max_splits_in_flight: self.max_splits_in_flight,
        };
        let builder = ScanBuilder::new(self.session.clone(), Arc::clone(&self.reader))
            .with_row_range(row_range)
            .with_selection(scan_request.selection)
            .with_projection(projection)
            .with_some_filter(filter)
            .with_some_scan_metrics(self.metrics.clone())
            .with_ordered(scan_request.ordered);

        Ok(Box::new(LayoutScan {
            options,
            row_count,
            plan: ScanPlan::Unprepared(Box::new(builder)),
        }))
    }

    async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

/// Clamps a requested row range to the rows of the source.
pub(crate) fn clamp_row_range(row_range: Option<Range<u64>>, total_rows: u64) -> Range<u64> {
    let row_range = row_range.unwrap_or(0..total_rows);
    let end = row_range.end.min(total_rows);
    row_range.start.min(end)..end
}

/// A scan's plan: built lazily, then prepared once and shared by every piece cut from it.
enum ScanPlan {
    Unprepared(Box<ScanBuilder<ArrayRef>>),
    Prepared {
        scan: Arc<PreparedScan>,
        /// The scan's units: its layout splits, in row order.
        ranges: Vec<Range<u64>>,
    },
}

impl ScanPlan {
    /// Prepares the plan if needed, returning the shared prepared scan and its split ranges.
    fn prepare(self) -> VortexResult<PlannedSplits> {
        match self {
            ScanPlan::Unprepared(builder) => {
                let scan = Arc::new(builder.prepare()?);
                let ranges = scan.split_ranges(None);
                Ok((scan, ranges))
            }
            ScanPlan::Prepared { scan, ranges } => Ok((scan, ranges)),
        }
    }
}

/// A planned scan of one layout reader, whose units are the layout's splits.
struct LayoutScan {
    options: LayoutScanOptions,
    row_count: Precision<u64>,
    plan: ScanPlan,
}

/// What a [`LayoutScan`] and every piece cut from it share.
#[derive(Clone)]
struct LayoutScanOptions {
    dtype: DType,
    ordered: bool,
    limit: Option<u64>,
    has_filter: bool,
    max_splits_in_flight: NonZeroUsize,
}

impl LayoutScanOptions {
    /// A scan over some of the prepared scan's split `ranges`.
    fn piece(&self, scan: Arc<PreparedScan>, ranges: Vec<Range<u64>>) -> LayoutScan {
        let selected = ranges
            .iter()
            .map(|range| scan.selected_row_count(range))
            .sum::<u64>();
        LayoutScan {
            options: self.clone(),
            row_count: if self.has_filter {
                Precision::inexact(selected)
            } else {
                Precision::exact(selected)
            },
            plan: ScanPlan::Prepared { scan, ranges },
        }
    }
}

impl Scan for LayoutScan {
    fn dtype(&self) -> &DType {
        &self.options.dtype
    }

    fn ordered(&self) -> bool {
        self.options.ordered
    }

    fn row_count(&self) -> Precision<u64> {
        self.row_count
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    fn split(self: Box<Self>, n: NonZeroUsize) -> VortexResult<Vec<ScanRef>> {
        let LayoutScan {
            options,
            row_count,
            plan,
        } = *self;
        let (scan, ranges) = plan.prepare()?;
        if options.limit.is_some() || ranges.len() <= 1 || n.get() == 1 {
            return Ok(vec![Box::new(LayoutScan {
                options,
                row_count,
                plan: ScanPlan::Prepared { scan, ranges },
            })]);
        }
        Ok(group_contiguous(ranges, n)
            .into_iter()
            .map(|ranges| Box::new(options.piece(Arc::clone(&scan), ranges)) as ScanRef)
            .collect())
    }

    fn open(self: Box<Self>) -> VortexResult<ScanPoolRef> {
        let LayoutScanOptions {
            dtype,
            ordered,
            limit,
            has_filter,
            max_splits_in_flight,
        } = self.options;
        if limit == Some(0) {
            return Ok(Arc::new(ConstantPool::new(dtype, 0)));
        }
        let (scan, ranges) = self.plan.prepare()?;
        let work = FileWork::start(scan, ranges)?;
        let options = PoolOptions {
            dtype,
            ordered,
            limit,
            has_filter,
            max_splits_in_flight,
        };
        Ok(Arc::new(SplitPool::new(
            options,
            vec![(0, work)],
            Vec::new(),
            1,
            None,
        )))
    }
}

/// Groups consecutive `ranges` into at most `n` contiguous groups of about equal row counts.
///
/// Never cuts a range: each range lands whole in one group.
pub(crate) fn group_contiguous(ranges: Vec<Range<u64>>, n: NonZeroUsize) -> Vec<Vec<Range<u64>>> {
    let total = ranges
        .iter()
        .map(|range| u128::from(range.end - range.start))
        .sum::<u128>()
        .max(1);
    let n_groups = n.get() as u128;
    let mut groups: Vec<Vec<Range<u64>>> = Vec::new();
    let mut current_group = None;
    let mut before = 0u128;
    for range in ranges {
        let group = (before * n_groups / total).min(n_groups - 1);
        before += u128::from(range.end - range.start);
        if current_group != Some(group) {
            current_group = Some(group);
            groups.push(Vec::new());
        }
        if let Some(last) = groups.last_mut() {
            last.push(range);
        }
    }
    groups
}

/// A scan that reads no payloads: `row_count` rows of the dtype's default value, such as an
/// empty struct projection, or nothing at all for a scan that statistics pruned.
struct ConstantScan {
    dtype: DType,
    row_count: u64,
}

impl Scan for ConstantScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn ordered(&self) -> bool {
        true
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::exact(self.row_count)
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::exact(0u64)
    }

    fn split(self: Box<Self>, _n: NonZeroUsize) -> VortexResult<Vec<ScanRef>> {
        Ok(vec![self])
    }

    fn open(self: Box<Self>) -> VortexResult<ScanPoolRef> {
        Ok(Arc::new(ConstantPool::new(self.dtype, self.row_count)))
    }
}

/// The pool of a [`ConstantScan`]: its workers share one count of rows left to emit.
pub(crate) struct ConstantPool {
    dtype: DType,
    row_count: u64,
    remaining: Arc<AtomicU64>,
}

impl ConstantPool {
    pub(crate) fn new(dtype: DType, row_count: u64) -> Self {
        Self {
            dtype,
            row_count,
            remaining: Arc::new(AtomicU64::new(row_count)),
        }
    }
}

impl ScanPool for ConstantPool {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn worker(self: Arc<Self>) -> SendableArrayStream {
        let scalar = Scalar::default_value(&self.dtype);
        let remaining = Arc::clone(&self.remaining);
        let arrays = stream::iter(std::iter::from_fn(move || {
            let max_chunk = u64::try_from(usize::MAX).unwrap_or(u64::MAX);
            let taken = remaining
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |rows| {
                    (rows > 0).then(|| rows.saturating_sub(max_chunk))
                })
                .ok()?;
            let rows = usize::try_from(taken.min(max_chunk)).unwrap_or(usize::MAX);
            Some(VortexResult::Ok(
                ConstantArray::new(scalar.clone(), rows).into_array(),
            ))
        }));
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(self.dtype.clone(), arrays))
    }

    fn is_exhausted(&self) -> bool {
        self.remaining.load(Ordering::Relaxed) == 0
    }

    fn progress(&self) -> f64 {
        if self.row_count == 0 {
            return 1.0;
        }
        1.0 - self.remaining.load(Ordering::Relaxed) as f64 / self.row_count as f64
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::Expression;
    use vortex_array::expr::root;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_io::runtime::BlockingRuntime;
    use vortex_io::runtime::single::SingleThreadRuntime;
    use vortex_metrics::DefaultMetricsRegistry;
    use vortex_metrics::MetricValue;
    use vortex_metrics::MetricsRegistry;
    use vortex_scan::DataSource;
    use vortex_scan::ScanRef;
    use vortex_scan::ScanRequest;
    use vortex_scan::execute_with_workers;
    use vortex_scan::selection::Selection;
    use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;

    use super::LayoutReaderDataSource;
    use super::group_contiguous;
    use crate::scan::test::TestLayoutReader;
    use crate::scan::test::collect_scan_values;
    use crate::scan::test::keep_odd;
    use crate::scan::test::new_session;
    use crate::scan::test::read_with_one_worker;
    use crate::scan::test::read_with_workers;
    use crate::scan::test::session_with_handle;

    fn counter_values(registry: &DefaultMetricsRegistry) -> Vec<u64> {
        registry
            .snapshot()
            .iter()
            .map(|metric| match metric.value() {
                MetricValue::Counter(counter) => counter.value(),
                _ => panic!("expected a scan counter"),
            })
            .collect()
    }

    fn plan(source: &LayoutReaderDataSource, request: ScanRequest) -> VortexResult<ScanRef> {
        futures::executor::block_on(source.scan(request))
    }

    /// Earlier splits finish last, so completion order is the reverse of row order.
    fn slow_start(row: u64) -> usize {
        usize::try_from(32u64.saturating_sub(row)).unwrap_or(0)
    }

    fn source(reader: TestLayoutReader) -> LayoutReaderDataSource {
        LayoutReaderDataSource::new(Arc::new(reader), new_session())
    }

    #[rstest]
    #[case::file_statistics(true, true, None, true, [0, 0, 0, 1, 1])]
    #[case::range_statistics(false, true, None, true, [1, 1, 0, 1, 0])]
    #[case::range_predicate(false, false, None, true, [1, 0, 1, 1, 0])]
    #[case::zero_limit(true, true, Some(0), true, [0, 0, 0, 0, 0])]
    #[case::empty_selection(true, true, None, false, [0, 0, 0, 0, 0])]
    fn early_rejection_counts_once(
        #[case] whole_file: bool,
        #[case] statistics: bool,
        #[case] limit: Option<u64>,
        #[case] selected: bool,
        #[case] expected: [u64; 5],
    ) -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let registry = Arc::new(DefaultMetricsRegistry::default());
        let reader = TestLayoutReader::new(8).with_pruning_rejection(whole_file, statistics);
        let source = LayoutReaderDataSource::new(Arc::new(reader), session)
            .with_metrics_registry(Arc::<DefaultMetricsRegistry>::clone(&registry));
        for execution in 1..=2 {
            let scan = runtime.block_on(source.scan(ScanRequest {
                filter: Some(root()),
                limit,
                selection: if selected {
                    Selection::All
                } else {
                    Selection::IncludeByIndex(StrictSortedBuffer::try_new(Buffer::empty())?)
                },
                ..Default::default()
            }))?;
            let pool = scan.open()?;
            assert!(pool.is_exhausted());
            let values = collect_scan_values(runtime.block_on_stream(pool.worker()))?;
            assert!(values.is_empty());
            assert_eq!(
                counter_values(&registry),
                expected.map(|count| count * execution)
            );
        }
        Ok(())
    }

    /// Each opened scan is one execution: it counts one file scan, and one attempt per split.
    #[test]
    fn executions_share_aggregate_handles() -> VortexResult<()> {
        let registry = Arc::new(DefaultMetricsRegistry::default());
        let source = source(TestLayoutReader::new(8).with_split_size(2))
            .with_metrics_registry(Arc::<DefaultMetricsRegistry>::clone(&registry));
        for execution in 1..=2 {
            let pool = plan(&source, ScanRequest::default())?.open()?;
            let values = read_with_workers(&pool, 2)?.concat();
            assert_eq!(values.len(), 8);
            assert_eq!(
                counter_values(&registry),
                [4 * execution, 0, 0, execution, 0]
            );
        }
        Ok(())
    }

    /// Acceptance: N workers on one pool read every unit exactly once.
    #[rstest]
    #[case::ordered(true)]
    #[case::unordered(false)]
    fn workers_read_every_split_once(#[case] ordered: bool) -> VortexResult<()> {
        let source = source(TestLayoutReader::new(64).with_split_size(1))
            .with_max_splits_in_flight(NonZeroUsize::MIN);
        let pool = plan(
            &source,
            ScanRequest {
                ordered,
                ..Default::default()
            },
        )?
        .open()?;
        let per_worker = read_with_workers(&pool, 4)?;

        let mut values = per_worker.concat();
        values.sort_unstable();
        assert_eq!(values, (0..64).collect::<Vec<_>>());
        assert!(pool.is_exhausted());
        assert_eq!(pool.progress(), 1.0);
        Ok(())
    }

    /// Acceptance: a single worker of an ordered scan yields rows in row order, even when later
    /// splits finish first.
    #[test]
    fn single_ordered_worker_yields_row_order() -> VortexResult<()> {
        let read = |ordered| {
            let source = source(
                TestLayoutReader::new(32)
                    .with_split_size(1)
                    .with_yields(slow_start),
            );
            let pool = plan(
                &source,
                ScanRequest {
                    ordered,
                    ..Default::default()
                },
            )?
            .open()?;
            read_with_one_worker(pool)
        };

        assert_eq!(read(true)?, (0..32).collect::<Vec<_>>());
        // The same splits complete out of order, so an unordered worker does not keep row order.
        assert_ne!(read(false)?, (0..32).collect::<Vec<_>>());
        Ok(())
    }

    /// Acceptance: a limit holds across several workers of an unordered pool, whether splits
    /// reserve their rows up front (no filter) or trim their output (filter).
    #[rstest]
    #[case::reserved(None)]
    #[case::trimmed(Some(root()))]
    fn unordered_limit_holds_across_workers(
        #[case] filter: Option<Expression>,
    ) -> VortexResult<()> {
        let source = source(TestLayoutReader::new(64).with_split_size(2))
            .with_max_splits_in_flight(NonZeroUsize::MIN);
        let pool = plan(
            &source,
            ScanRequest {
                filter,
                limit: Some(7),
                ..Default::default()
            },
        )?
        .open()?;
        let mut values = read_with_workers(&pool, 4)?.concat();
        values.sort_unstable();
        values.dedup();
        assert_eq!(values.len(), 7);
        assert!(pool.is_exhausted());
        Ok(())
    }

    /// Acceptance: an ordered scan with a limit read by one worker returns exactly the first
    /// `limit` rows.
    #[rstest]
    #[case::reserved(false, &[0, 1, 2, 3, 4])]
    #[case::trimmed(true, &[1, 3, 5, 7, 9])]
    fn ordered_limit_returns_the_first_rows(
        #[case] filtered: bool,
        #[case] expected: &[i32],
    ) -> VortexResult<()> {
        let source = source(
            TestLayoutReader::new(32)
                .with_split_size(2)
                .with_keep_row(keep_odd)
                .with_yields(slow_start),
        );
        let pool = plan(
            &source,
            ScanRequest {
                filter: filtered.then(root),
                ordered: true,
                limit: Some(5),
                ..Default::default()
            },
        )?
        .open()?;
        assert_eq!(read_with_one_worker(pool)?, expected);
        Ok(())
    }

    /// With a sparse selection and no filter, each split contributes only its *selected* rows, so
    /// the workers keep taking splits until the limit is actually filled.
    #[rstest]
    #[case::ordered(true)]
    #[case::unordered(false)]
    fn sparse_selection_with_limit_fills_the_limit(#[case] ordered: bool) -> VortexResult<()> {
        let source = source(TestLayoutReader::new(12).with_split_size(2));

        // Every two-row split contains exactly one selected row (the even ones).
        let selection =
            Selection::IncludeByIndex(StrictSortedBuffer::try_new(Buffer::from_iter([
                0u64, 2, 4, 6, 8, 10,
            ]))?);
        let pool = plan(
            &source,
            ScanRequest {
                selection,
                limit: Some(3),
                ordered,
                ..Default::default()
            },
        )?
        .open()?;
        let mut values = read_with_one_worker(pool)?;
        if !ordered {
            values.sort_unstable();
        }
        assert_eq!(values, [0, 2, 4]);
        Ok(())
    }

    /// Acceptance: `split(n)` pieces cover the scan exactly, are in row order, and never cut a
    /// layout split.
    #[rstest]
    #[case::fewer_pieces_than_splits(3, 3)]
    #[case::more_pieces_than_splits(16, 7)]
    fn split_pieces_cover_the_scan_on_split_boundaries(
        #[case] n: usize,
        #[case] expected_pieces: usize,
    ) -> VortexResult<()> {
        // Splits of three rows: [0, 3), [3, 6), ..., [18, 20).
        let source = source(TestLayoutReader::new(20).with_split_size(3));
        let scan = plan(&source, ScanRequest::default())?;
        let pieces = scan.split(NonZeroUsize::new(n).expect("non-zero"))?;
        assert_eq!(pieces.len(), expected_pieces);
        assert_eq!(
            pieces
                .iter()
                .map(|piece| piece.row_count().as_exact())
                .sum::<Option<u64>>(),
            Some(20)
        );

        let mut values = Vec::new();
        for piece in pieces {
            let piece_values = read_with_one_worker(piece.open()?)?;
            let first = *piece_values.first().expect("pieces are not empty");
            let last = *piece_values.last().expect("pieces are not empty");
            assert_eq!(first % 3, 0, "piece starts on a split boundary");
            assert!(
                (last + 1) % 3 == 0 || last == 19,
                "piece ends on a split boundary"
            );
            values.extend(piece_values);
        }
        assert_eq!(values, (0..20).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn split_keeps_a_limited_scan_whole() -> VortexResult<()> {
        let source = source(TestLayoutReader::new(20).with_split_size(3));
        let scan = plan(
            &source,
            ScanRequest {
                limit: Some(4),
                ..Default::default()
            },
        )?;
        let pieces = scan.split(NonZeroUsize::new(4).expect("non-zero"))?;
        assert_eq!(pieces.len(), 1);
        Ok(())
    }

    #[test]
    fn grouping_never_returns_more_groups_than_requested() {
        let ranges = (0..10u64)
            .map(|start| start * 10..start * 10 + 10)
            .collect();
        let groups = group_contiguous(ranges, NonZeroUsize::new(3).expect("non-zero"));
        assert_eq!(groups.len(), 3);
        assert_eq!(groups.concat().len(), 10);
    }

    /// Acceptance: `execute_with_workers` on an ordered scan returns rows in order.
    #[rstest]
    #[case::unlimited(None, 64)]
    #[case::limited(Some(10), 10)]
    fn execute_with_workers_keeps_row_order(
        #[case] limit: Option<u64>,
        #[case] rows: i32,
    ) -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let source = LayoutReaderDataSource::new(
            Arc::new(
                TestLayoutReader::new(64)
                    .with_split_size(2)
                    .with_yields(slow_start),
            ),
            session,
        );
        let scan = runtime.block_on(source.scan(ScanRequest {
            ordered: true,
            limit,
            ..Default::default()
        }))?;
        let stream = execute_with_workers(
            scan,
            NonZeroUsize::new(4).expect("non-zero"),
            &runtime.handle(),
        )?;
        let values = collect_scan_values(runtime.block_on_stream(stream))?;
        assert_eq!(values, (0..rows).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn execute_with_workers_reads_an_unordered_scan_once() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let source = LayoutReaderDataSource::new(
            Arc::new(TestLayoutReader::new(64).with_split_size(2)),
            session,
        );
        let scan = runtime.block_on(source.scan(ScanRequest::default()))?;
        let stream = execute_with_workers(
            scan,
            NonZeroUsize::new(4).expect("non-zero"),
            &runtime.handle(),
        )?;
        let mut values = collect_scan_values(runtime.block_on_stream(stream))?;
        values.sort_unstable();
        assert_eq!(values, (0..64).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn length_only_scan_shares_its_count_across_workers() -> VortexResult<()> {
        let source = source(TestLayoutReader::new(10).with_dtype(DType::Struct(
            StructFields::empty(),
            Nullability::NonNullable,
        )));
        let pool = plan(
            &source,
            ScanRequest {
                limit: Some(7),
                ..Default::default()
            },
        )?
        .open()?;
        let rows: usize = std::thread::scope(|scope| {
            let handles = (0..3)
                .map(|_| {
                    let worker = Arc::clone(&pool).worker();
                    scope.spawn(move || {
                        futures::executor::block_on_stream(worker)
                            .map(|array| array.map(|array| array.len()))
                            .sum::<VortexResult<usize>>()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("worker thread panicked"))
                .sum::<VortexResult<usize>>()
        })?;
        assert_eq!(rows, 7);
        Ok(())
    }
}
