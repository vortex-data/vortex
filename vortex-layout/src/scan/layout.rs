// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use async_trait::async_trait;
use futures::FutureExt;
use futures::Stream;
use futures::stream;
use futures::stream::StreamExt;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::stats::Precision;
use vortex_array::scalar::Scalar;
use vortex_array::stats::StatsSet;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_metrics::MetricsRegistry;
use vortex_scan::DataSource;
use vortex_scan::DataSourceScan;
use vortex_scan::DataSourceScanRef;
use vortex_scan::Partition;
use vortex_scan::PartitionRef;
use vortex_scan::PartitionStream;
use vortex_scan::ScanRequest;
use vortex_scan::pool::ScanPoolRef;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use crate::LayoutReaderRef;
use crate::scan::limit::RowLimit;
use crate::scan::metrics::ScanMetrics;
use crate::scan::metrics::ScanRecorder;
use crate::scan::pool::FileWork;
use crate::scan::pool::PiecePartition;
use crate::scan::pool::PoolFile;
use crate::scan::pool::PoolOptions;
use crate::scan::pool::PreparedScan;
use crate::scan::pool::open_pool;
use crate::scan::scan_builder::ScanBuilder;

/// An implementation of a [`DataSource`] that reads data from a [`LayoutReaderRef`].
///
/// When a scan is [split](DataSourceScan::split) or [opened](DataSourceScan::open), its units are
/// the layout's splits. Workers of an opened scan share them, each keeping several in flight on
/// its own thread.
pub struct LayoutReaderDataSource {
    reader: LayoutReaderRef,
    session: VortexSession,
    split_max_row_count: u64,
    metrics: Option<ScanMetrics>,
}

impl LayoutReaderDataSource {
    /// Creates a new [`LayoutReaderDataSource`].
    ///
    /// By default, the entire scan is returned as a single split. This best preserves V1
    /// `ScanBuilder` behavior where one scan covers the full row range, allowing the internal
    /// I/O pipeline and `SplitBy::Layout` chunking to operate without per-split overhead from
    /// redundant expression resolution and layout tree traversal.
    pub fn new(reader: LayoutReaderRef, session: VortexSession) -> Self {
        Self {
            reader,
            session,
            split_max_row_count: u64::MAX,
            metrics: None,
        }
    }

    /// Sets the maximum number of rows per Scan API split.
    ///
    /// Each split drives a [`ScanBuilder`] over its row range, which internally handles
    /// physical layout alignment and I/O pipelining. This controls the engine-level
    /// parallelism granularity, not the I/O granularity.
    pub fn with_split_max_row_count(mut self, row_count: u64) -> Self {
        self.split_max_row_count = row_count;
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

    fn deserialize_partition(
        &self,
        _data: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<PartitionRef> {
        vortex_bail!("LayoutReader splits are not yet serializable");
    }

    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        let total_rows = self.reader.row_count();
        let row_range = scan_request.row_range.unwrap_or(0..total_rows);

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
            // FIXME(ngates): extract out maybe?
            let row_count = scan_request.selection.row_count_in_range(&row_range);

            // Apply the limit.
            let row_count = if let Some(limit) = scan_request.limit {
                row_count.min(limit)
            } else {
                row_count
            };

            return Ok(Box::new(Empty { dtype, row_count }));
        }

        // If statistics reject the entire requested range, skip the scan before partition construction.
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

                return Ok(Box::new(Empty {
                    dtype,
                    row_count: 0,
                }));
            }
        }

        // Only unordered scans share a limit across external partitions: reservation order is
        // completion order, which an ordered scan cannot accept. Ordered partitions each apply
        // the limit locally and the engine trims the concatenated result.
        let row_limit = (!scan_request.ordered)
            .then(|| scan_request.limit.map(RowLimit::new))
            .flatten();

        Ok(Box::new(LayoutReaderScan {
            reader: Arc::clone(&self.reader),
            session: self.session.clone(),
            dtype,
            projection,
            filter,
            limit: scan_request.limit,
            remaining_limit: scan_request.limit,
            row_limit,
            selection: scan_request.selection,
            ordered: scan_request.ordered,
            split_concurrency: scan_request.split_concurrency,
            metrics: self.metrics.clone(),
            next_row: row_range.start,
            end_row: row_range.end,
            split_size: self.split_max_row_count,
        }))
    }

    async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

struct LayoutReaderScan {
    reader: LayoutReaderRef,
    session: VortexSession,
    dtype: DType,
    projection: BoundExpression,
    filter: Option<BoundExpression>,
    /// The request's limit.
    limit: Option<u64>,
    /// Limit rows not yet covered by the splits produced so far; production stops at zero.
    /// Decremented only for filterless scans, where each split's output is exactly its
    /// selected row count.
    remaining_limit: Option<u64>,
    /// A row budget shared by every split of an unordered limited scan; see [`RowLimit`].
    row_limit: Option<RowLimit>,
    ordered: bool,
    split_concurrency: Option<NonZeroUsize>,
    selection: Selection,
    metrics: Option<ScanMetrics>,
    next_row: u64,
    end_row: u64,
    split_size: u64,
}

impl LayoutReaderScan {
    /// Prepares the scan once, for splitting or opening.
    fn into_split_scan(self) -> VortexResult<LayoutSplitScan> {
        let row_count = DataSourceScan::row_count(&self);
        let options = LayoutScanOptions {
            dtype: self.dtype,
            ordered: self.ordered,
            limit: self.limit,
            has_filter: self.filter.is_some(),
            splits_in_flight: self.split_concurrency,
        };
        let row_range = clamp_row_range(self.next_row..self.end_row, self.reader.row_count());
        let scan = ScanBuilder::new(self.session, self.reader)
            .with_row_range(row_range)
            .with_selection(self.selection)
            .with_projection(self.projection)
            .with_some_filter(self.filter)
            .with_some_scan_metrics(self.metrics)
            .with_ordered(self.ordered)
            .prepare()?;
        let ranges = scan.split_ranges(None);
        Ok(LayoutSplitScan {
            options,
            row_count,
            scan: Arc::new(scan),
            ranges,
        })
    }
}

impl DataSourceScan for LayoutReaderScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        let (lower, upper) = self.size_hint();
        match upper {
            Some(u) if u == lower => Precision::exact(lower),
            Some(u) => Precision::inexact(u),
            None => Precision::inexact(lower),
        }
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        (*self).boxed()
    }

    fn row_count(&self) -> Precision<u64> {
        let selected = self
            .selection
            .row_count_in_range(&(self.next_row..self.end_row));
        let rows = self.limit.map_or(selected, |limit| selected.min(limit));
        if self.filter.is_some() {
            Precision::inexact(rows)
        } else {
            Precision::exact(rows)
        }
    }

    fn split(self: Box<Self>, n: NonZeroUsize) -> VortexResult<Vec<DataSourceScanRef>> {
        Box::new(self.into_split_scan()?).split(n)
    }

    fn open(self: Box<Self>) -> VortexResult<ScanPoolRef> {
        Box::new(self.into_split_scan()?).open()
    }
}

/// Clamps a requested row range to the rows of the source.
fn clamp_row_range(row_range: Range<u64>, total_rows: u64) -> Range<u64> {
    let end = row_range.end.min(total_rows);
    row_range.start.min(end)..end
}

/// What a [`LayoutSplitScan`] and every piece cut from it share.
#[derive(Clone)]
struct LayoutScanOptions {
    dtype: DType,
    ordered: bool,
    limit: Option<u64>,
    has_filter: bool,
    splits_in_flight: Option<NonZeroUsize>,
}

/// A prepared scan of one layout reader over some of its splits: a [`LayoutReaderScan`] that was
/// split or opened, or a piece cut from one.
struct LayoutSplitScan {
    options: LayoutScanOptions,
    row_count: Precision<u64>,
    scan: Arc<PreparedScan>,
    /// The scan's units: its layout splits, in row order.
    ranges: Vec<Range<u64>>,
}

impl LayoutSplitScan {
    /// A piece over some of the prepared scan's split `ranges`.
    fn piece(
        options: &LayoutScanOptions,
        scan: Arc<PreparedScan>,
        ranges: Vec<Range<u64>>,
    ) -> Self {
        let selected = ranges
            .iter()
            .map(|range| scan.selected_row_count(range))
            .sum::<u64>();
        Self {
            options: options.clone(),
            row_count: if options.has_filter {
                Precision::inexact(selected)
            } else {
                Precision::exact(selected)
            },
            scan,
            ranges,
        }
    }
}

impl DataSourceScan for LayoutSplitScan {
    fn dtype(&self) -> &DType {
        &self.options.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        Precision::exact(1usize)
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        let index = self.ranges.first().map_or(0, |range| {
            usize::try_from(range.start).unwrap_or(usize::MAX)
        });
        PiecePartition::stream(self, index)
    }

    fn row_count(&self) -> Precision<u64> {
        self.row_count
    }

    fn split(self: Box<Self>, n: NonZeroUsize) -> VortexResult<Vec<DataSourceScanRef>> {
        if self.options.limit.is_some() || self.ranges.len() <= 1 || n.get() == 1 {
            return Ok(vec![self]);
        }
        let LayoutSplitScan {
            options,
            scan,
            ranges,
            ..
        } = *self;
        Ok(group_contiguous(ranges, n)
            .into_iter()
            .map(|ranges| {
                Box::new(LayoutSplitScan::piece(&options, Arc::clone(&scan), ranges))
                    as DataSourceScanRef
            })
            .collect())
    }

    fn open(self: Box<Self>) -> VortexResult<ScanPoolRef> {
        let LayoutScanOptions {
            dtype,
            ordered,
            limit,
            has_filter,
            splits_in_flight,
        } = self.options;
        if limit == Some(0) {
            return Box::new(Empty::new(dtype, 0)).open();
        }
        let work = FileWork::start(self.scan, self.ranges)?;
        let options = PoolOptions {
            dtype,
            ordered,
            limit,
            has_filter,
            splits_in_flight,
        };
        Ok(open_pool(options, vec![PoolFile::Opened(work)], None))
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

impl Stream for LayoutReaderScan {
    type Item = VortexResult<PartitionRef>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if this.next_row >= this.end_row {
            return Poll::Ready(None);
        }

        if this.remaining_limit.is_some_and(|limit| limit == 0) {
            return Poll::Ready(None);
        }
        if this.row_limit.as_ref().is_some_and(RowLimit::is_exhausted) {
            return Poll::Ready(None);
        }

        let split_end = this
            .next_row
            .saturating_add(this.split_size)
            .min(this.end_row);
        let row_range = this.next_row..split_end;

        // Snapshot before decrementing: a split's cap must cover the rows it will itself return.
        let split_limit = this.remaining_limit;
        // Without a filter, a split returns exactly its selected rows, so the countdown stays
        // exact. With a filter the split's contribution is unknowable up front, so every split
        // receives the full remaining limit and the total is enforced by the shared `row_limit`
        // (unordered scans) or by the engine trimming the concatenated result (ordered scans).
        if this.filter.is_none()
            && let Some(ref mut remaining) = this.remaining_limit
        {
            *remaining = remaining.saturating_sub(this.selection.row_count_in_range(&row_range));
        }

        let split = Box::new(LayoutReaderSplit {
            reader: Arc::clone(&this.reader),
            session: this.session.clone(),
            projection: this.projection.clone(),
            filter: this.filter.clone(),
            limit: split_limit,
            row_limit: this.row_limit.clone(),
            ordered: this.ordered,
            split_concurrency: this.split_concurrency,
            row_range,
            selection: this.selection.clone(),
            metrics: this.metrics.clone(),
        }) as PartitionRef;

        this.next_row = split_end;

        Poll::Ready(Some(Ok(split)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.next_row >= self.end_row
            || self.remaining_limit.is_some_and(|limit| limit == 0)
            || self.row_limit.as_ref().is_some_and(RowLimit::is_exhausted)
        {
            return (0, Some(0));
        }
        let remaining_rows = self.end_row - self.next_row;
        let splits = remaining_rows.div_ceil(self.split_size);
        (0, Some(usize::try_from(splits).unwrap_or(usize::MAX)))
    }
}

struct LayoutReaderSplit {
    reader: LayoutReaderRef,
    session: VortexSession,
    projection: BoundExpression,
    filter: Option<BoundExpression>,
    /// Cap on this split's output rows: the scan's remaining limit when the split was produced.
    limit: Option<u64>,
    /// A row budget shared with the scan's other splits (unordered limited scans only).
    row_limit: Option<RowLimit>,
    ordered: bool,
    split_concurrency: Option<NonZeroUsize>,
    row_range: Range<u64>,
    selection: Selection,
    metrics: Option<ScanMetrics>,
}

impl Partition for LayoutReaderSplit {
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[expect(clippy::cast_possible_truncation)]
    fn index(&self) -> usize {
        // Row range is unique per split
        self.row_range.start as usize
    }

    fn row_count(&self) -> Precision<u64> {
        let row_count = self.selection.row_count_in_range(&self.row_range);
        let row_count = self.limit.map_or(row_count, |limit| row_count.min(limit));

        if self.filter.is_some() || self.row_limit.is_some() {
            Precision::inexact(row_count)
        } else {
            Precision::exact(row_count)
        }
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let builder = ScanBuilder::new(self.session, self.reader)
            .with_row_range(self.row_range)
            .with_selection(self.selection)
            .with_projection(self.projection)
            .with_some_filter(self.filter)
            .with_some_scan_metrics(self.metrics)
            .with_ordered(self.ordered)
            .with_some_max_splits_in_flight(self.split_concurrency);
        let builder = match self.row_limit {
            Some(row_limit) => builder.with_shared_limit(row_limit),
            None => builder.with_some_limit(self.limit),
        };

        let dtype = builder.dtype()?;
        // Use into_stream() which creates a LazyScanStream that spawns individual I/O
        // tasks onto the runtime, enabling parallel execution across executor threads.
        let stream = builder.into_stream()?.boxed();

        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype, stream,
        )))
    }
}

/// A scan that produces no data, only empty arrays with the correct row count.
pub(crate) struct Empty {
    dtype: DType,
    row_count: u64,
}

impl Empty {
    pub(crate) fn new(dtype: DType, row_count: u64) -> Self {
        Self { dtype, row_count }
    }
}

impl DataSourceScan for Empty {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        Precision::exact(1usize)
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        stream::iter([Ok(self as _)]).boxed()
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::exact(self.row_count)
    }
}

impl Partition for Empty {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        0
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::exact(self.row_count)
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::exact(0u64)
    }

    fn execute(mut self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let scalar = Scalar::default_value(&self.dtype);
        let dtype = self.dtype.clone();

        // Create an iterator of arrays with the correct row count, respecting u64::MAX limits.
        let iter = std::iter::from_fn(move || {
            if self.row_count == 0 {
                return None;
            }
            let chunk_size = usize::try_from(self.row_count).unwrap_or(usize::MAX);
            self.row_count -= chunk_size as u64;
            Some(VortexResult::Ok(
                ConstantArray::new(scalar.clone(), chunk_size).into_array(),
            ))
        });

        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype,
            stream::iter(iter),
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::StreamExt;
    use futures::TryStreamExt;
    use rstest::rstest;
    use vortex_array::expr::root;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_io::runtime::BlockingRuntime;
    use vortex_io::runtime::single::SingleThreadRuntime;
    use vortex_metrics::DefaultMetricsRegistry;
    use vortex_metrics::MetricValue;
    use vortex_metrics::MetricsRegistry;
    use vortex_scan::DataSource;
    use vortex_scan::ScanRequest;
    use vortex_scan::selection::Selection;
    use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;

    use super::LayoutReaderDataSource;
    use crate::scan::test::TestLayoutReader;
    use crate::scan::test::collect_scan_values;
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
            let partitions = runtime.block_on(scan.partitions().try_collect::<Vec<_>>())?;
            for partition in partitions {
                let arrays = runtime.block_on(partition.execute()?.try_collect::<Vec<_>>())?;
                assert!(arrays.is_empty());
            }
            assert_eq!(
                counter_values(&registry),
                expected.map(|count| count * execution)
            );
        }
        Ok(())
    }

    #[test]
    fn partitions_and_scans_share_aggregate_handles() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let registry = Arc::new(DefaultMetricsRegistry::default());
        let source = LayoutReaderDataSource::new(Arc::new(TestLayoutReader::new(8)), session)
            .with_split_max_row_count(2)
            .with_metrics_registry(Arc::<DefaultMetricsRegistry>::clone(&registry));
        for execution in 1..=2 {
            let scan = runtime.block_on(source.scan(ScanRequest::default()))?;
            let partitions = runtime.block_on(scan.partitions().try_collect::<Vec<_>>())?;
            assert_eq!(partitions.len(), 4);
            for partition in partitions {
                let arrays = runtime.block_on(partition.execute()?.try_collect::<Vec<_>>())?;
                assert_eq!(arrays.iter().map(|array| array.len()).sum::<usize>(), 2);
            }
            assert_eq!(
                counter_values(&registry),
                [4 * execution, 0, 0, 4 * execution, 0]
            );
        }
        Ok(())
    }

    /// An unordered limit is shared by every partition of the scan, so the partitions together
    /// return no more rows than the limit.
    #[test]
    fn unordered_limit_is_shared_across_partitions() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let source = LayoutReaderDataSource::new(Arc::new(TestLayoutReader::new(12)), session)
            .with_split_max_row_count(2);

        let scan = runtime.block_on(source.scan(ScanRequest {
            filter: Some(root()),
            limit: Some(3),
            ordered: false,
            ..Default::default()
        }))?;
        let partitions = runtime.block_on(scan.partitions().try_collect::<Vec<_>>())?;
        assert_eq!(partitions.len(), 6);

        let chunks = runtime.block_on(
            futures::stream::iter(partitions)
                .map(|partition| partition.execute())
                .try_flatten_unordered(Some(6))
                .try_collect::<Vec<_>>(),
        )?;
        let values = collect_scan_values(chunks.into_iter().map(Ok))?;

        assert_eq!(values.len(), 3);
        Ok(())
    }

    /// With a sparse selection and no filter, each split contributes only its *selected* rows, so
    /// the scan must keep producing splits until the limit is actually filled.
    #[rstest]
    #[case::ordered(true)]
    #[case::unordered(false)]
    fn sparse_selection_with_limit_fills_the_limit(#[case] ordered: bool) -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let source = LayoutReaderDataSource::new(Arc::new(TestLayoutReader::new(12)), session)
            .with_split_max_row_count(2);

        // Every two-row split contains exactly one selected row (the even ones).
        let selection =
            Selection::IncludeByIndex(StrictSortedBuffer::try_new(Buffer::from_iter([
                0u64, 2, 4, 6, 8, 10,
            ]))?);
        let scan = runtime.block_on(source.scan(ScanRequest {
            selection,
            limit: Some(3),
            ordered,
            ..Default::default()
        }))?;
        let partitions = runtime.block_on(scan.partitions().try_collect::<Vec<_>>())?;

        let mut values = Vec::new();
        for partition in partitions {
            values.extend(collect_scan_values(
                runtime.block_on_stream(partition.execute()?),
            )?);
        }

        assert_eq!(values, [0, 2, 4]);
        Ok(())
    }
}

#[cfg(test)]
mod pool_tests {
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
    use vortex_metrics::DefaultMetricsRegistry;
    use vortex_metrics::MetricValue;
    use vortex_metrics::MetricsRegistry;
    use vortex_scan::DataSource;
    use vortex_scan::DataSourceScanRef;
    use vortex_scan::ScanRequest;
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

    fn plan(
        source: &LayoutReaderDataSource,
        request: ScanRequest,
    ) -> VortexResult<DataSourceScanRef> {
        futures::executor::block_on(source.scan(request))
    }

    /// Earlier splits finish last, so completion order is the reverse of row order.
    fn slow_start(row: u64) -> usize {
        usize::try_from(32u64.saturating_sub(row)).unwrap_or(0)
    }

    fn source(reader: TestLayoutReader) -> LayoutReaderDataSource {
        LayoutReaderDataSource::new(Arc::new(reader), new_session())
    }

    /// Each opened scan is one execution: it counts one file scan, and one attempt per split.
    #[test]
    fn opened_executions_share_aggregate_handles() -> VortexResult<()> {
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

    /// N workers on one pool read every split exactly once.
    #[rstest]
    #[case::ordered(true)]
    #[case::unordered(false)]
    fn workers_read_every_split_once(#[case] ordered: bool) -> VortexResult<()> {
        let source = source(TestLayoutReader::new(64).with_split_size(1));
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
        Ok(())
    }

    /// A worker of an ordered pool yields rows in row order, even when later splits finish
    /// first, and later workers of the pool get nothing.
    #[test]
    fn ordered_worker_yields_row_order() -> VortexResult<()> {
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
            let first = read_with_one_worker(Arc::clone(&pool))?;
            let second = read_with_one_worker(pool)?;
            assert!(second.is_empty());
            Ok::<_, vortex_error::VortexError>(first)
        };

        assert_eq!(read(true)?, (0..32).collect::<Vec<_>>());
        // The same splits complete out of order, so an unordered worker does not keep row order.
        assert_ne!(read(false)?, (0..32).collect::<Vec<_>>());
        Ok(())
    }

    /// A limit holds across several workers of an unordered pool, whether splits reserve their
    /// rows up front (no filter) or trim their output (filter).
    #[rstest]
    #[case::reserved(None)]
    #[case::trimmed(Some(root()))]
    fn unordered_limit_holds_across_workers(
        #[case] filter: Option<Expression>,
    ) -> VortexResult<()> {
        let source = source(TestLayoutReader::new(64).with_split_size(2));
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
        Ok(())
    }

    /// An ordered pool with a limit returns exactly the first `limit` rows.
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

    /// With a sparse selection and no filter, each split contributes only its selected rows, so
    /// the workers keep taking splits until the limit is actually filled.
    #[rstest]
    #[case::ordered(true)]
    #[case::unordered(false)]
    fn opened_sparse_selection_with_limit_fills_the_limit(
        #[case] ordered: bool,
    ) -> VortexResult<()> {
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

    /// `split(n)` pieces cover the scan exactly, are in row order, and never cut a layout split.
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
            // The scan is unordered, so a piece's splits may finish in any order.
            let mut piece_values = read_with_one_worker(piece.open()?)?;
            piece_values.sort_unstable();
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

    /// A piece read through the partitions API returns the same rows as its pool.
    #[test]
    fn split_pieces_read_through_partitions() -> VortexResult<()> {
        let source = source(TestLayoutReader::new(20).with_split_size(3));
        let scan = plan(
            &source,
            ScanRequest {
                ordered: true,
                ..Default::default()
            },
        )?;
        let mut values = Vec::new();
        for piece in scan.split(NonZeroUsize::new(3).expect("non-zero"))? {
            let partitions =
                futures::executor::block_on_stream(piece.partitions()).collect::<Vec<_>>();
            assert_eq!(partitions.len(), 1);
            for partition in partitions {
                values.extend(collect_scan_values(futures::executor::block_on_stream(
                    partition?.execute()?,
                ))?);
            }
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

    /// A length-only scan opens into one pool whose workers together return its rows once.
    #[test]
    fn length_only_scan_returns_its_rows_once_across_workers() -> VortexResult<()> {
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
