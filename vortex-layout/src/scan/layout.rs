// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::num::NonZeroU64;
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
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_metrics::MetricsRegistry;
use vortex_scan::DataSource;
use vortex_scan::DataSourceScan;
use vortex_scan::DataSourceScanRef;
use vortex_scan::Partition;
use vortex_scan::PartitionRef;
use vortex_scan::PartitionStream;
use vortex_scan::ScanRequest;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use crate::LayoutReaderRef;
use crate::scan::limit::RowLimit;
use crate::scan::metrics::ScanMetrics;
use crate::scan::metrics::ScanRecorder;
use crate::scan::scan_builder::ScanBuilder;

/// Cursor over absolute split-grid partition indices for a [`LayoutReaderDataSource`].
///
/// Each split occupies a fixed `[p * split_size, (p + 1) * split_size)` slot on the source so
/// `Partition::index()` remains stable across `row_range`, `partition_range`, and
/// `partition_selection` requests.
struct PartitionCursor {
    row_range: Range<u64>,
    split_size: NonZeroU64,
    next_partition: u64,
    end_partition: u64,
    partition_selection: Selection,
}

impl PartitionCursor {
    fn new(
        row_range: Range<u64>,
        split_size: NonZeroU64,
        partition_range: Option<Range<u64>>,
        partition_selection: Selection,
    ) -> Self {
        let min_part = row_range.start / split_size;
        let max_part = if row_range.start >= row_range.end {
            min_part
        } else {
            row_range.end.div_ceil(split_size.get())
        };
        let start = partition_range
            .as_ref()
            .map_or(min_part, |range| min_part.max(range.start));
        let end = partition_range
            .as_ref()
            .map_or(max_part, |range| max_part.min(range.end));
        Self {
            row_range,
            split_size,
            next_partition: start.min(end),
            end_partition: end,
            partition_selection,
        }
    }

    fn empty() -> Self {
        Self {
            row_range: 0..0,
            split_size: NonZeroU64::MIN,
            next_partition: 0,
            end_partition: 0,
            partition_selection: Selection::All,
        }
    }

    /// Returns the next selected partition index and its clamped row range.
    ///
    /// Skips unselected partition indices linearly over `[next_partition, end_partition)`, which is
    /// bounded by the number of splits in the scan (1 by default, or `rows / split_size` when
    /// `with_split_max_row_count` is set).
    fn next_selected(&mut self) -> Option<(usize, Range<u64>)> {
        let mut partition = self.next_partition;
        while partition < self.end_partition && !self.partition_selection.contains(partition) {
            partition = partition.saturating_add(1);
        }
        if partition >= self.end_partition {
            self.next_partition = self.end_partition;
            return None;
        }

        self.next_partition = partition.saturating_add(1);
        let split_size = self.split_size.get();
        let split_start = partition
            .saturating_mul(split_size)
            .max(self.row_range.start);
        let split_end = partition
            .saturating_add(1)
            .saturating_mul(split_size)
            .min(self.row_range.end);
        let partition_idx = usize::try_from(partition).unwrap_or(usize::MAX);
        Some((partition_idx, split_start..split_end))
    }

    fn remaining_count(&self) -> u64 {
        if self.next_partition >= self.end_partition {
            return 0;
        }
        self.partition_selection
            .row_count_in_range(&(self.next_partition..self.end_partition))
    }
}

/// An implementation of a [`DataSource`] that reads data from a [`LayoutReaderRef`].
///
/// Partitions are assigned on a fixed split grid starting at row 0 of the file: split `i` covers
/// rows `[i * split_max_row_count, (i + 1) * split_max_row_count)` (clamped to the scan's
/// `row_range` and the file's row count).
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
        let split_size = NonZeroU64::new(self.split_max_row_count)
            .ok_or_else(|| vortex_err!("split_max_row_count must be greater than zero"))?;
        let total_rows = self.reader.row_count();
        let row_range = match scan_request.row_range {
            Some(range) => {
                let start = range.start.min(total_rows);
                start..range.end.clamp(start, total_rows)
            }
            None => 0..total_rows,
        };

        let projection = scan_request
            .projection
            .bind(self.reader.dtype())?
            .optimize_recursive()?;
        let filter = scan_request
            .filter
            .map(|expr| expr.bind(self.reader.dtype())?.optimize_recursive())
            .transpose()?;
        let dtype = projection.dtype().clone();

        let cursor = PartitionCursor::new(
            row_range.clone(),
            split_size,
            scan_request.partition_range,
            scan_request.partition_selection,
        );

        // If the dtype is an empty struct, and there is no filter, we can return a special
        // length-only scan that preserves the same split-grid partition identities.
        if let DType::Struct(fields, Nullability::NonNullable) = &dtype
            && fields.nfields() == 0
            && filter.is_none()
        {
            return Ok(Box::new(EmptyScan {
                dtype,
                cursor,
                selection: scan_request.selection,
                remaining_limit: scan_request.limit,
            }));
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

                return Ok(Box::new(EmptyScan::empty(dtype)));
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
            remaining_limit: scan_request.limit,
            row_limit,
            selection: scan_request.selection,
            ordered: scan_request.ordered,
            split_concurrency: scan_request.split_concurrency,
            metrics: self.metrics.clone(),
            cursor,
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
    /// Limit rows not yet covered by the splits produced so far in this request; production stops
    /// at zero. Decremented only for filterless scans, where each split's output is exactly its
    /// selected row count. Applies per scan request, so replaying a single partition applies that
    /// request's own limit.
    remaining_limit: Option<u64>,
    /// A row budget shared by every split of an unordered limited scan; see [`RowLimit`].
    row_limit: Option<RowLimit>,
    ordered: bool,
    split_concurrency: Option<NonZeroUsize>,
    selection: Selection,
    metrics: Option<ScanMetrics>,
    cursor: PartitionCursor,
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
}

impl Stream for LayoutReaderScan {
    type Item = VortexResult<PartitionRef>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if this.remaining_limit.is_some_and(|limit| limit == 0) {
            return Poll::Ready(None);
        }
        if this.row_limit.as_ref().is_some_and(RowLimit::is_exhausted) {
            return Poll::Ready(None);
        }

        let Some((partition_idx, row_range)) = this.cursor.next_selected() else {
            return Poll::Ready(None);
        };

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
            partition_idx,
        }) as PartitionRef;

        Poll::Ready(Some(Ok(split)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.remaining_limit.is_some_and(|limit| limit == 0)
            || self.row_limit.as_ref().is_some_and(RowLimit::is_exhausted)
        {
            return (0, Some(0));
        }
        let remaining = self.cursor.remaining_count();
        let upper = usize::try_from(remaining).ok();
        if self.remaining_limit.is_none() && self.row_limit.is_none() {
            match upper {
                Some(exact) => (exact, Some(exact)),
                None => (usize::MAX, None),
            }
        } else {
            (0, upper)
        }
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
    partition_idx: usize,
}

impl Partition for LayoutReaderSplit {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        self.partition_idx
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
struct Empty {
    dtype: DType,
    row_count: u64,
    partition_idx: usize,
}

struct EmptyScan {
    dtype: DType,
    cursor: PartitionCursor,
    selection: Selection,
    remaining_limit: Option<u64>,
}

impl EmptyScan {
    fn empty(dtype: DType) -> Self {
        Self {
            dtype,
            cursor: PartitionCursor::empty(),
            selection: Selection::All,
            remaining_limit: Some(0),
        }
    }
}

impl DataSourceScan for EmptyScan {
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
}

impl Stream for EmptyScan {
    type Item = VortexResult<PartitionRef>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if this.remaining_limit.is_some_and(|limit| limit == 0) {
            return Poll::Ready(None);
        }

        let Some((partition_idx, row_range)) = this.cursor.next_selected() else {
            return Poll::Ready(None);
        };

        let mut count = this.selection.row_count_in_range(&row_range);
        if let Some(ref mut remaining) = this.remaining_limit {
            count = count.min(*remaining);
            *remaining -= count;
        }

        Poll::Ready(Some(Ok(Box::new(Empty {
            dtype: this.dtype.clone(),
            row_count: count,
            partition_idx,
        }) as PartitionRef)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.remaining_limit.is_some_and(|limit| limit == 0) {
            return (0, Some(0));
        }
        let remaining = self.cursor.remaining_count();
        let upper = usize::try_from(remaining).ok();
        if self.remaining_limit.is_none() {
            match upper {
                Some(exact) => (exact, Some(exact)),
                None => (usize::MAX, None),
            }
        } else {
            (0, upper)
        }
    }
}

impl Partition for Empty {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        self.partition_idx
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
    use std::ops::Range;
    use std::sync::Arc;

    use futures::StreamExt;
    use futures::TryStreamExt;
    use rstest::rstest;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::Expression;
    use vortex_array::expr::eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::pack;
    use vortex_array::expr::root;
    use vortex_array::expr::stats::Precision;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
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
    #[expect(
        clippy::reversed_empty_ranges,
        reason = "the inverted range is the input"
    )]
    fn inverted_row_range_selects_no_rows() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        let source = LayoutReaderDataSource::new(Arc::new(TestLayoutReader::new(8)), session);
        // The filter reaches the pruning check, which needs the range's length.
        let scan = runtime.block_on(source.scan(ScanRequest {
            filter: Some(eq(root(), lit(1i32))),
            row_range: Some(5..3),
            ..Default::default()
        }))?;
        let mut values = Vec::new();
        for partition in runtime.block_on(scan.partitions().try_collect::<Vec<_>>())? {
            values.extend(collect_scan_values(
                runtime.block_on_stream(partition.execute()?),
            )?);
        }
        assert!(values.is_empty(), "{values:?}");
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
    #[test]
    fn zero_split_max_row_count_is_rejected() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let source = LayoutReaderDataSource::new(
            Arc::new(TestLayoutReader::new(10)),
            session_with_handle(runtime.handle()),
        )
        .with_split_max_row_count(0);

        let err = runtime
            .block_on(source.scan(ScanRequest::default()))
            .err()
            .expect("zero split_max_row_count should fail");
        assert!(
            err.to_string()
                .contains("split_max_row_count must be greater than zero")
        );

        Ok(())
    }

    fn empty_projection() -> Expression {
        pack(
            std::iter::empty::<(&str, Expression)>(),
            Nullability::NonNullable,
        )
    }

    fn index_selection(indices: &[u64], include: bool) -> VortexResult<Selection> {
        let buffer = StrictSortedBuffer::try_new(Buffer::from_iter(indices.iter().copied()))?;
        Ok(if include {
            Selection::IncludeByIndex(buffer)
        } else {
            Selection::ExcludeByIndex(buffer)
        })
    }

    fn roaring_selection(indices: &[u64], include: bool) -> VortexResult<Selection> {
        let mut map = roaring::RoaringTreemap::new();
        for &idx in indices {
            map.insert(idx);
        }
        Ok(if include {
            Selection::IncludeRoaring(map)
        } else {
            Selection::ExcludeRoaring(map)
        })
    }

    #[rstest]
    #[case::exclude_zero_with_limit(
        None,
        None,
        index_selection(&[0], false),
        Some(2),
        vec![1],
        vec![2, 3],
        Precision::inexact(2usize),
    )]
    #[case::empty_include(
        None,
        None,
        index_selection(&[], true),
        None,
        vec![],
        vec![],
        Precision::exact(0usize),
    )]
    #[case::roaring_include_with_partition_range(
        None,
        Some(1..3),
        roaring_selection(&[0, 2], true),
        None,
        vec![2],
        vec![4, 5],
        Precision::exact(1usize),
    )]
    #[case::roaring_exclude_with_partition_range(
        None,
        Some(0..3),
        roaring_selection(&[1], false),
        None,
        vec![0, 2],
        vec![0, 1, 4, 5],
        Precision::exact(2usize),
    )]
    #[case::row_range_preserves_split_grid_ids(
        Some(2..6),
        None,
        index_selection(&[1], true),
        None,
        vec![1],
        vec![2, 3],
        Precision::exact(1usize),
    )]
    fn single_file_partition_selectors_and_empty_projection_match(
        #[case] row_range: Option<Range<u64>>,
        #[case] partition_range: Option<Range<u64>>,
        #[case] partition_selection: VortexResult<Selection>,
        #[case] limit: Option<u64>,
        #[case] expected_ids: Vec<usize>,
        #[case] expected_values: Vec<i32>,
        #[case] expected_partition_count: Precision<usize>,
    ) -> VortexResult<()> {
        let partition_selection = partition_selection?;
        let runtime = SingleThreadRuntime::default();
        let session = session_with_handle(runtime.handle());
        // 6 rows split into 3 partitions of 2 rows each:
        // partition 0: [0, 1], partition 1: [2, 3], partition 2: [4, 5].
        let source = LayoutReaderDataSource::new(Arc::new(TestLayoutReader::new(6)), session)
            .with_split_max_row_count(2);

        for (is_empty_proj, projection) in [(false, root()), (true, empty_projection())] {
            let scan = runtime.block_on(source.scan(ScanRequest {
                projection,
                row_range: row_range.clone(),
                partition_range: partition_range.clone(),
                partition_selection: partition_selection.clone(),
                limit,
                ..Default::default()
            }))?;
            assert_eq!(scan.partition_count(), expected_partition_count);

            let parts = runtime.block_on(scan.partitions().try_collect::<Vec<_>>())?;
            let ids: Vec<usize> = parts.iter().map(|p| p.index()).collect();
            assert_eq!(ids, expected_ids);

            if is_empty_proj {
                let mut rows = 0usize;
                for part in parts {
                    for array in runtime.block_on(part.execute()?.try_collect::<Vec<_>>())? {
                        rows += array.len();
                    }
                }
                assert_eq!(rows, expected_values.len());
            } else {
                let mut values = Vec::new();
                for part in parts {
                    values.extend(collect_scan_values(
                        runtime.block_on_stream(part.execute()?),
                    )?);
                }
                assert_eq!(values, expected_values);
            }
        }

        Ok(())
    }

    #[test]
    fn empty_projection_single_split_reports_exact_partition_count() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let source = LayoutReaderDataSource::new(
            Arc::new(TestLayoutReader::new(6)),
            session_with_handle(runtime.handle()),
        );
        let scan = runtime.block_on(source.scan(ScanRequest {
            projection: empty_projection(),
            ..Default::default()
        }))?;
        assert_eq!(scan.partition_count(), Precision::exact(1usize));
        Ok(())
    }

    #[test]
    fn empty_projection_discovery_and_replay_match() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let source = LayoutReaderDataSource::new(
            Arc::new(TestLayoutReader::new(6)),
            session_with_handle(runtime.handle()),
        )
        .with_split_max_row_count(2);

        let discovered = runtime.block_on(source.scan(ScanRequest {
            projection: empty_projection(),
            ..Default::default()
        }))?;
        assert_eq!(discovered.partition_count(), Precision::exact(3usize));
        let discovered_parts = runtime.block_on(discovered.partitions().try_collect::<Vec<_>>())?;
        let discovered_ids: Vec<usize> = discovered_parts.iter().map(|p| p.index()).collect();
        assert_eq!(discovered_ids, vec![0, 1, 2]);

        let mut replayed_rows = 0usize;
        for id in discovered_ids {
            let replay = runtime.block_on(source.scan(ScanRequest {
                projection: empty_projection(),
                partition_selection: Selection::IncludeByIndex(StrictSortedBuffer::try_new(
                    Buffer::from_iter([id as u64]),
                )?),
                ..Default::default()
            }))?;
            assert_eq!(replay.partition_count(), Precision::exact(1usize));
            for part in runtime.block_on(replay.partitions().try_collect::<Vec<_>>())? {
                for array in runtime.block_on(part.execute()?.try_collect::<Vec<_>>())? {
                    replayed_rows += array.len();
                }
            }
        }
        assert_eq!(replayed_rows, 6);
        Ok(())
    }

    #[test]
    fn row_range_beyond_file_extent_is_clamped_before_cursor() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let source = LayoutReaderDataSource::new(
            Arc::new(TestLayoutReader::new(6)),
            session_with_handle(runtime.handle()),
        )
        .with_split_max_row_count(2);

        for proj in [empty_projection(), root()] {
            let nonexistent = runtime.block_on(source.scan(ScanRequest {
                projection: proj.clone(),
                row_range: Some(0..8),
                partition_range: Some(3..4),
                ..Default::default()
            }))?;
            assert_eq!(nonexistent.partition_count(), Precision::exact(0usize));
            assert!(
                runtime
                    .block_on(nonexistent.partitions().try_collect::<Vec<_>>())?
                    .is_empty()
            );

            let overlapping_end = runtime.block_on(source.scan(ScanRequest {
                projection: proj,
                row_range: Some(3..8),
                partition_range: Some(2..4),
                ..Default::default()
            }))?;
            assert_eq!(overlapping_end.partition_count(), Precision::exact(1usize));
            let parts = runtime.block_on(overlapping_end.partitions().try_collect::<Vec<_>>())?;
            let [part]: [_; 1] = parts
                .try_into()
                .map_err(|p: Vec<_>| vortex_err!("expected 1 partition, got {}", p.len()))?;
            assert_eq!(part.index(), 2);
            let rows: usize = runtime
                .block_on(part.execute()?.try_collect::<Vec<_>>())?
                .iter()
                .map(|a| a.len())
                .sum();
            assert_eq!(rows, 2);
        }

        Ok(())
    }
}
