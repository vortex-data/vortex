// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::iter;
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
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
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
use vortex_io::session::RuntimeSessionExt;
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
use vortex_scan::stream::ScanStream;
use vortex_session::VortexSession;

use crate::LayoutReaderRef;
use crate::scan::limit::RowLimit;
use crate::scan::metrics::ScanMetrics;
use crate::scan::metrics::ScanRecorder;
use crate::scan::repeated_scan::DEFAULT_SPLITS_PER_THREAD;
use crate::scan::scan_builder::ScanBuilder;
use crate::scan::scan_builder::referenced_field_masks;
use crate::scan::split_by::SplitBy;
use crate::scan::split_queue::SplitQueue;

/// An implementation of a [`DataSource`] that reads data from a [`LayoutReaderRef`].
pub struct LayoutReaderDataSource {
    reader: LayoutReaderRef,
    session: VortexSession,
    split_max_row_count: u64,
    metrics: Option<ScanMetrics>,
    /// The whole-source cut points computed by [`DataSource::cut_points`], keyed by the fields
    /// whose layout decides them. Scans whose fields match reuse them instead of walking the
    /// layout again.
    cut_points: Mutex<Vec<(Vec<FieldMask>, Arc<[u64]>)>>,
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
            cut_points: Mutex::new(Vec::new()),
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

    /// Adds this source's scans to counters registered elsewhere, without new registry entries.
    ///
    /// A host that opens one source per file registers the counters once with
    /// [`ScanMetrics::new`] and shares them across its sources.
    pub fn with_scan_metrics(mut self, metrics: ScanMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// The cached whole-source cut points for `field_mask`, if [`DataSource::cut_points`]
    /// computed them.
    fn cached_cut_points(&self, field_mask: &[FieldMask]) -> Option<Arc<[u64]>> {
        self.cut_points
            .lock()
            .iter()
            .find(|(mask, _)| mask.as_slice() == field_mask)
            .map(|(_, boundaries)| Arc::clone(boundaries))
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
        let natural_splits =
            self.cached_cut_points(&referenced_field_masks(&projection, filter.as_ref())?);

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
            remaining_limit: scan_request.limit,
            row_limit,
            selection: scan_request.selection,
            ordered: scan_request.ordered,
            split_concurrency: scan_request.split_concurrency,
            metrics: self.metrics.clone(),
            natural_splits,
            next_row: row_range.start,
            end_row: row_range.end,
            split_size: self.split_max_row_count,
        }))
    }

    /// The layout's split boundaries for the fields the request reads, as a scan of the whole
    /// source would compute them, clamped to the request's row range.
    ///
    /// The whole-source boundaries are computed once for each set of fields and cached, and later
    /// scans of this source that read the same fields cut at them instead of walking the layout.
    async fn cut_points(&self, scan_request: &ScanRequest) -> VortexResult<Option<Arc<[u64]>>> {
        let projection = scan_request
            .projection
            .bind(self.reader.dtype())?
            .optimize_recursive()?;
        let filter = scan_request
            .filter
            .as_ref()
            .map(|expr| expr.bind(self.reader.dtype())?.optimize_recursive())
            .transpose()?;
        let field_mask = referenced_field_masks(&projection, filter.as_ref())?;

        // Compute under the lock, so that concurrent callers wait for one layout walk instead of
        // each contending on the layout's lazily initialized children.
        let boundaries = {
            let mut cache = self.cut_points.lock();
            let cached = cache
                .iter()
                .find(|(mask, _)| *mask == field_mask)
                .map(|(_, boundaries)| Arc::clone(boundaries));
            match cached {
                Some(boundaries) => boundaries,
                None => {
                    let boundaries: Arc<[u64]> = SplitBy::default()
                        .splits(
                            self.reader.as_ref(),
                            &(0..self.reader.row_count()),
                            &field_mask,
                        )?
                        .into();
                    cache.push((field_mask, Arc::clone(&boundaries)));
                    boundaries
                }
            }
        };

        let Some(row_range) = &scan_request.row_range else {
            return Ok(Some(boundaries));
        };
        if row_range.start >= row_range.end {
            return Ok(Some(Arc::from([row_range.start])));
        }
        Ok(Some(
            iter::once(row_range.start)
                .chain(
                    boundaries
                        .iter()
                        .copied()
                        .filter(|&row| row > row_range.start && row < row_range.end),
                )
                .chain(iter::once(row_range.end))
                .collect(),
        ))
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
    /// The source's cached whole-source cut points for this scan's fields, if any.
    natural_splits: Option<Arc<[u64]>>,
    next_row: u64,
    end_row: u64,
    split_size: u64,
}

impl LayoutReaderScan {
    /// A scan builder over this scan's reader, with its selection, projection, filter, metrics
    /// and split concurrency. Callers give it the row range to read.
    fn builder(&self) -> ScanBuilder<ArrayRef> {
        let builder = ScanBuilder::new(self.session.clone(), Arc::clone(&self.reader))
            .with_selection(self.selection.clone())
            .with_projection(self.projection.clone())
            .with_some_filter(self.filter.clone())
            .with_some_scan_metrics(self.metrics.clone())
            .with_some_max_splits_in_flight(self.split_concurrency);
        match &self.natural_splits {
            Some(boundaries) => builder.with_natural_splits(Arc::clone(boundaries)),
            None => builder,
        }
    }

    /// This scan over `rows` only. Only a scan without a limit is cut into pieces, so a piece
    /// has no limit to share.
    fn piece(&self, rows: Range<u64>) -> Self {
        Self {
            reader: Arc::clone(&self.reader),
            session: self.session.clone(),
            dtype: self.dtype.clone(),
            projection: self.projection.clone(),
            filter: self.filter.clone(),
            remaining_limit: None,
            row_limit: None,
            ordered: self.ordered,
            split_concurrency: self.split_concurrency,
            selection: self.selection.clone(),
            metrics: self.metrics.clone(),
            natural_splits: self.natural_splits.clone(),
            next_row: rows.start,
            end_row: rows.end,
            split_size: self.split_size,
        }
    }
}

impl DataSourceScan for LayoutReaderScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Cuts the scan at the layout's natural split boundaries, into contiguous pieces of about
    /// equal rows.
    ///
    /// A scan with a limit stays whole, because the limit is one budget across all its rows.
    fn split(self: Box<Self>, n: NonZeroUsize) -> VortexResult<Vec<DataSourceScanRef>> {
        if n.get() == 1 || self.remaining_limit.is_some() || self.next_row >= self.end_row {
            return Ok(vec![self]);
        }
        let ranges = self
            .builder()
            .prepare()?
            .split_ranges(Some(self.next_row..self.end_row));
        let pieces: Vec<Range<u64>> = group_contiguous(ranges, n)
            .into_iter()
            .filter_map(|group| Some(group.first()?.start..group.last()?.end))
            .collect();
        if pieces.len() <= 1 {
            return Ok(vec![self]);
        }
        Ok(pieces
            .into_iter()
            .map(|rows| Box::new(self.piece(rows)) as DataSourceScanRef)
            .collect())
    }

    /// Opens the scan into a stream over a [`SplitQueue`] of its split futures, each clone of the
    /// stream keeping the stream pipeline's per-thread count of them in flight.
    ///
    /// Split futures cannot share a limit, and the clones of a queue finish splits out of row
    /// order. So a scan with a limit, or an ordered one, opens into its partition pipeline and
    /// hands its partitions to the clones.
    fn open(self: Box<Self>) -> VortexResult<ScanStream> {
        if self.remaining_limit.is_some() || self.ordered {
            return Ok(ScanStream::from_partitions(self));
        }
        // A cap on the scan's splits in flight also caps each clone's.
        let per_worker = self
            .split_concurrency
            .map_or(DEFAULT_SPLITS_PER_THREAD, |max| {
                max.get().min(DEFAULT_SPLITS_PER_THREAD)
            });
        let splits = self
            .builder()
            .prepare()?
            .execute(Some(self.next_row..self.end_row))?;
        let queue = Arc::new(SplitQueue::new(self.session.handle(), per_worker, splits));
        let progress = Arc::clone(&queue);
        Ok(
            ScanStream::new(self.dtype, move || Arc::clone(&queue).worker())
                .with_progress(move || progress.progress()),
        )
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
            natural_splits: this.natural_splits.clone(),
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
    natural_splits: Option<Arc<[u64]>>,
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
        let builder = match self.natural_splits {
            Some(boundaries) => builder.with_natural_splits(boundaries),
            None => builder,
        };
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
}

impl DataSourceScan for Empty {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn split(self: Box<Self>, _n: NonZeroUsize) -> VortexResult<Vec<DataSourceScanRef>> {
        Ok(vec![self])
    }

    fn open(self: Box<Self>) -> VortexResult<ScanStream> {
        Ok(ScanStream::from_partitions(self))
    }

    fn partition_count(&self) -> Precision<usize> {
        Precision::exact(1usize)
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        stream::iter([Ok(self as _)]).boxed()
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
mod split_open_tests {
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use futures::TryStreamExt;
    use futures::future::try_join_all;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_io::runtime::BlockingRuntime;
    use vortex_io::runtime::single::SingleThreadRuntime;
    use vortex_scan::DataSource;
    use vortex_scan::ScanRequest;

    use super::LayoutReaderDataSource;
    use super::group_contiguous;
    use crate::scan::test::TestLayoutReader;
    use crate::scan::test::collect_scan_values;
    use crate::scan::test::session_with_handle;

    fn source(runtime: &SingleThreadRuntime) -> LayoutReaderDataSource {
        LayoutReaderDataSource::new(
            Arc::new(TestLayoutReader::new(64).with_split_size(4)),
            session_with_handle(runtime.handle()),
        )
    }

    /// Clones of an opened scan's stream each claim splits until none is left, so together they
    /// read every row once.
    #[test]
    fn opened_clones_read_every_row_once() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let stream = runtime
            .block_on(source(&runtime).scan(ScanRequest::default()))?
            .open()?;
        let workers = (0..3).map(|_| stream.clone().try_collect::<Vec<_>>());
        let arrays = runtime.block_on(try_join_all(workers))?;
        let mut values = collect_scan_values(arrays.into_iter().flatten().map(Ok))?;
        values.sort_unstable();
        assert_eq!(values, (0..64).collect::<Vec<_>>());
        assert_eq!(stream.progress(), Some(1.0));
        Ok(())
    }

    /// The pieces of an unordered scan are fixed too: each owns a contiguous run of splits,
    /// which it may return in any order.
    #[test]
    fn unordered_pieces_own_their_rows() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let scan = runtime.block_on(source(&runtime).scan(ScanRequest::default()))?;
        let pieces = scan.split(NonZeroUsize::try_from(4)?)?;
        assert_eq!(pieces.len(), 4);
        for (i, piece) in (0i32..).zip(pieces) {
            let mut values = collect_scan_values(runtime.block_on_stream(piece.open()?))?;
            values.sort_unstable();
            assert_eq!(values, (16 * i..16 * (i + 1)).collect::<Vec<_>>());
        }
        Ok(())
    }

    /// An ordered scan is cut at the layout's split boundaries, and the pieces read the rows in
    /// order.
    #[test]
    fn ordered_split_cuts_at_split_boundaries() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let scan = runtime.block_on(source(&runtime).scan(ScanRequest {
            ordered: true,
            ..Default::default()
        }))?;
        let pieces = scan.split(NonZeroUsize::try_from(4)?)?;
        assert_eq!(pieces.len(), 4);
        let mut values = Vec::new();
        for piece in pieces {
            values.extend(collect_scan_values(runtime.block_on_stream(piece.open()?))?);
        }
        assert_eq!(values, (0..64).collect::<Vec<_>>());
        Ok(())
    }

    /// A limit is one budget across the scan, so a limited scan stays whole and opens into its
    /// partitions, which apply the limit.
    #[test]
    fn limited_scan_stays_whole() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let source = LayoutReaderDataSource::new(
            Arc::new(TestLayoutReader::new(64).with_split_size(4)),
            session_with_handle(runtime.handle()),
        );
        let request = || ScanRequest {
            limit: Some(10),
            ..Default::default()
        };
        let scan = runtime.block_on(source.scan(request()))?;
        assert_eq!(scan.split(NonZeroUsize::try_from(4)?)?.len(), 1);

        let stream = runtime.block_on(source.scan(request()))?.open()?;
        let values = collect_scan_values(runtime.block_on_stream(stream))?;
        assert_eq!(values.len(), 10);
        Ok(())
    }

    /// Cut points are the layout's split boundaries clamped to the request's row range, and
    /// scans cut at row ranges between them read each row once.
    #[test]
    fn cut_points_tile_the_rows() -> VortexResult<()> {
        let runtime = SingleThreadRuntime::default();
        let source = source(&runtime);
        let whole = runtime
            .block_on(source.cut_points(&ScanRequest::default()))?
            .vortex_expect("a layout source has cut points");
        assert_eq!(
            whole.as_ref(),
            (0..=16u64).map(|i| i * 4).collect::<Vec<_>>()
        );

        let clamped = runtime
            .block_on(source.cut_points(&ScanRequest {
                row_range: Some(6..18),
                ..Default::default()
            }))?
            .vortex_expect("a layout source has cut points");
        assert_eq!(clamped.as_ref(), [6, 8, 12, 16, 18]);

        let mut values = Vec::new();
        let cuts = whole.iter().copied().step_by(4).collect::<Vec<_>>();
        for rows in cuts.windows(2).map(|pair| pair[0]..pair[1]) {
            let stream = runtime
                .block_on(source.scan(ScanRequest {
                    row_range: Some(rows),
                    ordered: true,
                    ..Default::default()
                }))?
                .open()?;
            values.extend(collect_scan_values(runtime.block_on_stream(stream))?);
        }
        assert_eq!(values, (0..64).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn group_contiguous_keeps_ranges_whole() -> VortexResult<()> {
        // Each range joins the group its first row falls in.
        let ranges = vec![0..10, 10..16, 16..26, 26..32];
        let groups = group_contiguous(ranges, NonZeroUsize::try_from(2)?);
        assert_eq!(groups, vec![vec![0..10, 10..16], vec![16..26, 26..32]]);
        Ok(())
    }
}
