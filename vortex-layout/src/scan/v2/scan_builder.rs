// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::ready;

use futures::Stream;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::iter::ArrayIterator;
use vortex_array::iter::ArrayIteratorAdapter;
use vortex_array::stats::StatsSet;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;
use vortex_io::session::RuntimeSessionExt;
use vortex_metrics::MetricsRegistry;
use vortex_scan::selection::Selection;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::layouts::row_idx::RowIdx;
use crate::layouts::row_idx::RowIdxLayoutReader;
use crate::scan::scan_builder;
use crate::scan::scan_builder::referenced_field_masks;
use crate::scan::split_by::SplitBy;
use crate::scan::splits::Splits;
use crate::scan::splits::attempt_split_ranges;
use crate::scan::v2::RepeatedScanV2;

/// Alternative builder for scanning a [`LayoutReader`] into arrays, streams, or iterators.
///
/// This copy initially uses the same layout-reader execution as the default builder, so the new
/// scan implementation can be introduced incrementally behind [`enabled`](super::enabled).
///
/// A scan has three independent row restriction mechanisms:
///
/// - [`with_row_range`](Self::with_row_range) selects a contiguous range before scanning.
/// - [`with_selection`](Self::with_selection) applies a [`Selection`] inside that range.
/// - [`with_filter`](Self::with_filter) evaluates an expression predicate during execution.
///
/// Projection and filter expressions must be bound against the reader dtype. Work is divided by
/// the configured [`SplitBy`] strategy or by explicit selection ranges.
pub struct ScanBuilder<A> {
    session: VortexSession,
    layout_reader: LayoutReaderRef,
    projection: BoundExpression,
    filter: Option<BoundExpression>,
    /// Whether the scan needs to return splits in the order they appear in the file.
    ordered: bool,
    /// Optionally read a subset of the rows in the file.
    row_range: Option<Range<u64>>,
    /// The selection mask to apply to the selected row range.
    selection: Selection,
    /// How to split the file for concurrent processing.
    split_by: SplitBy,
    /// Precomputed full-file natural split boundaries; when set, [`prepare`](Self::prepare)
    /// uses them instead of walking the layout.
    natural_splits: Option<Arc<[u64]>>,
    /// The number of splits to make progress on concurrently **per-thread**.
    concurrency: usize,
    /// Function to apply to each [`ArrayRef`] within the spawned split tasks.
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,
    /// Should we try to prune the file (using stats) on open.
    file_stats: Option<Arc<[StatsSet]>>,
    /// Maximal number of rows to read (after filtering)
    limit: Option<u64>,
    /// The row-offset assigned to the first row of the file. Used by the `row_idx` expression,
    /// but not by the scan [`Selection`] which remains relative.
    row_offset: u64,
}

impl ScanBuilder<ArrayRef> {
    /// Create a scan builder over `layout_reader` using `session` for runtime and execution state.
    pub fn new(session: VortexSession, layout_reader: Arc<dyn LayoutReader>) -> Self {
        let projection = BoundExpression::new_root(layout_reader.dtype().clone());
        Self {
            session,
            layout_reader,
            projection,
            filter: None,
            ordered: true,
            row_range: None,
            selection: Default::default(),
            split_by: SplitBy::default(),
            natural_splits: None,
            // We default to four tasks per worker thread, which allows for some I/O lookahead
            // without too much impact on work-stealing.
            concurrency: 4,
            map_fn: Arc::new(Ok),
            metrics_registry: None,
            file_stats: None,
            limit: None,
            row_offset: 0,
        }
    }

    /// Returns an [`ArrayStream`] with tasks spawned onto the session's runtime handle.
    ///
    /// See [`ScanBuilder::into_stream`] for more details.
    pub fn into_array_stream(self) -> VortexResult<impl ArrayStream + Send + 'static> {
        let dtype = self.dtype()?;
        let stream = self.into_stream()?;
        Ok(ArrayStreamAdapter::new(dtype, stream))
    }

    /// Returns an [`ArrayIterator`] using the given blocking runtime.
    pub fn into_array_iter<B: BlockingRuntime>(
        self,
        runtime: &B,
    ) -> VortexResult<impl ArrayIterator + 'static> {
        let stream = self.into_array_stream()?;
        let dtype = stream.dtype().clone();
        Ok(ArrayIteratorAdapter::new(
            dtype,
            runtime.block_on_stream(stream),
        ))
    }
}

impl<A: 'static + Send> ScanBuilder<A> {
    /// Moves every option from a default builder into this executor.
    pub fn from_default(builder: scan_builder::ScanBuilder<A>) -> Self {
        let parts = builder.into_parts();
        Self {
            session: parts.session,
            layout_reader: parts.layout_reader,
            projection: parts.projection,
            filter: parts.filter,
            ordered: parts.ordered,
            row_range: parts.row_range,
            selection: parts.selection,
            split_by: parts.split_by,
            natural_splits: parts.natural_splits,
            concurrency: parts.concurrency,
            map_fn: parts.map_fn,
            metrics_registry: parts.metrics_registry,
            file_stats: parts.file_stats,
            limit: parts.limit,
            row_offset: parts.row_offset,
        }
    }

    /// Add a filter expression bound against the reader dtype.
    pub fn with_filter(mut self, filter: BoundExpression) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Add or clear a filter expression bound against the reader dtype.
    pub fn with_some_filter(mut self, filter: Option<BoundExpression>) -> Self {
        self.filter = filter;
        self
    }

    /// Set a projection expression bound against the reader dtype.
    pub fn with_projection(mut self, projection: BoundExpression) -> Self {
        self.projection = projection;
        self
    }

    /// Returns whether output chunks are yielded in file order.
    pub fn ordered(&self) -> bool {
        self.ordered
    }

    /// Configure whether output chunks must be yielded in file order.
    pub fn with_ordered(mut self, ordered: bool) -> Self {
        self.ordered = ordered;
        self
    }

    /// Restrict scanning to a contiguous row range.
    pub fn with_row_range(mut self, row_range: Range<u64>) -> Self {
        self.row_range = Some(row_range);
        self
    }

    /// Apply a row selection to the selected row range.
    pub fn with_selection(mut self, selection: Selection) -> Self {
        self.selection = selection;
        self
    }

    /// Select rows by strictly sorted absolute indices relative to the scan input.
    pub fn with_row_indices(mut self, row_indices: StrictSortedBuffer<u64>) -> Self {
        self.selection = Selection::IncludeByIndex(row_indices);
        self
    }

    /// Set the root row offset used by row-index expressions.
    pub fn with_row_offset(mut self, row_offset: u64) -> Self {
        self.row_offset = row_offset;
        self
    }

    /// Configure how natural scan work is split for concurrency.
    pub fn with_split_by(mut self, split_by: SplitBy) -> Self {
        self.split_by = split_by;
        self
    }

    /// Supply precomputed full-file natural split boundaries (see
    /// [`full_file_splits`](Self::full_file_splits)) so [`prepare`](Self::prepare) reuses them
    /// instead of walking the layout. Callers translating external partitions into row ranges
    /// can compute the boundaries once per file and share them across partitions.
    ///
    /// Takes precedence over [`with_split_by`](Self::with_split_by); boundaries outside the
    /// scan's row range are clamped during execution. Boundaries must be strictly increasing.
    pub fn with_natural_splits(mut self, boundaries: Arc<[u64]>) -> Self {
        debug_assert!(
            boundaries.windows(2).all(|w| w[0] < w[1]),
            "natural split boundaries must be strictly increasing"
        );
        self.natural_splits = Some(boundaries);
        self
    }

    /// Compute the full-file natural split boundaries for the fields referenced by this scan's
    /// projection and filter, ignoring any configured row range.
    ///
    /// These are the boundaries [`prepare`](Self::prepare) derives for a whole-file scan; hand
    /// them back via [`with_natural_splits`](Self::with_natural_splits) to skip the layout walk
    /// in `prepare`.
    pub fn full_file_splits(&self) -> VortexResult<Vec<u64>> {
        let field_mask = referenced_field_masks(&self.projection, self.filter.as_ref())?;
        self.split_by.splits(
            self.layout_reader.as_ref(),
            &(0..self.layout_reader.row_count()),
            &field_mask,
        )
    }

    /// Returns the per-worker row-split concurrency.
    pub fn concurrency(&self) -> usize {
        self.concurrency
    }

    /// The number of row splits to make progress on concurrently per-thread, must
    /// be greater than 0.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        assert!(concurrency > 0);
        self.concurrency = concurrency;
        self
    }

    /// Add or clear the metrics registry used by scan execution.
    pub fn with_some_metrics_registry(mut self, metrics: Option<Arc<dyn MetricsRegistry>>) -> Self {
        self.metrics_registry = metrics;
        self
    }

    /// Set the metrics registry used by scan execution.
    pub fn with_metrics_registry(mut self, metrics: Arc<dyn MetricsRegistry>) -> Self {
        self.metrics_registry = Some(metrics);
        self
    }

    /// Add or clear the maximum number of rows returned after filtering.
    pub fn with_some_limit(mut self, limit: Option<u64>) -> Self {
        self.limit = limit;
        self
    }

    /// Set the maximum number of rows returned after filtering.
    pub fn with_limit(mut self, limit: u64) -> Self {
        self.limit = Some(limit);
        self
    }

    /// The [`DType`] returned by the scan, after applying the projection.
    pub fn dtype(&self) -> VortexResult<DType> {
        Ok(self.projection.dtype().clone())
    }

    /// The session used by the scan.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// Map each split of the scan. The function will be run on the spawned task.
    pub fn map<B: 'static>(
        self,
        map_fn: impl Fn(A) -> VortexResult<B> + 'static + Send + Sync,
    ) -> ScanBuilder<B> {
        let old_map_fn = self.map_fn;
        ScanBuilder {
            session: self.session,
            layout_reader: self.layout_reader,
            projection: self.projection,
            filter: self.filter,
            ordered: self.ordered,
            row_range: self.row_range,
            selection: self.selection,
            split_by: self.split_by,
            natural_splits: self.natural_splits,
            concurrency: self.concurrency,
            metrics_registry: self.metrics_registry,
            file_stats: self.file_stats,
            limit: self.limit,
            row_offset: self.row_offset,
            map_fn: Arc::new(move |a| old_map_fn(a).and_then(&map_fn)),
        }
    }

    /// Optimize expressions, compute split ranges, and return an executable repeated scan.
    pub fn prepare(self) -> VortexResult<RepeatedScanV2<A>> {
        let dtype = self.dtype()?;

        if self.filter.is_some() && self.limit.is_some() {
            vortex_bail!("Vortex doesn't support scans with both a filter and a limit")
        }

        let mut layout_reader = self.layout_reader;

        // Row-index expressions need the scan offset applied by the reader.
        let mut found_row_idx = self.projection.contains::<RowIdx>()?;
        if !found_row_idx && let Some(filter) = self.filter.as_ref() {
            found_row_idx = filter.contains::<RowIdx>()?;
        }
        if found_row_idx {
            layout_reader = Arc::new(RowIdxLayoutReader::new(
                self.row_offset,
                layout_reader,
                self.session.clone(),
            ));
        }

        let bound_projection = self.projection;
        let bound_filter = self.filter;

        let splits =
            if let Some(ranges) = attempt_split_ranges(&self.selection, self.row_range.as_ref()) {
                Splits::Ranges(ranges)
            } else if let Some(boundaries) = self.natural_splits {
                // Caller-supplied full-file boundaries; execution clamps them to the row range.
                Splits::Natural(boundaries)
            } else {
                let field_mask = referenced_field_masks(&bound_projection, bound_filter.as_ref())?;
                let split_range = self
                    .row_range
                    .clone()
                    .unwrap_or_else(|| 0..layout_reader.row_count());
                Splits::Natural(
                    self.split_by
                        .splits(layout_reader.as_ref(), &split_range, &field_mask)?
                        .into(),
                )
            };

        Ok(RepeatedScanV2::new(
            self.session.clone(),
            layout_reader,
            bound_projection,
            bound_filter,
            self.ordered,
            self.row_range,
            self.selection,
            splits,
            self.concurrency,
            self.map_fn,
            self.limit,
            dtype,
        ))
    }

    /// Constructs a task per row split of the scan, returned as a vector of futures.
    pub fn build(self) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        if self.limit.is_some_and(|l| l == 0) {
            return Ok(vec![]);
        }

        self.prepare()?.execute(None)
    }

    /// Returns a [`Stream`] with tasks spawned onto the session's runtime handle.
    pub fn into_stream(
        self,
    ) -> VortexResult<impl Stream<Item = VortexResult<A>> + Send + 'static + use<A>> {
        Ok(LazyScanStream::new(self))
    }

    /// Returns an [`Iterator`] using the session's runtime.
    pub fn into_iter<B: BlockingRuntime>(
        self,
        runtime: &B,
    ) -> VortexResult<impl Iterator<Item = VortexResult<A>> + 'static> {
        let stream = self.into_stream()?;
        Ok(runtime.block_on_stream(stream))
    }
}

enum LazyScanState<A: 'static + Send> {
    Builder(Option<Box<ScanBuilder<A>>>),
    Preparing(PreparingScan<A>),
    Stream(BoxStream<'static, VortexResult<A>>),
    Error(Option<vortex_error::VortexError>),
}

type PreparedScanTasks<A> = Vec<BoxFuture<'static, VortexResult<Option<A>>>>;

struct PreparingScan<A: 'static + Send> {
    ordered: bool,
    concurrency: usize,
    handle: Handle,
    task: Task<VortexResult<PreparedScanTasks<A>>>,
}

struct LazyScanStream<A: 'static + Send> {
    state: LazyScanState<A>,
}

impl<A: 'static + Send> LazyScanStream<A> {
    fn new(builder: ScanBuilder<A>) -> Self {
        Self {
            state: LazyScanState::Builder(Some(Box::new(builder))),
        }
    }
}

impl<A: 'static + Send> Unpin for LazyScanStream<A> {}

impl<A: 'static + Send> Stream for LazyScanStream<A> {
    type Item = VortexResult<A>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                LazyScanState::Builder(builder) => {
                    let builder = builder.take().vortex_expect("polled after completion");
                    let ordered = builder.ordered;
                    let num_workers = get_available_parallelism().unwrap_or(1);
                    let concurrency = builder.concurrency * num_workers;
                    let handle = builder.session.handle();
                    let task = handle
                        .spawn_cpu(move || builder.prepare().and_then(|scan| scan.execute(None)));
                    self.state = LazyScanState::Preparing(PreparingScan {
                        ordered,
                        concurrency,
                        handle,
                        task,
                    });
                }
                LazyScanState::Preparing(preparing) => {
                    match ready!(Pin::new(&mut preparing.task).poll(cx)) {
                        Ok(tasks) => {
                            let ordered = preparing.ordered;
                            let concurrency = preparing.concurrency;
                            let handle = preparing.handle.clone();
                            let stream =
                                futures::stream::iter(tasks).map(move |task| handle.spawn(task));
                            let stream = if ordered {
                                stream.buffered(concurrency).boxed()
                            } else {
                                stream.buffer_unordered(concurrency).boxed()
                            };
                            let stream = stream
                                .filter_map(|chunk| async move { chunk.transpose() })
                                .boxed();
                            self.state = LazyScanState::Stream(stream);
                        }
                        Err(err) => self.state = LazyScanState::Error(Some(err)),
                    }
                }
                LazyScanState::Stream(stream) => return stream.as_mut().poll_next(cx),
                LazyScanState::Error(err) => return Poll::Ready(err.take().map(Err)),
            }
        }
    }
}
